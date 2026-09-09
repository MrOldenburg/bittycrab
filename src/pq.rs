// Product Quantization for the tied embedding table's scan-only candidate
// filter. Same architecture discipline as the int8 path: PQ is only ever
// used to cheaply narrow the full vocab down to a handful of candidates;
// the final answer always comes from an exact F16 rescore of those
// candidates, so PQ's approximation error can never produce a wrong final
// token by itself -- only a worse candidate set (which the harness below
// checks empirically, the same way every other quantization step here was
// validated, not assumed).
//
// PQ splits each D_MODEL-dim row into M sub-vectors and replaces each
// sub-vector with the index of its nearest of K trained centroids (1
// byte/subvector). Query time uses ADC (asymmetric distance computation):
// build one per-query M*K lookup table (dot of each query sub-vector
// against every centroid in that subspace), then every row's score is just
// M table reads + adds -- no per-row multiplies at all, unlike int8's VNNI
// dot product.
//
// Memory: codebooks are K*D_MODEL floats total regardless of how D_MODEL
// is split into M subspaces (M*K*(D_MODEL/M) = K*D_MODEL), so M mainly
// trades code bytes/row (M bytes) for quantization fineness (smaller
// subspaces cluster more accurately). Codes: vocab*M bytes.

use crate::gguf::{f16_bits_to_f32, Gguf};
use crate::model::D_MODEL;
use std::fs::File;
use std::os::unix::fs::FileExt;

pub struct PqTable {
    pub m: usize,
    pub sub_dim: usize,
    pub k: usize,
    pub codebooks: Vec<f32>, // [m][k][sub_dim], trained on UNIT-NORMALIZED rows
    pub codes: Vec<u8>,      // [vocab][m], codes of each row's unit direction
    pub norms: Vec<f32>,     // [vocab], each row's real L2 norm (magnitude lost by normalizing)
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

/// Persist a trained table to disk: a k-means build like the one that
/// produced the 13/15-accurate config (M=128, 300 iterations, full-vocab
/// training) took ~7 minutes -- paying that on every process start instead
/// of once would defeat the entire point of a "slow to build, cheap to
/// run" mode. Raw little-endian dump (header + three flat buffers), no
/// external serialization crate needed.
pub fn save_pq(pq: &PqTable, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let vocab = pq.norms.len();
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(&(pq.m as u64).to_le_bytes())?;
    f.write_all(&(pq.sub_dim as u64).to_le_bytes())?;
    f.write_all(&(pq.k as u64).to_le_bytes())?;
    f.write_all(&(vocab as u64).to_le_bytes())?;
    f.write_all(as_bytes(&pq.codebooks))?;
    f.write_all(&pq.codes)?;
    f.write_all(as_bytes(&pq.norms))?;
    f.flush()
}

/// Load a table saved by `save_pq`, skipping training entirely.
pub fn load_pq(path: &str) -> std::io::Result<PqTable> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(File::open(path)?);
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr)?;
    let m = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
    let sub_dim = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let k = u64::from_le_bytes(hdr[16..24].try_into().unwrap()) as usize;
    let vocab = u64::from_le_bytes(hdr[24..32].try_into().unwrap()) as usize;

    let mut codebooks = vec![0f32; m * k * sub_dim];
    f.read_exact(as_bytes_mut(&mut codebooks))?;
    let mut codes = vec![0u8; vocab * m];
    f.read_exact(&mut codes)?;
    let mut norms = vec![0f32; vocab];
    f.read_exact(as_bytes_mut(&mut norms))?;

    Ok(PqTable { m, sub_dim, k, codebooks, codes, norms })
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Lloyd's algorithm k-means for one subspace, trained on a subsample of
/// the full table (standard PQ practice -- codebook quality saturates well
/// before using every row, and training on ~20k rows instead of 128k keeps
/// this a several-second operation instead of a minutes-long one).
fn train_subspace(data: &[f32], n_points: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut centroids = vec![0f32; k * dim];
    for c in 0..k {
        let idx = (xorshift(&mut state) as usize) % n_points;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
    }
    let mut assign = vec![0u32; n_points];
    for _ in 0..iters {
        for p in 0..n_points {
            let point = &data[p * dim..(p + 1) * dim];
            let mut best = f32::INFINITY;
            let mut best_c = 0u32;
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let mut d = 0f32;
                for i in 0..dim {
                    let diff = point[i] - cent[i];
                    d += diff * diff;
                }
                if d < best {
                    best = d;
                    best_c = c as u32;
                }
            }
            assign[p] = best_c;
        }
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0u32; k];
        for p in 0..n_points {
            let c = assign[p] as usize;
            counts[c] += 1;
            for i in 0..dim {
                sums[c * dim + i] += data[p * dim + i];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for i in 0..dim {
                    centroids[c * dim + i] = sums[c * dim + i] / counts[c] as f32;
                }
            }
            // empty cluster: leave its centroid at the random init point
            // it started from -- rare at this scale, not worth reseeding.
        }
    }
    centroids
}

/// Train + encode the whole embedding table via explicit positioned file
/// reads (pread), never mmap -- so unlike the earlier version, this never
/// permanently pages the ~656MB F16 table into this process's resident
/// memory. Training samples are read one small row at a time (random
/// access, cheap); encoding streams the table in bounded waves (same
/// windowed-wave pattern as quantize_embd_streaming_windowed) so the
/// transient read buffer never exceeds one wave's size regardless of vocab
/// size or thread count. Once this returns, `raw` bytes have been read and
/// discarded -- only PqTable's own few-MB structures remain resident.
pub fn build_pq(g: &Gguf, path: &str, vocab: usize, m: usize, k: usize, train_samples: usize, iters: usize) -> PqTable {
    assert_eq!(D_MODEL % m, 0, "D_MODEL must be divisible by m");
    let sub_dim = D_MODEL / m;
    let i = g.index_of("token_embd.weight");
    let data_off = g.data_base + g.offsets[i] as usize;
    let row_bytes = D_MODEL * 2;
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));

    // Train on UNIT-NORMALIZED rows: k-means minimizes Euclidean distance,
    // which only matches inner-product ranking (what we actually score by)
    // when vectors share a magnitude. Embedding rows don't -- normalizing
    // out that magnitude before training/encoding, then reapplying the
    // real per-row norm at scan time (see below), keeps the k-means
    // objective consistent with what's actually being ranked.
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut sample_rows = vec![0f32; train_samples * D_MODEL];
    let mut row_buf = vec![0u8; row_bytes];
    for s in 0..train_samples {
        let t = (xorshift(&mut state) as usize) % vocab;
        file.read_exact_at(&mut row_buf, (data_off + t * row_bytes) as u64).unwrap();
        let mut norm_sq = 0f32;
        for d in 0..D_MODEL {
            let b = d * 2;
            let v = f16_bits_to_f32(u16::from_le_bytes([row_buf[b], row_buf[b + 1]]));
            sample_rows[s * D_MODEL + d] = v;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for d in 0..D_MODEL {
                sample_rows[s * D_MODEL + d] /= norm;
            }
        }
    }

    let mut codebooks = vec![0f32; m * k * sub_dim];
    std::thread::scope(|scope| {
        for (mi, cb_chunk) in codebooks.chunks_mut(k * sub_dim).enumerate() {
            let sample_rows = &sample_rows;
            scope.spawn(move || {
                let mut sub_data = vec![0f32; train_samples * sub_dim];
                for s in 0..train_samples {
                    let src = &sample_rows[s * D_MODEL + mi * sub_dim..s * D_MODEL + (mi + 1) * sub_dim];
                    sub_data[s * sub_dim..(s + 1) * sub_dim].copy_from_slice(src);
                }
                let trained = train_subspace(&sub_data, train_samples, sub_dim, k, iters, 0xA5A5A5A5u64 + mi as u64);
                cb_chunk.copy_from_slice(&trained);
            });
        }
    });
    drop(sample_rows);

    let mut codes = vec![0u8; vocab * m];
    let mut norms = vec![0f32; vocab];
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    const WAVE_ROWS: usize = 4096;
    let mut row0 = 0usize;
    while row0 < vocab {
        let wave_end = (row0 + WAVE_ROWS).min(vocab);
        let wave_len = wave_end - row0;
        let rows_per_thread = wave_len.div_ceil(n_threads);
        let codes_wave = &mut codes[row0 * m..wave_end * m];
        let norms_wave = &mut norms[row0..wave_end];

        std::thread::scope(|scope| {
            for (codes_chunk, (norms_chunk, sub_row0)) in codes_wave
                .chunks_mut(rows_per_thread * m)
                .zip(norms_wave.chunks_mut(rows_per_thread).zip((0..wave_len).step_by(rows_per_thread)))
            {
                let file = &file;
                let codebooks = &codebooks;
                scope.spawn(move || {
                    let n_rows = norms_chunk.len();
                    let mut buf = vec![0u8; n_rows * row_bytes];
                    file.read_exact_at(&mut buf, (data_off + (row0 + sub_row0) * row_bytes) as u64).unwrap();
                    let mut row = vec![0f32; D_MODEL];
                    for r in 0..n_rows {
                        let off = r * row_bytes;
                        let mut norm_sq = 0f32;
                        for d in 0..D_MODEL {
                            let b = off + d * 2;
                            let v = f16_bits_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]]));
                            row[d] = v;
                            norm_sq += v * v;
                        }
                        let norm = norm_sq.sqrt();
                        norms_chunk[r] = norm;
                        if norm > 0.0 {
                            for d in 0..D_MODEL {
                                row[d] /= norm;
                            }
                        }
                        for mi in 0..m {
                            let sub = &row[mi * sub_dim..(mi + 1) * sub_dim];
                            let cb = &codebooks[mi * k * sub_dim..(mi + 1) * k * sub_dim];
                            let mut best = f32::INFINITY;
                            let mut best_c = 0u8;
                            for c in 0..k {
                                let cent = &cb[c * sub_dim..(c + 1) * sub_dim];
                                let mut d = 0f32;
                                for i in 0..sub_dim {
                                    let diff = sub[i] - cent[i];
                                    d += diff * diff;
                                }
                                if d < best {
                                    best = d;
                                    best_c = c as u8;
                                }
                            }
                            codes_chunk[r * m + mi] = best_c;
                        }
                    }
                    // `buf` (this wave's shard of the F16 table) is dropped
                    // here -- transient, never permanently resident.
                });
            }
        });
        row0 = wave_end;
    }

    PqTable { m, sub_dim, k, codebooks, codes, norms }
}

/// Per-query lookup table: for each subspace, the dot product of that
/// query sub-vector against every centroid (ADC scores by inner product,
/// matching the exact lm_head's dot-product ranking -- not L2 distance,
/// which would rank differently).
pub fn build_lut(pq: &PqTable, h: &[f32]) -> Vec<f32> {
    let mut lut = vec![0f32; pq.m * pq.k];
    for mi in 0..pq.m {
        let hsub = &h[mi * pq.sub_dim..(mi + 1) * pq.sub_dim];
        let cb = &pq.codebooks[mi * pq.k * pq.sub_dim..(mi + 1) * pq.k * pq.sub_dim];
        for c in 0..pq.k {
            let cent = &cb[c * pq.sub_dim..(c + 1) * pq.sub_dim];
            let mut dot = 0f32;
            for i in 0..pq.sub_dim {
                dot += hsub[i] * cent[i];
            }
            lut[mi * pq.k + c] = dot;
        }
    }
    lut
}

fn topk_insert(list: &mut Vec<(f32, i64)>, k: usize, score: f32, id: i64) {
    if list.len() < k {
        list.push((score, id));
        let mut pos = list.len() - 1;
        while pos > 0 && list[pos - 1].0 < score {
            list.swap(pos, pos - 1);
            pos -= 1;
        }
        return;
    }
    if score <= list[k - 1].0 {
        return;
    }
    list[k - 1] = (score, id);
    let mut pos = k - 1;
    while pos > 0 && list[pos - 1].0 < score {
        list.swap(pos, pos - 1);
        pos -= 1;
    }
}

/// ADC scan: score every row via M table lookups + adds (no multiplies) in
/// parallel, then select the TRUE global top-k over all vocab scores.
///
/// Earlier version kept each worker's LOCAL top-k independently (same
/// pattern as the int8 scan) -- that's fine when scoring error is close to
/// independent per-token, which int8's per-element quantization noise is,
/// but PQ's errors are NOT: measured directly, a token with global PQ rank
/// 803 (comfortably inside a few thousand candidates) had a LOCAL rank of
/// 348 within its own ~4000-row shard -- token IDs aren't randomly
/// distributed w.r.t. PQ score (low-ID tokens cluster together non-
/// uniformly), so per-shard top-k silently drops globally-good candidates
/// whenever their shard happens to be locally crowded. Selecting top-k
/// over the full merged score array fixes this at the cost of one extra
/// O(vocab) pass, which is cheap next to the scan itself.
pub fn pq_topk_scan(pq: &PqTable, lut: &[f32], vocab: usize, k: usize) -> Vec<(f32, i64)> {
    let mut scores = vec![0f32; vocab];
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let rows_per_chunk = vocab.div_ceil(n_threads);
    std::thread::scope(|scope| {
        for (chunk_idx, scores_chunk) in scores.chunks_mut(rows_per_chunk).enumerate() {
            let row0 = chunk_idx * rows_per_chunk;
            let codes = &pq.codes;
            let norms = &pq.norms;
            let m = pq.m;
            let kk = pq.k;
            scope.spawn(move || {
                for (i, s) in scores_chunk.iter_mut().enumerate() {
                    let t = row0 + i;
                    let base = t * m;
                    let mut dir_score = 0f32;
                    for mi in 0..m {
                        let code = codes[base + mi] as usize;
                        dir_score += lut[mi * kk + code];
                    }
                    *s = dir_score * norms[t];
                }
            });
        }
    });

    let mut top: Vec<(f32, i64)> = Vec::with_capacity(k);
    for (t, &s) in scores.iter().enumerate() {
        topk_insert(&mut top, k, s, t as i64);
    }
    top
}

/// logits[t] = row_t(token_embd) . h_normed ; return argmax, via PQ ADC
/// scan (cheap, approximate) + exact F16 rescore of the candidates.
pub fn lm_head_argmax_pq_rescore(reader: &crate::embd::EmbdFileReader, pq: &PqTable, h_normed: &[f32], vocab: usize, k: usize) -> i64 {
    let lut = build_lut(pq, h_normed);
    let cands = pq_topk_scan(pq, &lut, vocab, k);
    let mut best = f32::NEG_INFINITY;
    let mut best_id = cands[0].1;
    for &(_, tid) in &cands {
        let row = reader.read_row_f32(tid, D_MODEL);
        let mut dot = 0f32;
        for d in 0..D_MODEL {
            dot += row[d] * h_normed[d];
        }
        if dot > best {
            best = dot;
            best_id = tid;
        }
    }
    best_id
}

/// Batched sibling: B independent PQ scans (each is already cheap -- just
/// table lookups, no per-row multiply -- so this doesn't need the fused
/// pool-dispatch treatment the int8 path needed to amortize DRAM reads).
pub fn lm_head_argmax_pq_rescore_batched(reader: &crate::embd::EmbdFileReader, pq: &PqTable, h_batch: &[Vec<f32>], vocab: usize, k: usize) -> Vec<i64> {
    h_batch.iter().map(|h| lm_head_argmax_pq_rescore(reader, pq, h, vocab, k)).collect()
}
