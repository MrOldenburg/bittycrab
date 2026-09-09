// Optimized Product Quantization (Ge et al., 2013). Plain PQ (pq.rs) splits
// each row into M contiguous chunks and clusters each independently -- that
// split is arbitrary, and if the embedding's variance/correlation structure
// doesn't respect those boundaries (measured: it doesn't, plain PQ needed
// k=1200+ and heavy overtraining to get partial accuracy), each subspace's
// k-means is stuck with an unnecessarily hard clustering problem.
//
// OPQ learns an orthogonal (distance-preserving) rotation R before the
// split, chosen so that after rotation, variance is spread evenly across
// the M subspaces. R is learned jointly with the codebooks, alternating:
//   1. rotate training data by the current R
//   2. train per-subspace codebooks on the rotated data (same k-means as
//      plain PQ)
//   3. reconstruct the rotated data from its codes, then solve for the R
//      that best re-aligns the ORIGINAL (unrotated) data to those
//      reconstructions -- this is the classic Orthogonal Procrustes
//      problem: minimize ||X@R - Y||_F over orthogonal R, solved by
//      SVD(X^T @ Y) = U*S*V^T, R = U @ V^T.
// Repeat for a few outer iterations. Query time: rotate the query by R
// before building the ADC lookup table (codebooks live in rotated space);
// everything else (ADC scan, exact F16 rescore) is identical to plain PQ.

use crate::gguf::{f16_bits_to_f32, Gguf};
use crate::model::D_MODEL;
use nalgebra::DMatrix;
use std::fs::File;
use std::os::unix::fs::FileExt;

pub struct OpqTable {
    pub m: usize,
    pub sub_dim: usize,
    pub k: usize,
    pub rotation: Vec<f32>,  // [D_MODEL][D_MODEL] row-major, orthogonal
    pub codebooks: Vec<f32>, // [m][k][sub_dim], trained on ROTATED unit-normalized rows
    pub codes: Vec<u8>,      // [vocab][m]
    pub norms: Vec<f32>,     // [vocab]
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// x (n x D_MODEL row-major) times rotation R (D_MODEL x D_MODEL row-major):
/// out[i] = x[i] @ R. Uses nalgebra's matmul (SIMD-optimized via the
/// `matrixmultiply` crate under the hood) rather than a naive triple loop.
fn rotate_rows(x: &[f32], n: usize, rotation: &[f32]) -> Vec<f32> {
    let xm = DMatrix::from_row_slice(n, D_MODEL, x);
    let rm = DMatrix::from_row_slice(D_MODEL, D_MODEL, rotation);
    let out = xm * rm;
    out.transpose().as_slice().to_vec() // nalgebra is column-major internally; re-linearize to row-major
}

fn identity_rotation() -> Vec<f32> {
    let mut r = vec![0f32; D_MODEL * D_MODEL];
    for i in 0..D_MODEL {
        r[i * D_MODEL + i] = 1.0;
    }
    r
}

/// Solve the Orthogonal Procrustes problem: R minimizing ||X@R - Y||_F over
/// orthogonal D_MODEL x D_MODEL R, both X and Y given as n x D_MODEL
/// row-major. R = U @ V^T where SVD(X^T @ Y) = U * S * V^T.
fn solve_procrustes(x: &[f32], y: &[f32], n: usize) -> Vec<f32> {
    let xm = DMatrix::from_row_slice(n, D_MODEL, x);
    let ym = DMatrix::from_row_slice(n, D_MODEL, y);
    let c = xm.transpose() * ym; // D_MODEL x D_MODEL
    let svd = c.svd(true, true);
    let u = svd.u.expect("svd u");
    let vt = svd.v_t.expect("svd v_t");
    let r = u * vt;
    // row-major linearize
    let mut out = vec![0f32; D_MODEL * D_MODEL];
    for i in 0..D_MODEL {
        for j in 0..D_MODEL {
            out[i * D_MODEL + j] = r[(i, j)];
        }
    }
    out
}

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
        }
    }
    centroids
}

fn train_codebooks(rotated: &[f32], n: usize, m: usize, sub_dim: usize, k: usize, iters: usize) -> Vec<f32> {
    let mut codebooks = vec![0f32; m * k * sub_dim];
    std::thread::scope(|scope| {
        for (mi, cb_chunk) in codebooks.chunks_mut(k * sub_dim).enumerate() {
            let rotated = rotated;
            scope.spawn(move || {
                let mut sub_data = vec![0f32; n * sub_dim];
                for s in 0..n {
                    let src = &rotated[s * D_MODEL + mi * sub_dim..s * D_MODEL + (mi + 1) * sub_dim];
                    sub_data[s * sub_dim..(s + 1) * sub_dim].copy_from_slice(src);
                }
                let trained = train_subspace(&sub_data, n, sub_dim, k, iters, 0xA5A5A5A5u64 + mi as u64);
                cb_chunk.copy_from_slice(&trained);
            });
        }
    });
    codebooks
}

/// Encode `rotated` (n x D_MODEL) against `codebooks`, returning n*m codes.
fn encode_rows(rotated: &[f32], n: usize, m: usize, sub_dim: usize, k: usize, codebooks: &[f32]) -> Vec<u8> {
    let mut codes = vec![0u8; n * m];
    for s in 0..n {
        let row = &rotated[s * D_MODEL..(s + 1) * D_MODEL];
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
            codes[s * m + mi] = best_c;
        }
    }
    codes
}

/// Reconstruct n x D_MODEL rows from codes (nearest-centroid lookup per
/// subspace), used only to build the target for the Procrustes step.
fn reconstruct_rows(codes: &[u8], n: usize, m: usize, sub_dim: usize, k: usize, codebooks: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; n * D_MODEL];
    for s in 0..n {
        for mi in 0..m {
            let code = codes[s * m + mi] as usize;
            let cent = &codebooks[(mi * k + code) * sub_dim..(mi * k + code + 1) * sub_dim];
            out[s * D_MODEL + mi * sub_dim..s * D_MODEL + (mi + 1) * sub_dim].copy_from_slice(cent);
        }
    }
    out
}

/// Train the rotation + codebooks (on a training subsample, unit-normalized
/// same as plain PQ), then encode the full vocab. Reads via pread only
/// (never mmap) -- same discipline as build_pq. `outer_iters` controls how
/// many rotate/train/re-solve-R rounds run; each one costs a D_MODEL x
/// D_MODEL SVD (the expensive, slow part -- this is deliberately not fast).
pub fn build_opq(
    g: &Gguf,
    path: &str,
    vocab: usize,
    m: usize,
    k: usize,
    train_samples: usize,
    kmeans_iters: usize,
    outer_iters: usize,
) -> OpqTable {
    assert_eq!(D_MODEL % m, 0, "D_MODEL must be divisible by m");
    let sub_dim = D_MODEL / m;
    let i = g.index_of("token_embd.weight");
    let data_off = g.data_base + g.offsets[i] as usize;
    let row_bytes = D_MODEL * 2;
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut samples = vec![0f32; train_samples * D_MODEL];
    let mut row_buf = vec![0u8; row_bytes];
    for s in 0..train_samples {
        let t = (xorshift(&mut state) as usize) % vocab;
        file.read_exact_at(&mut row_buf, (data_off + t * row_bytes) as u64).unwrap();
        let mut norm_sq = 0f32;
        for d in 0..D_MODEL {
            let b = d * 2;
            let v = f16_bits_to_f32(u16::from_le_bytes([row_buf[b], row_buf[b + 1]]));
            samples[s * D_MODEL + d] = v;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for d in 0..D_MODEL {
                samples[s * D_MODEL + d] /= norm;
            }
        }
    }

    let mut rotation = identity_rotation();
    let mut codebooks = vec![0f32; m * k * sub_dim];
    for outer in 0..outer_iters {
        let t0 = std::time::Instant::now();
        let rotated = rotate_rows(&samples, train_samples, &rotation);
        codebooks = train_codebooks(&rotated, train_samples, m, sub_dim, k, kmeans_iters);
        let codes = encode_rows(&rotated, train_samples, m, sub_dim, k, &codebooks);
        let recon = reconstruct_rows(&codes, train_samples, m, sub_dim, k, &codebooks);
        rotation = solve_procrustes(&samples, &recon, train_samples);
        println!("  opq outer iter {}/{}: {:.2}s", outer + 1, outer_iters, t0.elapsed().as_secs_f64());
    }

    // Final encode of the whole vocab against the trained rotation+codebooks.
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
                let rotation = &rotation;
                scope.spawn(move || {
                    let n_rows = norms_chunk.len();
                    let mut buf = vec![0u8; n_rows * row_bytes];
                    file.read_exact_at(&mut buf, (data_off + (row0 + sub_row0) * row_bytes) as u64).unwrap();

                    // Gather + normalize all this chunk's rows first, then
                    // rotate them in ONE batched matmul (SIMD-optimized via
                    // nalgebra/matrixmultiply) instead of a per-row scalar
                    // D_MODEL x D_MODEL triple loop -- that naive version
                    // measured ~109s for just 128256 rows in a smoke test;
                    // this is the same fix rotate_rows already uses for
                    // training samples, just applied here too.
                    let mut rows_flat = vec![0f32; n_rows * D_MODEL];
                    for r in 0..n_rows {
                        let off = r * row_bytes;
                        let mut norm_sq = 0f32;
                        for d in 0..D_MODEL {
                            let b = off + d * 2;
                            let v = f16_bits_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]]));
                            rows_flat[r * D_MODEL + d] = v;
                            norm_sq += v * v;
                        }
                        let norm = norm_sq.sqrt();
                        norms_chunk[r] = norm;
                        if norm > 0.0 {
                            for d in 0..D_MODEL {
                                rows_flat[r * D_MODEL + d] /= norm;
                            }
                        }
                    }
                    let rotated = rotate_rows(&rows_flat, n_rows, rotation);

                    for r in 0..n_rows {
                        let rotated_row = &rotated[r * D_MODEL..(r + 1) * D_MODEL];
                        for mi in 0..m {
                            let sub = &rotated_row[mi * sub_dim..(mi + 1) * sub_dim];
                            let cb = &codebooks[mi * k * sub_dim..(mi + 1) * k * sub_dim];
                            let mut best = f32::INFINITY;
                            let mut best_c = 0u8;
                            for c in 0..k {
                                let cent = &cb[c * sub_dim..(c + 1) * sub_dim];
                                let mut d = 0f32;
                                for ii in 0..sub_dim {
                                    let diff = sub[ii] - cent[ii];
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
                });
            }
        });
        row0 = wave_end;
    }

    OpqTable { m, sub_dim, k, rotation, codebooks, codes, norms }
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn as_bytes_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

pub fn save_opq(t: &OpqTable, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let vocab = t.norms.len();
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(&(t.m as u64).to_le_bytes())?;
    f.write_all(&(t.sub_dim as u64).to_le_bytes())?;
    f.write_all(&(t.k as u64).to_le_bytes())?;
    f.write_all(&(vocab as u64).to_le_bytes())?;
    f.write_all(as_bytes(&t.rotation))?;
    f.write_all(as_bytes(&t.codebooks))?;
    f.write_all(&t.codes)?;
    f.write_all(as_bytes(&t.norms))?;
    f.flush()
}

pub fn load_opq(path: &str) -> std::io::Result<OpqTable> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(File::open(path)?);
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr)?;
    let m = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
    let sub_dim = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let k = u64::from_le_bytes(hdr[16..24].try_into().unwrap()) as usize;
    let vocab = u64::from_le_bytes(hdr[24..32].try_into().unwrap()) as usize;

    let mut rotation = vec![0f32; D_MODEL * D_MODEL];
    f.read_exact(as_bytes_mut(&mut rotation))?;
    let mut codebooks = vec![0f32; m * k * sub_dim];
    f.read_exact(as_bytes_mut(&mut codebooks))?;
    let mut codes = vec![0u8; vocab * m];
    f.read_exact(&mut codes)?;
    let mut norms = vec![0f32; vocab];
    f.read_exact(as_bytes_mut(&mut norms))?;

    Ok(OpqTable { m, sub_dim, k, rotation, codebooks, codes, norms })
}

fn rotate_vec(h: &[f32], rotation: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; D_MODEL];
    for j in 0..D_MODEL {
        let mut acc = 0f32;
        for l in 0..D_MODEL {
            acc += h[l] * rotation[l * D_MODEL + j];
        }
        out[j] = acc;
    }
    out
}

pub fn build_lut(t: &OpqTable, h: &[f32]) -> Vec<f32> {
    let hr = rotate_vec(h, &t.rotation);
    let mut lut = vec![0f32; t.m * t.k];
    for mi in 0..t.m {
        let hsub = &hr[mi * t.sub_dim..(mi + 1) * t.sub_dim];
        let cb = &t.codebooks[mi * t.k * t.sub_dim..(mi + 1) * t.k * t.sub_dim];
        for c in 0..t.k {
            let cent = &cb[c * t.sub_dim..(c + 1) * t.sub_dim];
            let mut dot = 0f32;
            for i in 0..t.sub_dim {
                dot += hsub[i] * cent[i];
            }
            lut[mi * t.k + c] = dot;
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

pub fn opq_topk_scan(t: &OpqTable, lut: &[f32], vocab: usize, k: usize) -> Vec<(f32, i64)> {
    let mut scores = vec![0f32; vocab];
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let rows_per_chunk = vocab.div_ceil(n_threads);
    std::thread::scope(|scope| {
        for (chunk_idx, scores_chunk) in scores.chunks_mut(rows_per_chunk).enumerate() {
            let row0 = chunk_idx * rows_per_chunk;
            let codes = &t.codes;
            let norms = &t.norms;
            let m = t.m;
            let kk = t.k;
            scope.spawn(move || {
                for (i, s) in scores_chunk.iter_mut().enumerate() {
                    let tid = row0 + i;
                    let base = tid * m;
                    let mut dir_score = 0f32;
                    for mi in 0..m {
                        let code = codes[base + mi] as usize;
                        dir_score += lut[mi * kk + code];
                    }
                    *s = dir_score * norms[tid];
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

pub fn lm_head_argmax_opq_rescore(reader: &crate::embd::EmbdFileReader, t: &OpqTable, h_normed: &[f32], vocab: usize, k: usize) -> i64 {
    let lut = build_lut(t, h_normed);
    let cands = opq_topk_scan(t, &lut, vocab, k);
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

pub fn lm_head_argmax_opq_rescore_batched(reader: &crate::embd::EmbdFileReader, t: &OpqTable, h_batch: &[Vec<f32>], vocab: usize, k: usize) -> Vec<i64> {
    h_batch.iter().map(|h| lm_head_argmax_opq_rescore(reader, t, h, vocab, k)).collect()
}
