// int8-quantized tied embedding + top-K rescore, ported from
// embd_quant.mojo's "everything stacked" config: cheap int8 AVX-VNNI scan
// of the full vocab for a top-K candidate set, then exact F16 rescore of
// just those K rows -- cuts the dominant lm_head cost (which was ~51% of
// decode time, a straight 656MB F16 scan every token) down to a ~164MB
// int8 scan plus a handful of exact re-checks, with no accuracy loss:
// the rescore always picks the true argmax among the K candidates, and K
// is chosen generously (16) so the true best essentially never falls
// outside the int8 scan's top-K.

use crate::gguf::{f16_bits_to_f32, Gguf};
use crate::model::D_MODEL;
use std::fs::File;
use std::os::unix::fs::FileExt;

pub struct QuantEmbd {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}

/// Mem-mode embedding access: explicit positioned file reads (pread) for
/// the F16 token_embd table instead of the mmap path -- like Mojo's
/// EmbdFileReader/_read_at, this never maps the whole ~656MB table into
/// this process's virtual memory, so it never shows up in RSS. The cost is
/// a syscall per row instead of a pointer offset; on a warm OS page cache
/// (the common case once the file has been read once, by anyone) the
/// underlying bytes are still cheap to fetch, so this trades a few
/// hundred MB of resident memory for some extra per-row syscall overhead --
/// see quantize_embd_streaming / lm_head_argmax_topk_rescore_pread for
/// where that overhead actually lands.
pub struct EmbdFileReader {
    file: File,
    data_off: usize,
}

impl EmbdFileReader {
    pub fn open(path: &str, g: &Gguf) -> Self {
        let i = g.index_of("token_embd.weight");
        let data_off = g.data_base + g.offsets[i] as usize;
        let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
        EmbdFileReader { file, data_off }
    }

    pub fn read_row_bytes(&self, token_id: i64, d_model: usize) -> Vec<u8> {
        let off = self.data_off + token_id as usize * d_model * 2;
        let mut buf = vec![0u8; d_model * 2];
        self.file.read_exact_at(&mut buf, off as u64).unwrap();
        buf
    }

    pub fn read_row_f32(&self, token_id: i64, d_model: usize) -> Vec<f32> {
        let buf = self.read_row_bytes(token_id, d_model);
        (0..d_model)
            .map(|d| f16_bits_to_f32(u16::from_le_bytes([buf[d * 2], buf[d * 2 + 1]])))
            .collect()
    }
}

/// Same result as quantize_embd, but reads the F16 table via explicit
/// positioned reads on a plain File instead of the mmap -- each thread
/// reads and quantizes its shard's bytes into a transient buffer that's
/// freed when this function returns, instead of the mmap path's pages,
/// which -- absent memory pressure -- never get reclaimed for the rest of
/// the process's life once touched. Peak RSS during this call can still
/// spike (all shards' buffers are alive concurrently), but afterward it
/// drops back down, unlike quantize_embd's permanent ~656MB.
pub fn quantize_embd_streaming(path: &str, g: &Gguf, vocab: usize) -> QuantEmbd {
    let i = g.index_of("token_embd.weight");
    let data_off = g.data_base + g.offsets[i] as usize;
    let row_bytes = D_MODEL * 2;
    let mut codes = vec![0i8; vocab * D_MODEL];
    let mut scales = vec![0f32; vocab];

    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let rows_per_chunk = vocab.div_ceil(n_threads);

    std::thread::scope(|s| {
        for (codes_chunk, (scales_chunk, row0)) in codes
            .chunks_mut(rows_per_chunk * D_MODEL)
            .zip(scales.chunks_mut(rows_per_chunk).zip((0..vocab).step_by(rows_per_chunk)))
        {
            let file = &file;
            s.spawn(move || {
                let n_rows = scales_chunk.len();
                let mut buf = vec![0u8; n_rows * row_bytes];
                file.read_exact_at(&mut buf, (data_off + row0 * row_bytes) as u64).unwrap();
                for r in 0..n_rows {
                    let off = r * row_bytes;
                    let mut vals = [0f32; D_MODEL];
                    let mut amax = 0f32;
                    for d in 0..D_MODEL {
                        let b = off + d * 2;
                        let v = f16_bits_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]]));
                        vals[d] = v;
                        amax = amax.max(v.abs());
                    }
                    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    scales_chunk[r] = scale;
                    let row_codes = &mut codes_chunk[r * D_MODEL..(r + 1) * D_MODEL];
                    for d in 0..D_MODEL {
                        row_codes[d] = (vals[d] / scale).round().clamp(-128.0, 127.0) as i8;
                    }
                }
                // `buf` (this shard's chunk of the F16 table) is dropped
                // here, at end of scope -- transient, not permanent RSS.
            });
        }
    });

    QuantEmbd { codes, scales }
}

/// Same result again, but processed in small bounded waves (like Mojo's
/// chunk_rows=4096 streaming quantizer) instead of handing each thread its
/// entire shard at once -- caps peak transient memory to one wave's size
/// regardless of vocab size or thread count. `wave_rows` is the total rows
/// per wave (split across threads within that wave); Mojo used ~4096.
pub fn quantize_embd_streaming_windowed(path: &str, g: &Gguf, vocab: usize, wave_rows: usize) -> QuantEmbd {
    let i = g.index_of("token_embd.weight");
    let data_off = g.data_base + g.offsets[i] as usize;
    let row_bytes = D_MODEL * 2;
    let mut codes = vec![0i8; vocab * D_MODEL];
    let mut scales = vec![0f32; vocab];

    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    let mut row0 = 0usize;
    while row0 < vocab {
        let wave_end = (row0 + wave_rows).min(vocab);
        let wave_len = wave_end - row0;
        let rows_per_thread = wave_len.div_ceil(n_threads);

        let codes_wave = &mut codes[row0 * D_MODEL..wave_end * D_MODEL];
        let scales_wave = &mut scales[row0..wave_end];

        std::thread::scope(|s| {
            for (codes_chunk, (scales_chunk, sub_row0)) in codes_wave
                .chunks_mut(rows_per_thread * D_MODEL)
                .zip(scales_wave.chunks_mut(rows_per_thread).zip((0..wave_len).step_by(rows_per_thread)))
            {
                let file = &file;
                s.spawn(move || {
                    let n_rows = scales_chunk.len();
                    let mut buf = vec![0u8; n_rows * row_bytes];
                    file.read_exact_at(&mut buf, (data_off + (row0 + sub_row0) * row_bytes) as u64).unwrap();
                    for r in 0..n_rows {
                        let off = r * row_bytes;
                        let mut vals = [0f32; D_MODEL];
                        let mut amax = 0f32;
                        for d in 0..D_MODEL {
                            let b = off + d * 2;
                            let v = f16_bits_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]]));
                            vals[d] = v;
                            amax = amax.max(v.abs());
                        }
                        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                        scales_chunk[r] = scale;
                        let row_codes = &mut codes_chunk[r * D_MODEL..(r + 1) * D_MODEL];
                        for d in 0..D_MODEL {
                            row_codes[d] = (vals[d] / scale).round().clamp(-128.0, 127.0) as i8;
                        }
                    }
                });
            }
        });
        // this wave's buffers are all freed here, before the next wave starts
        row0 = wave_end;
    }

    QuantEmbd { codes, scales }
}

/// Per-row absmax int8 quantization of the tied embedding table, matching
/// bitlinear's activation quantizer convention (scale = max(|x|)/127,
/// round+clamp). Parallelized over row chunks -- this runs once at model
/// load, not per token, so a plain scoped-thread split is enough (no need
/// to route through the persistent pool).
pub fn quantize_embd(g: &Gguf, vocab: usize) -> QuantEmbd {
    let raw = g.f16_tensor_bytes("token_embd.weight");
    let mut codes = vec![0i8; vocab * D_MODEL];
    let mut scales = vec![0f32; vocab];

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let rows_per_chunk = vocab.div_ceil(n_threads);

    std::thread::scope(|s| {
        for (codes_chunk, (scales_chunk, row0)) in codes
            .chunks_mut(rows_per_chunk * D_MODEL)
            .zip(scales.chunks_mut(rows_per_chunk).zip((0..vocab).step_by(rows_per_chunk)))
        {
            s.spawn(move || {
                let n_rows = scales_chunk.len();
                for r in 0..n_rows {
                    let t = row0 + r;
                    let off = t * D_MODEL * 2;
                    let mut vals = [0f32; D_MODEL];
                    let mut amax = 0f32;
                    for d in 0..D_MODEL {
                        let b = off + d * 2;
                        let v = f16_bits_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]));
                        vals[d] = v;
                        amax = amax.max(v.abs());
                    }
                    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    scales_chunk[r] = scale;
                    let row_codes = &mut codes_chunk[r * D_MODEL..(r + 1) * D_MODEL];
                    for d in 0..D_MODEL {
                        row_codes[d] = (vals[d] / scale).round().clamp(-128.0, 127.0) as i8;
                    }
                }
            });
        }
    });

    QuantEmbd { codes, scales }
}

fn quant_act_sum(h: &[f32]) -> (Vec<i8>, f32, i32) {
    let amax = h.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let mut sum = 0i32;
    let codes: Vec<i8> = h
        .iter()
        .map(|&x| {
            let q = (x / scale).round().clamp(-128.0, 127.0) as i32;
            sum += q;
            q as i8
        })
        .collect();
    (codes, scale, sum)
}

/// logits[t] = row_t(token_embd) . h_normed ; return argmax, via int8 VNNI
/// top-K scan (cheap, approximate) + exact F16 rescore of the K candidates
/// (exact, from the mmap'd table -- same math as the plain lm_head_argmax).
pub fn lm_head_argmax_topk_rescore(
    embd_raw: &[u8],
    qe: &QuantEmbd,
    h_normed: &[f32],
    d_model: usize,
    vocab: usize,
    k: usize,
) -> i64 {
    let (acts, act_scale, act_sum) = quant_act_sum(h_normed);
    let ids = crate::pool::dispatch_topk_scan(
        &qe.codes, &qe.scales, &acts, act_sum, act_scale, d_model, vocab, k,
    );

    let row_bytes = d_model * 2;
    let mut best = f32::NEG_INFINITY;
    let mut best_id = ids[0];
    for &tid in &ids {
        let row = &embd_raw[tid as usize * row_bytes..(tid as usize + 1) * row_bytes];
        let mut dot = 0f32;
        for d in 0..d_model {
            let b = d * 2;
            let ev = f16_bits_to_f32(u16::from_le_bytes([row[b], row[b + 1]]));
            dot += ev * h_normed[d];
        }
        if dot > best {
            best = dot;
            best_id = tid;
        }
    }
    best_id
}

/// Batched version of lm_head_argmax_topk_rescore: verifies B hidden
/// states (e.g. B prompt-lookup-decoding positions) against the quantized
/// embedding table in one dispatch, then exactly rescores each one's
/// candidates. Returns one argmax id per b, same order as `h_normed_batch`.
pub fn lm_head_argmax_topk_rescore_batched(
    embd_raw: &[u8],
    qe: &QuantEmbd,
    h_normed_batch: &[Vec<f32>],
    d_model: usize,
    vocab: usize,
    k: usize,
) -> Vec<i64> {
    let b_count = h_normed_batch.len();
    let mut acts_flat = vec![0i8; b_count * d_model];
    let mut act_sums = vec![0i32; b_count];
    let mut act_scales = vec![0f32; b_count];
    for (b, h) in h_normed_batch.iter().enumerate() {
        let (acts, scale, sum) = quant_act_sum(h);
        acts_flat[b * d_model..(b + 1) * d_model].copy_from_slice(&acts);
        act_sums[b] = sum;
        act_scales[b] = scale;
    }

    let ids_per_b = crate::pool::dispatch_topk_scan_batched(
        &qe.codes, &qe.scales, &acts_flat, &act_sums, &act_scales, d_model, vocab, k,
    );

    let row_bytes = d_model * 2;
    ids_per_b
        .iter()
        .zip(h_normed_batch)
        .map(|(ids, h_normed)| {
            let mut best = f32::NEG_INFINITY;
            let mut best_id = ids[0];
            for &tid in ids {
                let row = &embd_raw[tid as usize * row_bytes..(tid as usize + 1) * row_bytes];
                let mut dot = 0f32;
                for d in 0..d_model {
                    let b = d * 2;
                    let ev = f16_bits_to_f32(u16::from_le_bytes([row[b], row[b + 1]]));
                    dot += ev * h_normed[d];
                }
                if dot > best {
                    best = dot;
                    best_id = tid;
                }
            }
            best_id
        })
        .collect()
}

/// Mem-mode sibling of lm_head_argmax_topk_rescore_batched: same int8-scan
/// + exact-rescore logic, but the exact rescore reads candidate rows via
/// `EmbdFileReader` (pread) instead of an mmap slice.
pub fn lm_head_argmax_topk_rescore_batched_pread(
    reader: &EmbdFileReader,
    qe: &QuantEmbd,
    h_normed_batch: &[Vec<f32>],
    d_model: usize,
    vocab: usize,
    k: usize,
) -> Vec<i64> {
    let b_count = h_normed_batch.len();
    let mut acts_flat = vec![0i8; b_count * d_model];
    let mut act_sums = vec![0i32; b_count];
    let mut act_scales = vec![0f32; b_count];
    for (b, h) in h_normed_batch.iter().enumerate() {
        let (acts, scale, sum) = quant_act_sum(h);
        acts_flat[b * d_model..(b + 1) * d_model].copy_from_slice(&acts);
        act_sums[b] = sum;
        act_scales[b] = scale;
    }

    let ids_per_b = crate::pool::dispatch_topk_scan_batched(
        &qe.codes, &qe.scales, &acts_flat, &act_sums, &act_scales, d_model, vocab, k,
    );

    ids_per_b
        .iter()
        .zip(h_normed_batch)
        .map(|(ids, h_normed)| {
            let mut best = f32::NEG_INFINITY;
            let mut best_id = ids[0];
            for &tid in ids {
                let row = reader.read_row_f32(tid, d_model);
                let mut dot = 0f32;
                for d in 0..d_model {
                    dot += row[d] * h_normed[d];
                }
                if dot > best {
                    best = dot;
                    best_id = tid;
                }
            }
            best_id
        })
        .collect()
}

/// Mem-mode sibling of the single-position lm_head_argmax_topk_rescore.
pub fn lm_head_argmax_topk_rescore_pread(
    reader: &EmbdFileReader,
    qe: &QuantEmbd,
    h_normed: &[f32],
    d_model: usize,
    vocab: usize,
    k: usize,
) -> i64 {
    lm_head_argmax_topk_rescore_batched_pread(reader, qe, std::slice::from_ref(&h_normed.to_vec()), d_model, vocab, k)[0]
}

/// Same as lm_head_argmax_topk_rescore but returns (id, quant_time,
/// scan_time, rescore_time, n_candidates) -- diagnostic only.
pub fn lm_head_argmax_topk_rescore_timed(
    embd_raw: &[u8],
    qe: &QuantEmbd,
    h_normed: &[f32],
    d_model: usize,
    vocab: usize,
    k: usize,
) -> (i64, std::time::Duration, std::time::Duration, std::time::Duration, usize) {
    let t0 = std::time::Instant::now();
    let (acts, act_scale, act_sum) = quant_act_sum(h_normed);
    let t_quant = t0.elapsed();

    let t0 = std::time::Instant::now();
    let ids = crate::pool::dispatch_topk_scan(
        &qe.codes, &qe.scales, &acts, act_sum, act_scale, d_model, vocab, k,
    );
    let t_scan = t0.elapsed();

    let t0 = std::time::Instant::now();
    let row_bytes = d_model * 2;
    let mut best = f32::NEG_INFINITY;
    let mut best_id = ids[0];
    for &tid in &ids {
        let row = &embd_raw[tid as usize * row_bytes..(tid as usize + 1) * row_bytes];
        let mut dot = 0f32;
        for d in 0..d_model {
            let b = d * 2;
            let ev = f16_bits_to_f32(u16::from_le_bytes([row[b], row[b + 1]]));
            dot += ev * h_normed[d];
        }
        if dot > best {
            best = dot;
            best_id = tid;
        }
    }
    let t_rescore = t0.elapsed();
    (best_id, t_quant, t_scan, t_rescore, ids.len())
}
