// Prompt-lookup decoding: draft candidate continuation tokens by finding a
// repeat of the current suffix earlier in the sequence (no second model, no
// training), verify them all in ONE batched forward pass, and accept the
// longest matching prefix. This is provably lossless under greedy decoding:
// every accepted token, and the mandatory "bonus" token after a rejection,
// is computed by the exact same math as the plain one-token-at-a-time path
// (transformer_layer_batched does identical per-position math to
// transformer_layer, just reordered so the position-independent matmuls
// share one dispatch across positions -- see its docstring). The only thing
// that changes is how many tokens get produced per pass through the model's
// ~1.1GB of weights, which is where the speedup comes from on a
// memory-bandwidth-bound decode (verifying B candidates costs about the
// same as computing 1, since the weight bytes are read once either way).

use crate::embd::{
    lm_head_argmax_topk_rescore_batched, lm_head_argmax_topk_rescore_batched_pread,
    lm_head_argmax_topk_rescore_pread, EmbdFileReader, QuantEmbd,
};
use crate::model::{
    embed_token, lm_head_argmax, lm_head_argmax_batched, rmsnorm, rope_inv_freqs,
    transformer_layer_batched, Model, D_MODEL, EOS_ID, N_LAYER, RMS_EPS, VOCAB,
};
use crate::pool::MAX_BATCH;

/// Try each n-gram size in `ngrams` in order (longest/most-specific first is
/// the intended usage, e.g. &[4,3,2]), and return the draft from the first
/// size that finds a prior occurrence, CAPPED at that n-gram's own length
/// rather than always `max_draft`.
///
/// Why cap by match length: a short n-gram (e.g. 2 tokens) recurring is weak
/// evidence -- any repeated word creates one -- so drafting the full
/// max_draft on that signal usually gets rejected almost immediately, but
/// still pays the full batched-verify cost (measured: adding an uncapped
/// ngram=2 fallback made the 15-prompt harness slower, 43.4 vs 49.9 tok/s,
/// exactly this failure mode). A longer match is much stronger evidence the
/// continuation genuinely repeats, so it's allowed to draft further. Every
/// draft is still fully verified regardless of length, so this only trades
/// off wasted-round cost vs hit rate -- correctness is unaffected either way.
fn find_draft(history: &[i64], ngrams: &[usize], max_draft: usize) -> Vec<i64> {
    for &ngram in ngrams {
        let n = history.len();
        if n < ngram + 1 {
            continue;
        }
        let needle = &history[n - ngram..];
        for start in (0..(n - ngram)).rev() {
            if &history[start..start + ngram] == needle {
                let cap = ngram.min(max_draft);
                let avail = n - (start + ngram);
                let take = avail.min(cap);
                return history[start + ngram..start + ngram + take].to_vec();
            }
        }
    }
    Vec::new()
}

pub fn generate_pld(
    m: &Model,
    qe: &QuantEmbd,
    prompt_ids: &[i64],
    max_new: usize,
    k: usize,
    ngrams: &[usize],
    max_draft: usize,
) -> Vec<i64> {
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    // Prompt priming isn't speculative at all -- every prompt token is
    // already exactly known, nothing to draft or verify -- so it gets the
    // same batching win unconditionally instead of PLD's data-dependent
    // one: process up to MAX_BATCH prompt positions per pass through the
    // model's weights instead of one at a time. Same math as the serial
    // loop (transformer_layer_batched is position-for-position identical
    // to transformer_layer), just fewer full weight-tensor reads for
    // prompts longer than one token.
    let mut h = Vec::new();
    let mut pos = 0usize;
    let mut start = 0usize;
    while start < prompt_ids.len() {
        let end = (start + MAX_BATCH).min(prompt_ids.len());
        let mut xs: Vec<Vec<f32>> = prompt_ids[start..end].iter().map(|&t| embed_token(m, t)).collect();
        for il in 0..N_LAYER {
            xs = transformer_layer_batched(&xs, &m.layers[il], start, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = xs.last().unwrap().clone();
        pos = end - 1;
        start = end;
    }

    let embd_raw = m.g.f16_tensor_bytes("token_embd.weight");
    let mut history: Vec<i64> = prompt_ids.to_vec();
    let mut out = Vec::new();
    // `bonus` (below) already computed round N+1's real_next exactly, via
    // the same batched-rescore path lm_head_argmax_topk would use anyway --
    // recomputing it from scratch every round was pure waste. Only the very
    // first round has nothing carried over yet, so it still does one plain
    // single-position lookup.
    let mut pending_real: Option<i64> = None;

    'outer: while out.len() < max_new {
        // The next real token is always computed exactly (single-position or
        // carried over from the prior round's verify), regardless of
        // drafting -- this guarantees output can never be wrong even if
        // every draft misses.
        let real_next = match pending_real {
            Some(v) => v,
            None => {
                let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
                crate::model::lm_head_argmax_topk(m, qe, &hn, k)
            }
        };

        let mut probe = history.clone();
        probe.push(real_next);
        let max_draft_here = max_draft.min(MAX_BATCH - 1);
        let draft = find_draft(&probe, ngrams, max_draft_here);

        let mut batch_tokens = Vec::with_capacity(1 + draft.len());
        batch_tokens.push(real_next);
        batch_tokens.extend(&draft);
        let b = batch_tokens.len();

        let xs: Vec<Vec<f32>> = batch_tokens.iter().map(|&t| embed_token(m, t)).collect();
        let mut cur = xs;
        for il in 0..N_LAYER {
            cur = transformer_layer_batched(&cur, &m.layers[il], pos + 1, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }

        let hns: Vec<Vec<f32>> = cur.iter().map(|c| rmsnorm(c, &m.output_norm, RMS_EPS)).collect();
        let predicted = lm_head_argmax_topk_rescore_batched(embd_raw, qe, &hns, D_MODEL, VOCAB, k);

        let mut accept = 0usize;
        while accept < b - 1 && predicted[accept] == batch_tokens[accept + 1] {
            accept += 1;
        }
        let bonus = predicted[accept];
        let keep = accept + 1; // positions pos+1 .. pos+keep were real (verified) work

        out.push(real_next);
        history.push(real_next);
        if real_next == EOS_ID || out.len() >= max_new {
            break 'outer;
        }
        for j in 0..accept {
            out.push(draft[j]);
            history.push(draft[j]);
            if draft[j] == EOS_ID || out.len() >= max_new {
                break 'outer;
            }
        }

        // Drop the speculative KV cache entries for rejected draft
        // positions (batched layer pass optimistically appended all b).
        let drop = b - keep;
        if drop > 0 {
            for il in 0..N_LAYER {
                let new_len = kcaches[il].len() - drop * crate::model::KV_DIM;
                kcaches[il].truncate(new_len);
                vcaches[il].truncate(new_len);
            }
        }
        pos += keep;
        h = cur[accept].clone();
        pending_real = Some(bonus);
    }

    out.truncate(max_new);
    out
}

/// Steady-state decode tokens/sec via prompt-lookup decoding.
pub fn bench_decode_pld(m: &Model, prompt_ids: &[i64], qe: &QuantEmbd, k: usize, ngrams: &[usize], max_draft: usize) -> f64 {
    // PLD's throughput is data-dependent (depends on how often drafts hit),
    // so "steady state" here means: prime with the prompt, then time
    // generation of a fixed token budget from there -- comparable across
    // runs, unlike a fixed-step-count loop where a good draft run finishes
    // early.
    const BUDGET: usize = 60;
    let t0 = std::time::Instant::now();
    let out = generate_pld(m, qe, prompt_ids, BUDGET, k, ngrams, max_draft);
    let elapsed = t0.elapsed().as_secs_f64();
    out.len() as f64 / elapsed
}

/// Mem-mode sibling of generate_pld: identical algorithm (same draft/verify
/// logic, same correctness guarantee), but every embedding-table access --
/// input-token embedding during priming and decode, and the exact rescore
/// step -- goes through `reader` (explicit positioned file reads) instead
/// of the mmap. This is the -mem flag's actual mechanism: it trades some
/// per-row syscall overhead for never letting the ~656MB F16 table become
/// permanently resident, mirroring how the Mojo side avoided it (its
/// EmbdFileReader / _read_at, never mmap). The layer weights are unaffected
/// either way -- they're already zero-copy slices into the mmap regardless
/// of this flag (see model.rs's WRef), so -mem only changes the embedding
/// table's memory behavior, not the ternary weights'.
pub fn generate_pld_mem(
    m: &Model,
    reader: &EmbdFileReader,
    qe: &QuantEmbd,
    prompt_ids: &[i64],
    max_new: usize,
    k: usize,
    ngrams: &[usize],
    max_draft: usize,
) -> Vec<i64> {
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    let mut pos = 0usize;
    let mut start = 0usize;
    while start < prompt_ids.len() {
        let end = (start + MAX_BATCH).min(prompt_ids.len());
        let mut xs: Vec<Vec<f32>> = prompt_ids[start..end].iter().map(|&t| reader.read_row_f32(t, D_MODEL)).collect();
        for il in 0..N_LAYER {
            xs = transformer_layer_batched(&xs, &m.layers[il], start, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = xs.last().unwrap().clone();
        pos = end - 1;
        start = end;
    }

    let mut history: Vec<i64> = prompt_ids.to_vec();
    let mut out = Vec::new();
    let mut pending_real: Option<i64> = None;

    'outer: while out.len() < max_new {
        let real_next = match pending_real {
            Some(v) => v,
            None => {
                let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
                lm_head_argmax_topk_rescore_pread(reader, qe, &hn, D_MODEL, VOCAB, k)
            }
        };

        let mut probe = history.clone();
        probe.push(real_next);
        let max_draft_here = max_draft.min(MAX_BATCH - 1);
        let draft = find_draft(&probe, ngrams, max_draft_here);

        let mut batch_tokens = Vec::with_capacity(1 + draft.len());
        batch_tokens.push(real_next);
        batch_tokens.extend(&draft);
        let b = batch_tokens.len();

        let xs: Vec<Vec<f32>> = batch_tokens.iter().map(|&t| reader.read_row_f32(t, D_MODEL)).collect();
        let mut cur = xs;
        for il in 0..N_LAYER {
            cur = transformer_layer_batched(&cur, &m.layers[il], pos + 1, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }

        let hns: Vec<Vec<f32>> = cur.iter().map(|c| rmsnorm(c, &m.output_norm, RMS_EPS)).collect();
        let predicted = lm_head_argmax_topk_rescore_batched_pread(reader, qe, &hns, D_MODEL, VOCAB, k);

        let mut accept = 0usize;
        while accept < b - 1 && predicted[accept] == batch_tokens[accept + 1] {
            accept += 1;
        }
        let bonus = predicted[accept];
        let keep = accept + 1;

        out.push(real_next);
        history.push(real_next);
        if real_next == EOS_ID || out.len() >= max_new {
            break 'outer;
        }
        for j in 0..accept {
            out.push(draft[j]);
            history.push(draft[j]);
            if draft[j] == EOS_ID || out.len() >= max_new {
                break 'outer;
            }
        }

        let drop = b - keep;
        if drop > 0 {
            for il in 0..N_LAYER {
                let new_len = kcaches[il].len() - drop * crate::model::KV_DIM;
                kcaches[il].truncate(new_len);
                vcaches[il].truncate(new_len);
            }
        }
        pos += keep;
        h = cur[accept].clone();
        pending_real = Some(bonus);
    }

    out.truncate(max_new);
    out
}

/// Base-fp16 sibling of generate_pld: same draft/verify algorithm, same
/// batched layers, same batched prompt priming, same bonus-reuse -- but the
/// verify step uses the REAL F16 embedding table directly (lm_head_argmax /
/// lm_head_argmax_batched), no int8 quantization or rescore at all. This
/// isolates one question: does the int8+rescore lm_head actually earn its
/// keep over just batching the exact F16 scan the same way PLD already
/// batches everything else? Everything else in the pipeline (ternary
/// layers, attention, RoPE, batching, drafting) is identical between this
/// and generate_pld, so any speed difference is attributable to int8 vs f16
/// specifically, not some other variable.
pub fn generate_pld_fp16(
    m: &Model,
    prompt_ids: &[i64],
    max_new: usize,
    ngrams: &[usize],
    max_draft: usize,
) -> Vec<i64> {
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    let mut pos = 0usize;
    let mut start = 0usize;
    while start < prompt_ids.len() {
        let end = (start + MAX_BATCH).min(prompt_ids.len());
        let mut xs: Vec<Vec<f32>> = prompt_ids[start..end].iter().map(|&t| embed_token(m, t)).collect();
        for il in 0..N_LAYER {
            xs = transformer_layer_batched(&xs, &m.layers[il], start, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = xs.last().unwrap().clone();
        pos = end - 1;
        start = end;
    }

    let mut history: Vec<i64> = prompt_ids.to_vec();
    let mut out = Vec::new();
    let mut pending_real: Option<i64> = None;

    'outer: while out.len() < max_new {
        let real_next = match pending_real {
            Some(v) => v,
            None => {
                let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
                lm_head_argmax(m, &hn)
            }
        };

        let mut probe = history.clone();
        probe.push(real_next);
        let max_draft_here = max_draft.min(MAX_BATCH - 1);
        let draft = find_draft(&probe, ngrams, max_draft_here);

        let mut batch_tokens = Vec::with_capacity(1 + draft.len());
        batch_tokens.push(real_next);
        batch_tokens.extend(&draft);
        let b = batch_tokens.len();

        let xs: Vec<Vec<f32>> = batch_tokens.iter().map(|&t| embed_token(m, t)).collect();
        let mut cur = xs;
        for il in 0..N_LAYER {
            cur = transformer_layer_batched(&cur, &m.layers[il], pos + 1, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }

        let hns: Vec<Vec<f32>> = cur.iter().map(|c| rmsnorm(c, &m.output_norm, RMS_EPS)).collect();
        let predicted = lm_head_argmax_batched(m, &hns);

        let mut accept = 0usize;
        while accept < b - 1 && predicted[accept] == batch_tokens[accept + 1] {
            accept += 1;
        }
        let bonus = predicted[accept];
        let keep = accept + 1;

        out.push(real_next);
        history.push(real_next);
        if real_next == EOS_ID || out.len() >= max_new {
            break 'outer;
        }
        for j in 0..accept {
            out.push(draft[j]);
            history.push(draft[j]);
            if draft[j] == EOS_ID || out.len() >= max_new {
                break 'outer;
            }
        }

        let drop = b - keep;
        if drop > 0 {
            for il in 0..N_LAYER {
                let new_len = kcaches[il].len() - drop * crate::model::KV_DIM;
                kcaches[il].truncate(new_len);
                vcaches[il].truncate(new_len);
            }
        }
        pos += keep;
        h = cur[accept].clone();
        pending_real = Some(bonus);
    }

    out.truncate(max_new);
    out
}

/// PQ sibling of generate_pld: same draft/verify algorithm, same batched
/// layers/priming/bonus-reuse, but the lm_head verify step uses the
/// product-quantized embedding scan (see pq.rs) instead of int8. The query
/// side stays exact (fp16-derived hidden state); only the embedding
/// table's stored rows are PQ-compressed. Exact F16 rescore of the
/// candidates still guarantees the final answer can't be wrong regardless
/// of PQ's approximation quality -- only how good the candidate set is.
pub fn generate_pld_pq(
    m: &Model,
    reader: &crate::embd::EmbdFileReader,
    pq: &crate::pq::PqTable,
    prompt_ids: &[i64],
    max_new: usize,
    k: usize,
    ngrams: &[usize],
    max_draft: usize,
) -> Vec<i64> {
    use crate::pq::{lm_head_argmax_pq_rescore, lm_head_argmax_pq_rescore_batched};

    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    let mut pos = 0usize;
    let mut start = 0usize;
    while start < prompt_ids.len() {
        let end = (start + MAX_BATCH).min(prompt_ids.len());
        let mut xs: Vec<Vec<f32>> = prompt_ids[start..end].iter().map(|&t| reader.read_row_f32(t, D_MODEL)).collect();
        for il in 0..N_LAYER {
            xs = transformer_layer_batched(&xs, &m.layers[il], start, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = xs.last().unwrap().clone();
        pos = end - 1;
        start = end;
    }

    let mut history: Vec<i64> = prompt_ids.to_vec();
    let mut out = Vec::new();
    let mut pending_real: Option<i64> = None;

    'outer: while out.len() < max_new {
        let real_next = match pending_real {
            Some(v) => v,
            None => {
                let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
                lm_head_argmax_pq_rescore(reader, pq, &hn, VOCAB, k)
            }
        };

        let mut probe = history.clone();
        probe.push(real_next);
        let max_draft_here = max_draft.min(MAX_BATCH - 1);
        let draft = find_draft(&probe, ngrams, max_draft_here);

        let mut batch_tokens = Vec::with_capacity(1 + draft.len());
        batch_tokens.push(real_next);
        batch_tokens.extend(&draft);
        let b = batch_tokens.len();

        let xs: Vec<Vec<f32>> = batch_tokens.iter().map(|&t| reader.read_row_f32(t, D_MODEL)).collect();
        let mut cur = xs;
        for il in 0..N_LAYER {
            cur = transformer_layer_batched(&cur, &m.layers[il], pos + 1, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }

        let hns: Vec<Vec<f32>> = cur.iter().map(|c| rmsnorm(c, &m.output_norm, RMS_EPS)).collect();
        let predicted = lm_head_argmax_pq_rescore_batched(reader, pq, &hns, VOCAB, k);

        let mut accept = 0usize;
        while accept < b - 1 && predicted[accept] == batch_tokens[accept + 1] {
            accept += 1;
        }
        let bonus = predicted[accept];
        let keep = accept + 1;

        out.push(real_next);
        history.push(real_next);
        if real_next == EOS_ID || out.len() >= max_new {
            break 'outer;
        }
        for j in 0..accept {
            out.push(draft[j]);
            history.push(draft[j]);
            if draft[j] == EOS_ID || out.len() >= max_new {
                break 'outer;
            }
        }

        let drop = b - keep;
        if drop > 0 {
            for il in 0..N_LAYER {
                let new_len = kcaches[il].len() - drop * crate::model::KV_DIM;
                kcaches[il].truncate(new_len);
                vcaches[il].truncate(new_len);
            }
        }
        pos += keep;
        h = cur[accept].clone();
        pending_real = Some(bonus);
    }

    out.truncate(max_new);
    out
}

/// OPQ sibling of generate_pld_pq: identical structure, but the lm_head
/// scan uses OPQ (rotation-learned Product Quantization, see opq.rs)
/// instead of plain PQ -- same query-stays-exact / table-only-approximate
/// design, same exact F16 rescore safety net.
pub fn generate_pld_opq(
    m: &Model,
    reader: &crate::embd::EmbdFileReader,
    t: &crate::opq::OpqTable,
    prompt_ids: &[i64],
    max_new: usize,
    k: usize,
    ngrams: &[usize],
    max_draft: usize,
) -> Vec<i64> {
    use crate::opq::{lm_head_argmax_opq_rescore, lm_head_argmax_opq_rescore_batched};

    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    let mut pos = 0usize;
    let mut start = 0usize;
    while start < prompt_ids.len() {
        let end = (start + MAX_BATCH).min(prompt_ids.len());
        let mut xs: Vec<Vec<f32>> = prompt_ids[start..end].iter().map(|&tk| reader.read_row_f32(tk, D_MODEL)).collect();
        for il in 0..N_LAYER {
            xs = transformer_layer_batched(&xs, &m.layers[il], start, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = xs.last().unwrap().clone();
        pos = end - 1;
        start = end;
    }

    let mut history: Vec<i64> = prompt_ids.to_vec();
    let mut out = Vec::new();
    let mut pending_real: Option<i64> = None;

    'outer: while out.len() < max_new {
        let real_next = match pending_real {
            Some(v) => v,
            None => {
                let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
                lm_head_argmax_opq_rescore(reader, t, &hn, VOCAB, k)
            }
        };

        let mut probe = history.clone();
        probe.push(real_next);
        let max_draft_here = max_draft.min(MAX_BATCH - 1);
        let draft = find_draft(&probe, ngrams, max_draft_here);

        let mut batch_tokens = Vec::with_capacity(1 + draft.len());
        batch_tokens.push(real_next);
        batch_tokens.extend(&draft);
        let b = batch_tokens.len();

        let xs: Vec<Vec<f32>> = batch_tokens.iter().map(|&tk| reader.read_row_f32(tk, D_MODEL)).collect();
        let mut cur = xs;
        for il in 0..N_LAYER {
            cur = transformer_layer_batched(&cur, &m.layers[il], pos + 1, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }

        let hns: Vec<Vec<f32>> = cur.iter().map(|c| rmsnorm(c, &m.output_norm, RMS_EPS)).collect();
        let predicted = lm_head_argmax_opq_rescore_batched(reader, t, &hns, VOCAB, k);

        let mut accept = 0usize;
        while accept < b - 1 && predicted[accept] == batch_tokens[accept + 1] {
            accept += 1;
        }
        let bonus = predicted[accept];
        let keep = accept + 1;

        out.push(real_next);
        history.push(real_next);
        if real_next == EOS_ID || out.len() >= max_new {
            break 'outer;
        }
        for j in 0..accept {
            out.push(draft[j]);
            history.push(draft[j]);
            if draft[j] == EOS_ID || out.len() >= max_new {
                break 'outer;
            }
        }

        let drop = b - keep;
        if drop > 0 {
            for il in 0..N_LAYER {
                let new_len = kcaches[il].len() - drop * crate::model::KV_DIM;
                kcaches[il].truncate(new_len);
                vcaches[il].truncate(new_len);
            }
        }
        pos += keep;
        h = cur[accept].clone();
        pending_real = Some(bonus);
    }

    out.truncate(max_new);
    out
}
