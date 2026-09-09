// BitNet b1.58 2B4T transformer, ported 1:1 from model.mojo. Same flow,
// same hyperparameters, same tied-embedding lm_head. Row-level parallelism
// (bitlinear matmuls, lm_head argmax scan) goes through Rayon.

use crate::bitlinear::{bitlinear_forward, bitlinear_forward_batched, bitlinear_forward_fused};
use crate::gguf::{f16_bits_to_f32, Gguf};
use std::arch::x86_64::*;

pub const N_LAYER: usize = 30;
pub const D_MODEL: usize = 2560;
pub const D_FF: usize = 6912;
pub const N_HEAD: usize = 20;
pub const N_HEAD_KV: usize = 5;
pub const HEAD_DIM: usize = 128;
pub const RMS_EPS: f32 = 1e-5;
pub const ROPE_DIM: usize = 128;
pub const ROPE_THETA: f32 = 500000.0;
pub const VOCAB: usize = 128256;
pub const EOS_ID: i64 = 128001;
pub const GQA_GROUP: usize = N_HEAD / N_HEAD_KV; // 4
pub const KV_DIM: usize = N_HEAD_KV * HEAD_DIM; // 640

pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    x.iter().zip(weight).map(|(&xi, &wi)| xi * inv * wi).collect()
}

pub fn rope_inv_freqs() -> Vec<f32> {
    let half = ROPE_DIM / 2;
    (0..half)
        .map(|ic| ROPE_THETA.powf(-2.0 * ic as f32 / ROPE_DIM as f32))
        .collect()
}

/// In-place NeoX RoPE on a contiguous [n_heads * HEAD_DIM] vector: rotates
/// pair (ic, ic + HEAD_DIM/2) within each head by angle pos * inv_freq[ic].
pub fn rope_apply_neox(v: &mut [f32], pos: usize, n_heads: usize, inv_freq: &[f32]) {
    let half = ROPE_DIM / 2;
    for h in 0..n_heads {
        let base = h * HEAD_DIM;
        for ic in 0..half {
            let theta = pos as f32 * inv_freq[ic];
            let c = theta.cos();
            let s = theta.sin();
            let x0 = v[base + ic];
            let x1 = v[base + ic + half];
            v[base + ic] = x0 * c - x1 * s;
            v[base + ic + half] = x0 * s + x1 * c;
        }
    }
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&xi| xi / (1.0 + (-xi).exp())).collect()
}

/// Grouped-query scaled-dot-product attention for one decode step. Query
/// head h reads kv head h/GQA_GROUP. Causal is structural: this handles
/// exactly one new query position against a cache holding positions
/// 0..t_len-1, so an unmasked softmax over all t_len keys is already causal
/// -- this does NOT generalize to multi-position prefill without a mask.
pub fn sdpa_gqa(q: &[f32], k_cache: &[f32], v_cache: &[f32], t_len: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0f32; N_HEAD * HEAD_DIM];
    let mut scores = vec![0f32; t_len];

    for h in 0..N_HEAD {
        let kvh = h / GQA_GROUP;
        let qoff = h * HEAD_DIM;

        let mut smax = f32::NEG_INFINITY;
        for j in 0..t_len {
            let koff = (j * N_HEAD_KV + kvh) * HEAD_DIM;
            let mut dot = 0f32;
            for d in 0..HEAD_DIM {
                dot += q[qoff + d] * k_cache[koff + d];
            }
            dot *= scale;
            scores[j] = dot;
            if dot > smax {
                smax = dot;
            }
        }

        let mut denom = 0f32;
        for j in 0..t_len {
            let e = (scores[j] - smax).exp();
            scores[j] = e;
            denom += e;
        }
        let inv = 1.0 / denom;

        for j in 0..t_len {
            let w = scores[j] * inv;
            let voff = (j * N_HEAD_KV + kvh) * HEAD_DIM;
            for d in 0..HEAD_DIM {
                out[qoff + d] += w * v_cache[voff + d];
            }
        }
    }
    out
}

/// (mmap byte offset, byte len) for one I2_S tensor -- resolved back to a
/// `&[u8]` via `Gguf::slice_at` at use time instead of being copied out of
/// the mmap at load time. Small and Copy, so LayerWeights stays cheap to
/// move around despite no longer owning the weight bytes.
#[derive(Clone, Copy)]
pub struct WRef {
    off: usize,
    len: usize,
}

impl WRef {
    #[inline]
    pub fn slice<'g>(&self, g: &'g Gguf) -> &'g [u8] {
        g.slice_at(self.off, self.len)
    }
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: WRef,
    pub wq_s: f32,
    pub wk: WRef,
    pub wk_s: f32,
    pub wv: WRef,
    pub wv_s: f32,
    pub wo: WRef,
    pub wo_s: f32,
    pub attn_sub_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: WRef,
    pub w_gate_s: f32,
    pub w_up: WRef,
    pub w_up_s: f32,
    pub w_down: WRef,
    pub w_down_s: f32,
    pub ffn_sub_norm: Vec<f32>,
}

pub fn load_layer(g: &Gguf, il: usize) -> LayerWeights {
    let p = format!("blk.{}.", il);
    let (wq_off, wq_len, wq_s) = g.i2s_range(&format!("{}attn_q.weight", p));
    let (wk_off, wk_len, wk_s) = g.i2s_range(&format!("{}attn_k.weight", p));
    let (wv_off, wv_len, wv_s) = g.i2s_range(&format!("{}attn_v.weight", p));
    let (wo_off, wo_len, wo_s) = g.i2s_range(&format!("{}attn_output.weight", p));
    let (wg_off, wg_len, w_gate_s) = g.i2s_range(&format!("{}ffn_gate.weight", p));
    let (wu_off, wu_len, w_up_s) = g.i2s_range(&format!("{}ffn_up.weight", p));
    let (wd_off, wd_len, w_down_s) = g.i2s_range(&format!("{}ffn_down.weight", p));
    LayerWeights {
        attn_norm: g.load_f32(&format!("{}attn_norm.weight", p)),
        wq: WRef { off: wq_off, len: wq_len },
        wq_s,
        wk: WRef { off: wk_off, len: wk_len },
        wk_s,
        wv: WRef { off: wv_off, len: wv_len },
        wv_s,
        wo: WRef { off: wo_off, len: wo_len },
        wo_s,
        attn_sub_norm: g.load_f32(&format!("{}attn_sub_norm.weight", p)),
        ffn_norm: g.load_f32(&format!("{}ffn_norm.weight", p)),
        w_gate: WRef { off: wg_off, len: wg_len },
        w_gate_s,
        w_up: WRef { off: wu_off, len: wu_len },
        w_up_s,
        w_down: WRef { off: wd_off, len: wd_len },
        w_down_s,
        ffn_sub_norm: g.load_f32(&format!("{}ffn_sub_norm.weight", p)),
    }
}

/// BitNet FFN sub-block (SiLU-gated, SubLN before ffn_down):
///   f = rmsnorm(x, ffn_norm); f = silu(gate(f)) * up(f);
///   f = rmsnorm(f, ffn_sub_norm); return down(f)
pub fn ffn_block(x: &[f32], lw: &LayerWeights, gguf: &Gguf) -> Vec<f32> {
    let f = rmsnorm(x, &lw.ffn_norm, RMS_EPS);
    let mut gu = bitlinear_forward_fused(
        &[
            (lw.w_gate.slice(gguf), lw.w_gate_s, D_FF),
            (lw.w_up.slice(gguf), lw.w_up_s, D_FF),
        ],
        &f,
        D_MODEL,
    )
    .into_iter();
    let gate_out = gu.next().unwrap();
    let u = gu.next().unwrap();
    let sg = silu(&gate_out);
    let hid: Vec<f32> = sg.iter().zip(&u).map(|(&a, &b)| a * b).collect();
    let hn = rmsnorm(&hid, &lw.ffn_sub_norm, RMS_EPS);
    bitlinear_forward(lw.w_down.slice(gguf), lw.w_down_s, &hn, D_FF, D_MODEL)
}

/// One BitNet block: attn_norm -> qkv -> RoPE(q,k) -> cache -> sdpa ->
/// attn_sub_norm -> wo -> residual(=ffn_inp) -> ffn_block -> residual.
pub fn transformer_layer(
    h: &[f32],
    lw: &LayerWeights,
    pos: usize,
    kcache: &mut Vec<f32>,
    vcache: &mut Vec<f32>,
    inv_freq: &[f32],
    g: &Gguf,
) -> Vec<f32> {
    let a = rmsnorm(h, &lw.attn_norm, RMS_EPS);
    let mut qkv = bitlinear_forward_fused(
        &[
            (lw.wq.slice(g), lw.wq_s, D_MODEL),
            (lw.wk.slice(g), lw.wk_s, KV_DIM),
            (lw.wv.slice(g), lw.wv_s, KV_DIM),
        ],
        &a,
        D_MODEL,
    )
    .into_iter();
    let mut q = qkv.next().unwrap();
    let mut k = qkv.next().unwrap();
    let v = qkv.next().unwrap();

    rope_apply_neox(&mut q, pos, N_HEAD, inv_freq);
    rope_apply_neox(&mut k, pos, N_HEAD_KV, inv_freq);

    kcache.extend(k);
    vcache.extend(v);
    let t_len = kcache.len() / KV_DIM;

    let o = sdpa_gqa(&q, kcache, vcache, t_len);
    let o = rmsnorm(&o, &lw.attn_sub_norm, RMS_EPS);
    let o = bitlinear_forward(lw.wo.slice(g), lw.wo_s, &o, D_MODEL, D_MODEL);

    let ffn_inp: Vec<f32> = h.iter().zip(&o).map(|(&hi, &oi)| hi + oi).collect();
    let ffn_out = ffn_block(&ffn_inp, lw, g);
    ffn_inp.iter().zip(&ffn_out).map(|(&fi, &foi)| fi + foi).collect()
}

/// Same block as transformer_layer, but for B positions at once (prompt-
/// lookup decoding's draft-verify batch). Math is IDENTICAL per position to
/// calling transformer_layer B times in sequence -- q/k/v/wo/gate/up/down
/// projections only depend on that position's own (already-known) input, so
/// they're computed for all B positions in one batched dispatch each;
/// attention is the one part with a real cross-position dependency (position
/// i needs position i-1's K/V in this same layer's cache already appended),
/// so it stays a plain sequential loop over the (cheap, ~3.5% of decode
/// time) sdpa_gqa calls, extending kcache/vcache as it goes -- same
/// causal-append pattern transformer_layer already uses one position at a
/// time. xs[i] is the input at position start_pos+i.
pub fn transformer_layer_batched(
    xs: &[Vec<f32>],
    lw: &LayerWeights,
    start_pos: usize,
    kcache: &mut Vec<f32>,
    vcache: &mut Vec<f32>,
    inv_freq: &[f32],
    gguf: &Gguf,
) -> Vec<Vec<f32>> {
    let b = xs.len();
    let a: Vec<Vec<f32>> = xs.iter().map(|x| rmsnorm(x, &lw.attn_norm, RMS_EPS)).collect();

    let q_all = bitlinear_forward_batched(lw.wq.slice(gguf), lw.wq_s, &a, D_MODEL, D_MODEL);
    let k_all = bitlinear_forward_batched(lw.wk.slice(gguf), lw.wk_s, &a, D_MODEL, KV_DIM);
    let v_all = bitlinear_forward_batched(lw.wv.slice(gguf), lw.wv_s, &a, D_MODEL, KV_DIM);

    let mut qs = q_all;
    let mut ks = k_all;
    let vs = v_all;
    for i in 0..b {
        rope_apply_neox(&mut qs[i], start_pos + i, N_HEAD, inv_freq);
        rope_apply_neox(&mut ks[i], start_pos + i, N_HEAD_KV, inv_freq);
    }

    let mut os = Vec::with_capacity(b);
    for i in 0..b {
        kcache.extend(ks[i].iter().copied());
        vcache.extend(vs[i].iter().copied());
        let t_len = kcache.len() / KV_DIM;
        os.push(sdpa_gqa(&qs[i], kcache, vcache, t_len));
    }

    let on: Vec<Vec<f32>> = os.iter().map(|o| rmsnorm(o, &lw.attn_sub_norm, RMS_EPS)).collect();
    let ao_all = bitlinear_forward_batched(lw.wo.slice(gguf), lw.wo_s, &on, D_MODEL, D_MODEL);

    let ffn_inp: Vec<Vec<f32>> = xs
        .iter()
        .zip(&ao_all)
        .map(|(x, ao)| x.iter().zip(ao).map(|(&xi, &aoi)| xi + aoi).collect())
        .collect();

    let f: Vec<Vec<f32>> = ffn_inp.iter().map(|fi| rmsnorm(fi, &lw.ffn_norm, RMS_EPS)).collect();
    let g_all = bitlinear_forward_batched(lw.w_gate.slice(gguf), lw.w_gate_s, &f, D_MODEL, D_FF);
    let u_all = bitlinear_forward_batched(lw.w_up.slice(gguf), lw.w_up_s, &f, D_MODEL, D_FF);
    let hid: Vec<Vec<f32>> = g_all
        .iter()
        .zip(&u_all)
        .map(|(g, u)| {
            let sg = silu(g);
            sg.iter().zip(u).map(|(&s, &uu)| s * uu).collect()
        })
        .collect();
    let hn: Vec<Vec<f32>> = hid.iter().map(|h| rmsnorm(h, &lw.ffn_sub_norm, RMS_EPS)).collect();
    let dn_all = bitlinear_forward_batched(lw.w_down.slice(gguf), lw.w_down_s, &hn, D_FF, D_MODEL);

    ffn_inp
        .iter()
        .zip(&dn_all)
        .map(|(fi, dn)| fi.iter().zip(dn).map(|(&fii, &dni)| fii + dni).collect())
        .collect()
}

pub struct Model {
    pub g: Gguf,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
}

pub fn load_model(path: &str) -> Model {
    let g = crate::gguf::open_gguf(path);
    let layers: Vec<LayerWeights> = (0..N_LAYER).map(|il| load_layer(&g, il)).collect();
    let output_norm = g.load_f32("output_norm.weight");
    Model { g, layers, output_norm }
}

/// One row of the tied F16 token_embd.weight, read straight from the mmap
/// (zero-copy -- no owned blob needed the way the Mojo port keeps one).
pub fn embed_token(m: &Model, token_id: i64) -> Vec<f32> {
    let raw = m.g.f16_tensor_bytes("token_embd.weight");
    let off = token_id as usize * D_MODEL * 2;
    (0..D_MODEL)
        .map(|d| {
            let b = off + d * 2;
            f16_bits_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]))
        })
        .collect()
}

/// One row's dot product against h_normed, F16 widened 8-wide via hardware
/// F16C (vcvtph2ps) + FMA instead of a scalar bit-twiddle per element --
/// this loop runs 128256 times per generated token, so scalar widening
/// dominates decode time.
#[target_feature(enable = "f16c,fma,avx2")]
pub(crate) unsafe fn lm_dot_row_f16c(row: &[u8], h: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut d = 0usize;
    while d + 8 <= D_MODEL {
        let bits = _mm_loadu_si128(row.as_ptr().add(d * 2) as *const __m128i);
        let ev = _mm256_cvtph_ps(bits);
        let hv = _mm256_loadu_ps(h.as_ptr().add(d));
        acc = _mm256_fmadd_ps(ev, hv, acc);
        d += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut total: f32 = tmp.iter().sum();
    while d < D_MODEL {
        let b = d * 2;
        total += f16_bits_to_f32(u16::from_le_bytes([row[b], row[b + 1]])) * h[d];
        d += 1;
    }
    total
}

/// logits[t] = row_t(token_embd) . h_normed ; return argmax. Tied weights,
/// scanned in parallel across the 128256-row vocab via Rayon, each row dot
/// vectorized with F16C+FMA.
pub fn lm_head_argmax(m: &Model, h_normed: &[f32]) -> i64 {
    let raw = m.g.f16_tensor_bytes("token_embd.weight");
    crate::pool::dispatch_lmhead(raw, h_normed, D_MODEL, VOCAB)
}

/// Batched sibling of lm_head_argmax: B hidden states verified against the
/// real F16 embedding table (no int8 approximation) in one pool sync --
/// PLD's verify step with the exact/base lm_head instead of int8+rescore,
/// to see whether the int8 shortcut is actually earning its keep.
pub fn lm_head_argmax_batched(m: &Model, h_normed_batch: &[Vec<f32>]) -> Vec<i64> {
    let raw = m.g.f16_tensor_bytes("token_embd.weight");
    let b_count = h_normed_batch.len();
    let mut flat = vec![0f32; b_count * D_MODEL];
    for (b, h) in h_normed_batch.iter().enumerate() {
        flat[b * D_MODEL..(b + 1) * D_MODEL].copy_from_slice(h);
    }
    crate::pool::dispatch_lmhead_batched(raw, &flat, D_MODEL, VOCAB, b_count)
}

pub fn generate(m: &Model, prompt_ids: &[i64], max_new: usize) -> Vec<i64> {
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    let mut pos = 0usize;
    for (p, &tok) in prompt_ids.iter().enumerate() {
        let mut x = embed_token(m, tok);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], p, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
        pos = p;
    }

    let mut out = Vec::new();
    for _step in 0..max_new {
        let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
        let nxt = lm_head_argmax(m, &hn);
        out.push(nxt);
        if nxt == EOS_ID {
            break;
        }
        pos += 1;
        let mut x = embed_token(m, nxt);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], pos, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
    }
    out
}

/// Same as lm_head_argmax but via the int8-quantized-embedding top-K
/// rescore path (see embd.rs) instead of a full F16 scan.
pub fn lm_head_argmax_topk(m: &Model, qe: &crate::embd::QuantEmbd, h_normed: &[f32], k: usize) -> i64 {
    let raw = m.g.f16_tensor_bytes("token_embd.weight");
    crate::embd::lm_head_argmax_topk_rescore(raw, qe, h_normed, D_MODEL, VOCAB, k)
}

pub fn generate_topk(m: &Model, qe: &crate::embd::QuantEmbd, prompt_ids: &[i64], max_new: usize, k: usize) -> Vec<i64> {
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    let mut pos = 0usize;
    for (p, &tok) in prompt_ids.iter().enumerate() {
        let mut x = embed_token(m, tok);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], p, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
        pos = p;
    }

    let mut out = Vec::new();
    for _step in 0..max_new {
        let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
        let nxt = lm_head_argmax_topk(m, qe, &hn, k);
        out.push(nxt);
        if nxt == EOS_ID {
            break;
        }
        pos += 1;
        let mut x = embed_token(m, nxt);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], pos, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
    }
    out
}

/// Steady-state decode tokens/sec via the int8 top-K rescore lm_head.
pub fn bench_decode_topk(m: &Model, prompt_ids: &[i64], qe: &crate::embd::QuantEmbd, k: usize) -> f64 {
    const WARMUP: usize = 3;
    const STEADY: usize = 30;
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    for (p, &tok) in prompt_ids.iter().enumerate() {
        let mut x = embed_token(m, tok);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], p, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
    }
    let mut pos = prompt_ids.len() - 1;

    let mut t_start = std::time::Instant::now();
    let mut t_lm_total = std::time::Duration::ZERO;
    let mut t_layers_total = std::time::Duration::ZERO;
    for step in 0..(WARMUP + STEADY) {
        if step == WARMUP {
            t_start = std::time::Instant::now();
            t_lm_total = std::time::Duration::ZERO;
            t_layers_total = std::time::Duration::ZERO;
        }
        let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
        let t0 = std::time::Instant::now();
        let nxt = lm_head_argmax_topk(m, qe, &hn, k);
        if step >= WARMUP {
            t_lm_total += t0.elapsed();
        }
        pos += 1;
        let mut x = embed_token(m, nxt);
        let t0 = std::time::Instant::now();
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], pos, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        if step >= WARMUP {
            t_layers_total += t0.elapsed();
        }
        h = x;
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    println!(
        "  [bench_decode_topk] lm_head: {:.3} ms/step   layers: {:.3} ms/step   (of {:.3} ms/step total)",
        t_lm_total.as_secs_f64() * 1000.0 / STEADY as f64,
        t_layers_total.as_secs_f64() * 1000.0 / STEADY as f64,
        elapsed * 1000.0 / STEADY as f64,
    );
    STEADY as f64 / elapsed
}

/// Steady-state decode tokens/sec: prime prompt, drop 3 warm-up tokens, time
/// the next 30. Directly comparable to the Mojo engine's bench_decode.
pub fn bench_decode(m: &Model, prompt_ids: &[i64]) -> f64 {
    const WARMUP: usize = 3;
    const STEADY: usize = 30;
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    for (p, &tok) in prompt_ids.iter().enumerate() {
        let mut x = embed_token(m, tok);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], p, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
    }
    let mut pos = prompt_ids.len() - 1;

    let mut t_start = std::time::Instant::now();
    for step in 0..(WARMUP + STEADY) {
        if step == WARMUP {
            t_start = std::time::Instant::now();
        }
        let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
        let nxt = lm_head_argmax(m, &hn);
        pos += 1;
        let mut x = embed_token(m, nxt);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], pos, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    STEADY as f64 / elapsed
}

/// Per-section wall-clock over N steady decode steps, to see where time
/// goes. Mirrors model.mojo's profile_decode section-by-section.
pub fn profile_decode(m: &Model, prompt_ids: &[i64]) {
    const WARMUP: usize = 3;
    const STEPS: usize = 15;
    let inv = rope_inv_freqs();
    let mut kcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();
    let mut vcaches: Vec<Vec<f32>> = (0..N_LAYER).map(|_| Vec::new()).collect();

    let mut h = Vec::new();
    for (p, &tok) in prompt_ids.iter().enumerate() {
        let mut x = embed_token(m, tok);
        for il in 0..N_LAYER {
            x = transformer_layer(&x, &m.layers[il], p, &mut kcaches[il], &mut vcaches[il], &inv, &m.g);
        }
        h = x;
    }
    let mut pos = prompt_ids.len() - 1;

    let mut t_lm = std::time::Duration::ZERO;
    let mut t_norm = std::time::Duration::ZERO;
    let mut t_qkv = std::time::Duration::ZERO;
    let mut t_rope = std::time::Duration::ZERO;
    let mut t_sdpa = std::time::Duration::ZERO;
    let mut t_wo = std::time::Duration::ZERO;
    let mut t_gate = std::time::Duration::ZERO;
    let mut t_up = std::time::Duration::ZERO;
    let mut t_down = std::time::Duration::ZERO;
    let mut t_res = std::time::Duration::ZERO;

    for step in 0..(WARMUP + STEPS) {
        let acc = step >= WARMUP;
        let t0 = std::time::Instant::now();
        let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
        let nxt = lm_head_argmax(m, &hn);
        if acc {
            t_lm += t0.elapsed();
        }
        pos += 1;
        let mut x = embed_token(m, nxt);
        for il in 0..N_LAYER {
            let lw = &m.layers[il];
            let t0 = std::time::Instant::now();
            let a = rmsnorm(&x, &lw.attn_norm, RMS_EPS);
            if acc {
                t_norm += t0.elapsed();
            }
            let t0 = std::time::Instant::now();
            let mut q = bitlinear_forward(lw.wq.slice(&m.g), lw.wq_s, &a, D_MODEL, D_MODEL);
            let mut k = bitlinear_forward(lw.wk.slice(&m.g), lw.wk_s, &a, D_MODEL, KV_DIM);
            let v = bitlinear_forward(lw.wv.slice(&m.g), lw.wv_s, &a, D_MODEL, KV_DIM);
            if acc {
                t_qkv += t0.elapsed();
            }
            let t0 = std::time::Instant::now();
            rope_apply_neox(&mut q, pos, N_HEAD, &inv);
            rope_apply_neox(&mut k, pos, N_HEAD_KV, &inv);
            if acc {
                t_rope += t0.elapsed();
            }
            kcaches[il].extend(k);
            vcaches[il].extend(v);
            let t_len = kcaches[il].len() / KV_DIM;
            let t0 = std::time::Instant::now();
            let o = sdpa_gqa(&q, &kcaches[il], &vcaches[il], t_len);
            if acc {
                t_sdpa += t0.elapsed();
            }
            let t0 = std::time::Instant::now();
            let on = rmsnorm(&o, &lw.attn_sub_norm, RMS_EPS);
            let ao = bitlinear_forward(lw.wo.slice(&m.g), lw.wo_s, &on, D_MODEL, D_MODEL);
            if acc {
                t_wo += t0.elapsed();
            }
            let fi: Vec<f32> = x.iter().zip(&ao).map(|(&xi, &aoi)| xi + aoi).collect();
            let fnr = rmsnorm(&fi, &lw.ffn_norm, RMS_EPS);
            let t0 = std::time::Instant::now();
            let g = bitlinear_forward(lw.w_gate.slice(&m.g), lw.w_gate_s, &fnr, D_MODEL, D_FF);
            if acc {
                t_gate += t0.elapsed();
            }
            let t0 = std::time::Instant::now();
            let u = bitlinear_forward(lw.w_up.slice(&m.g), lw.w_up_s, &fnr, D_MODEL, D_FF);
            if acc {
                t_up += t0.elapsed();
            }
            let sg = silu(&g);
            let hid: Vec<f32> = sg.iter().zip(&u).map(|(&a, &b)| a * b).collect();
            let hnf = rmsnorm(&hid, &lw.ffn_sub_norm, RMS_EPS);
            let t0 = std::time::Instant::now();
            let dn = bitlinear_forward(lw.w_down.slice(&m.g), lw.w_down_s, &hnf, D_FF, D_MODEL);
            if acc {
                t_down += t0.elapsed();
            }
            let t0 = std::time::Instant::now();
            let nx: Vec<f32> = fi.iter().zip(&dn).map(|(&fii, &dni)| fii + dni).collect();
            if acc {
                t_res += t0.elapsed();
            }
            x = nx;
        }
        h = x;
    }

    let tot = t_lm + t_norm + t_qkv + t_rope + t_sdpa + t_wo + t_gate + t_up + t_down + t_res;
    println!("--- profile: {} steady steps, per-section (ms / pct) ---", STEPS);
    let prow = |name: &str, v: std::time::Duration| {
        println!(
            "{:14}: {:.3} ms   {:.2} pct",
            name,
            v.as_secs_f64() * 1000.0,
            100.0 * v.as_secs_f64() / tot.as_secs_f64()
        );
    };
    prow("lm_head_argmax", t_lm);
    prow("qkv proj", t_qkv);
    prow("ffn_gate proj", t_gate);
    prow("ffn_up proj", t_up);
    prow("ffn_down proj", t_down);
    prow("wo proj+subN", t_wo);
    prow("sdpa", t_sdpa);
    prow("rmsnorm(attn)", t_norm);
    prow("rope", t_rope);
    prow("residual adds", t_res);
    println!(
        "total accounted: {:.3} ms over {} steps",
        tot.as_secs_f64() * 1000.0,
        STEPS
    );
}
