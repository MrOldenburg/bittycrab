// Ternary weight x int8 activation dot product, AVX-VNNI accelerated,
// ported from bitlinear.mojo. Same I2_S packing (4 ternary weights/byte,
// 2 bits each, MSB-first, map2bit = [-1,0,1,0]), same unsigned-dot-minus-
// sum(act) correction, same 128-elements-per-step / 4-plane SIMD kernel --
// built with the 4-independent-accumulator ILP fix from the start (already
// validated: pure integer reassociation, ~2.6x on the raw kernel).
// Row-level parallelism via Rayon instead of a hand-rolled thread pool.

use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn dpbusd(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    _mm256_dpbusd_avx_epi32(acc, a, b)
}

/// Unsigned ternary dot over one row: sum_{i}(code(i)*act(pair(i))), code in
/// {0,1,2}. Caller applies -sum(act) to get the signed ternary dot.
///
/// Within each 128-element block, plane p (bits extracted by shift 6-2p from
/// all 32 bytes) pairs with the CONTIGUOUS activation slice
/// [base+p*32 .. base+p*32+32) -- natural per-element index order, no
/// permutation on the activation side. This does not compute sum_i(w_i*a_i)
/// in natural order; it's the exact pairing bitnet.cpp's AVX2 I2_S kernel
/// uses and that the real GGUF-packed weight bytes are validated against
/// (see bitlinear.mojo's _ternary_dot_ptr / _plane_major_ref) -- ported
/// verbatim rather than re-derived, since that's the empirically-verified
/// ground truth against the real model.
#[target_feature(enable = "avxvnni")]
pub(crate) unsafe fn ternary_dot_row(packed: &[u8], acts: &[i8], n: usize) -> i32 {
    let mask3 = _mm256_set1_epi8(3);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    let n_blk = n / 128;
    let pptr = packed.as_ptr();
    let aptr = acts.as_ptr() as *const u8;
    for bl in 0..n_blk {
        let pk = _mm256_loadu_si256(pptr.add(bl * 32) as *const __m256i);
        let base = bl * 128;
        let a0 = _mm256_loadu_si256(aptr.add(base) as *const __m256i);
        let a1 = _mm256_loadu_si256(aptr.add(base + 32) as *const __m256i);
        let a2 = _mm256_loadu_si256(aptr.add(base + 64) as *const __m256i);
        let a3 = _mm256_loadu_si256(aptr.add(base + 96) as *const __m256i);

        let p0 = _mm256_and_si256(_mm256_srli_epi16(pk, 6), mask3);
        let p1 = _mm256_and_si256(_mm256_srli_epi16(pk, 4), mask3);
        let p2 = _mm256_and_si256(_mm256_srli_epi16(pk, 2), mask3);
        let p3 = _mm256_and_si256(pk, mask3);

        acc0 = dpbusd(acc0, p0, a0);
        acc1 = dpbusd(acc1, p1, a1);
        acc2 = dpbusd(acc2, p2, a2);
        acc3 = dpbusd(acc3, p3, a3);
    }

    let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
    let mut total: i32 = tmp.iter().sum();

    for i in (n_blk * 128)..n {
        let byte_idx = i / 4;
        let bit_pos = 6 - 2 * (i % 4);
        let code = ((packed[byte_idx] >> bit_pos) & 0x03) as i32;
        total += code * acts[i] as i32;
    }
    total
}

/// Correctness oracle, deliberately dumb -- mirrors ternary_dot_scalar in
/// bitlinear.mojo, used only by the test suite (tail-only, n<128).
#[allow(dead_code)]
pub fn ternary_dot_scalar(packed: &[u8], acts: &[i8], n: usize) -> i32 {
    let mut acc = 0i32;
    for i in 0..n {
        let byte_idx = i / 4;
        let bit_pos = 6 - 2 * (i % 4);
        let code = (packed[byte_idx] >> bit_pos) & 0x03;
        let w: i32 = match code {
            0 => -1,
            2 => 1,
            _ => 0,
        };
        acc += w * acts[i] as i32;
    }
    acc
}

/// Signed ternary dot with the kernel's actual plane-major activation
/// pairing: weight (byte b, plane p) pairs with act[base + p*32 + b].
/// Mirrors bitlinear.mojo's _plane_major_ref -- the correct oracle for
/// full 128-blocks (ternary_dot_scalar only agrees with the kernel on the
/// tail, n<128).
#[cfg(test)]
fn plane_major_ref(packed: &[u8], acts: &[i8], n: usize) -> i32 {
    let signed_code = |k: usize| -> i32 {
        let c = (packed[k / 4] >> (6 - 2 * (k % 4))) & 0x03;
        match c {
            0 => -1,
            2 => 1,
            _ => 0,
        }
    };
    let mut s = 0i32;
    let n_blk = n / 128;
    for bl in 0..n_blk {
        let base = bl * 128;
        for b in 0..32 {
            for p in 0..4 {
                s += signed_code(base + b * 4 + p) * acts[base + p * 32 + b] as i32;
            }
        }
    }
    for i in (n_blk * 128)..n {
        s += signed_code(i) * acts[i] as i32;
    }
    s
}

/// Per-token absmax int8 activation quantization: act_scale = max(|x|)/127,
/// q = round(x/act_scale) clamped [-128,127]. Returns (codes, scale, sum).
pub fn quant_i8_sum(input: &[f32]) -> (Vec<i8>, f32, i32) {
    let amax = input.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let mut sum = 0i32;
    let codes: Vec<i8> = input
        .iter()
        .map(|&x| {
            let q = (x / scale).round().clamp(-128.0, 127.0) as i32;
            sum += q;
            q as i8
        })
        .collect();
    (codes, scale, sum)
}

/// One BitLinear layer: y[r] = <row_r, quant(x)> * weight_scale * act_scale.
/// Rows are independent -> parallelized across cores via Rayon.
pub fn bitlinear_forward(
    packed_weights: &[u8],
    weight_scale: f32,
    input: &[f32],
    n_in: usize,
    n_out: usize,
) -> Vec<f32> {
    let (acts, act_scale, act_sum) = quant_i8_sum(input);
    let row_bytes = (n_in + 3) / 4;
    let sc = (weight_scale as f64 * act_scale as f64) as f32;

    let mut out = vec![0f32; n_out];
    crate::pool::dispatch_matmul(packed_weights, row_bytes, &acts, act_sum, sc, n_in, n_out, &mut out);
    out
}

/// Several BitLinear layers that share one input activation (q/k/v, or
/// gate/up), computed via a single fused pool dispatch instead of one per
/// layer -- same math as calling bitlinear_forward N times, activation
/// quantized once. `weights` is (packed_weights, weight_scale, n_out) per
/// layer; results come back in the same order.
pub fn bitlinear_forward_fused(weights: &[(&[u8], f32, usize)], input: &[f32], n_in: usize) -> Vec<Vec<f32>> {
    let (acts, act_scale, act_sum) = quant_i8_sum(input);
    let row_bytes = (n_in + 3) / 4;

    let mut outs: Vec<Vec<f32>> = weights.iter().map(|&(_, _, n_out)| vec![0f32; n_out]).collect();
    let mut segs: Vec<(&[u8], usize, f32, &mut [f32])> = weights
        .iter()
        .zip(outs.iter_mut())
        .map(|(&(packed, wscale, _), out)| {
            let sc = (wscale as f64 * act_scale as f64) as f32;
            (packed, row_bytes, sc, out.as_mut_slice())
        })
        .collect();
    crate::pool::dispatch_matmul_fused(&mut segs, &acts, act_sum, n_in);
    outs
}

/// Several independent activation vectors (e.g. B drafted decode positions)
/// against ONE weight tensor, computed in a single fused pool dispatch --
/// see dispatch_matmul_batched for why this beats B separate
/// bitlinear_forward calls (same total math, ~1/B the DRAM traffic).
pub fn bitlinear_forward_batched(
    packed_weights: &[u8],
    weight_scale: f32,
    inputs: &[Vec<f32>],
    n_in: usize,
    n_out: usize,
) -> Vec<Vec<f32>> {
    let b_count = inputs.len();
    let row_bytes = (n_in + 3) / 4;
    let mut acts_flat = vec![0i8; b_count * n_in];
    let mut act_sums = vec![0i32; b_count];
    let mut scs = vec![0f32; b_count];
    for (b, input) in inputs.iter().enumerate() {
        let (acts, act_scale, act_sum) = quant_i8_sum(input);
        acts_flat[b * n_in..(b + 1) * n_in].copy_from_slice(&acts);
        act_sums[b] = act_sum;
        scs[b] = (weight_scale as f64 * act_scale as f64) as f32;
    }

    let mut out_flat = vec![0f32; b_count * n_out];
    crate::pool::dispatch_matmul_batched(
        packed_weights, row_bytes, &acts_flat, &act_sums, &scs, n_in, n_out, &mut out_flat,
    );
    out_flat.chunks(n_out).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn simd_matches_scalar() {
        let mut rng = rand::thread_rng();
        let lengths = [1, 4, 32, 100, 127, 128, 129, 255, 256, 511, 512, 1000, 2560, 6912];
        for &n in &lengths {
            for _ in 0..200 {
                let nbytes = (n + 3) / 4;
                // real I2_S data only ever has 2-bit codes {0,1,2} (index 3
                // unused) -- build random bytes respecting that, matching
                // what the unsigned-dot-minus-sum(act) correction assumes.
                let packed: Vec<u8> = (0..nbytes)
                    .map(|_| {
                        let mut byte = 0u8;
                        for _ in 0..4 {
                            let code: u8 = rng.gen_range(0..=2);
                            byte = (byte << 2) | code;
                        }
                        byte
                    })
                    .collect();
                let acts: Vec<i8> = (0..n).map(|_| rng.gen_range(-128..=127i8)).collect();
                let asum: i32 = acts.iter().map(|&a| a as i32).sum();
                let simd = unsafe { ternary_dot_row(&packed, &acts, n) } - asum;
                let want = plane_major_ref(&packed, &acts, n);
                assert_eq!(want, simd, "mismatch at n={}", n);
                if n < 128 {
                    assert_eq!(
                        simd,
                        ternary_dot_scalar(&packed, &acts, n),
                        "tail mismatch at n={}",
                        n
                    );
                }
            }
        }
    }
}
