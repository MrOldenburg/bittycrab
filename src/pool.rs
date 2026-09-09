// Persistent spin-wait worker pool, ported from bitlinear.mojo's thread
// pool: threads are spawned once and spin on an atomic generation counter
// instead of parking/condvar-waking (rayon's per-call task injection has
// enough wake latency, ~30-40us/dispatch here, to dominate decode time when
// called 200+ times/token). Two job kinds, dispatched synchronously (the
// caller spin-waits for completion before returning, so raw pointers into
// caller-owned buffers are safe for the duration of one dispatch).

use crate::bitlinear::ternary_dot_row;
use crate::gguf::f16_bits_to_f32;
use std::arch::x86_64::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
use std::sync::OnceLock;

const KIND_MATMUL: u32 = 0;
const KIND_LMHEAD: u32 = 1;
const KIND_TOPK: u32 = 2;
const KIND_MATMUL_FUSED: u32 = 3;
const KIND_MATMUL_BATCHED: u32 = 4;
const KIND_TOPK_BATCHED: u32 = 5;
const KIND_LMHEAD_BATCHED: u32 = 6;
const MAX_FUSED_SEGS: usize = 3;
/// Max positions verified together by prompt-lookup decoding in one round
/// (1 guaranteed-real token + up to MAX_BATCH-1 drafted tokens).
pub const MAX_BATCH: usize = 8;

struct WorkerSlot {
    best_val: AtomicU32, // f32 bits
    best_idx: AtomicUsize,
    // pad to a full cache line so workers don't false-share result slots
    _pad: [u8; 64 - 4 - 8],
}

pub struct Pool {
    n_workers: usize,
    shutdown: AtomicBool,
    generation: AtomicU64,
    completed: AtomicUsize,
    kind: AtomicU32,
    ctr: AtomicUsize,
    chunk: AtomicUsize,

    // matmul job
    wp: AtomicUsize,
    row_bytes: AtomicUsize,
    ap: AtomicUsize,
    act_sum: AtomicI32,
    sc_bits: AtomicU32,
    n_in: AtomicUsize,
    n_out: AtomicUsize,
    op: AtomicUsize,

    // lmhead job
    ep: AtomicUsize,
    hp: AtomicUsize,
    d_model: AtomicUsize,
    vocab: AtomicUsize,
    slots: Vec<WorkerSlot>,

    // fused matmul job: several weight blocks sharing one activation vector
    // (q/k/v, or gate/up), dispatched as a single pool sync instead of one
    // per block -- cuts dispatch count (and its ~13-20us/call fixed
    // overhead) from 7 to ~4 per transformer layer.
    seg_wp: [AtomicUsize; MAX_FUSED_SEGS],
    seg_row_bytes: [AtomicUsize; MAX_FUSED_SEGS],
    seg_sc_bits: [AtomicU32; MAX_FUSED_SEGS],
    seg_nout: [AtomicUsize; MAX_FUSED_SEGS],
    seg_op: [AtomicUsize; MAX_FUSED_SEGS],
    n_segs: AtomicUsize,

    // topk (int8 vnni scan) job -- shares ap/act_sum/n_in/n_out(=vocab) with
    // matmul's fields where the meaning lines up (quantized weight codes go
    // in wp, quantized act codes in ap, act_sum reused, row scales are new).
    row_scales: AtomicUsize, // per-row f32 scale ptr (quantized embedding)
    act_scale_bits: AtomicU32,
    topk_k: AtomicUsize,
    // one top-k result buffer per worker, written only by its owner thread
    // and read by main only after the completion barrier -- safe without
    // further synchronization, hence the raw UnsafeCell instead of a Mutex.
    topk_out: Vec<UnsafeCell<Vec<(f32, i64)>>>,

    // Batched jobs (prompt-lookup decoding): B positions verified in one
    // pool sync instead of B separate dispatches. The weight bytes for a
    // given row are read from DRAM once and reused across the B activation
    // sets while they're still hot in L1 -- same row_bytes fit comfortably
    // (<=1728 bytes for D_FF, <=2560 bytes for the topk embedding scan),
    // which is where the actual bandwidth savings comes from.
    bap: AtomicUsize,                      // flat batched acts, i8, [b*n_in]
    bhp: AtomicUsize,                      // flat batched hidden states, f32, [b*d_model] (KIND_LMHEAD_BATCHED)
    b_count: AtomicUsize,
    b_act_sum: [AtomicI32; MAX_BATCH],
    // KIND_MATMUL_BATCHED: weight_scale*act_scale_b combined.
    // KIND_TOPK_BATCHED: act_scale_b alone (row scale still varies per row).
    b_sc_bits: [AtomicU32; MAX_BATCH],
    bop: AtomicUsize,                      // flat batched output, f32, [b*n_out]
    // per-worker, per-b top-k result buffers for KIND_TOPK_BATCHED.
    topk_bout: Vec<UnsafeCell<Vec<Vec<(f32, i64)>>>>,
}

unsafe impl Sync for Pool {}

impl Pool {
    pub fn global() -> &'static Pool {
        static POOL: OnceLock<&'static Pool> = OnceLock::new();
        *POOL.get_or_init(|| {
            // Leave headroom for the main thread: spin-wait workers are
            // always runnable, so filling every logical core starves the
            // main thread's own serial work (rmsnorm/RoPE/sdpa/etc)
            // between dispatches -- measured ~20x slowdown at full
            // subscription vs one core held back.
            let n_workers = std::env::var("BITTYCRAB_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0)
                // Measured sweep on this machine (2 vs 6/8/12/16/20/24/28/31
                // threads, 2 runs each): throughput is flat at ~45-46 t/s
                // from 6 through 28 threads, with 6-12 the most stable (low
                // run-to-run variance); 2 is compute-starved (~26 t/s) and
                // >=20 gets noisy/occasionally regresses (contention with
                // the main thread and OS). This workload is memory-bandwidth
                // bound past a handful of cores, not compute bound, so more
                // spinning workers past ~8 buys nothing and sometimes hurts.
                .unwrap_or(8);
            let pool: &'static Pool = Box::leak(Box::new(Pool {
                n_workers,
                shutdown: AtomicBool::new(false),
                generation: AtomicU64::new(0),
                completed: AtomicUsize::new(0),
                kind: AtomicU32::new(KIND_MATMUL),
                ctr: AtomicUsize::new(0),
                chunk: AtomicUsize::new(1),
                wp: AtomicUsize::new(0),
                row_bytes: AtomicUsize::new(0),
                ap: AtomicUsize::new(0),
                act_sum: AtomicI32::new(0),
                sc_bits: AtomicU32::new(0),
                n_in: AtomicUsize::new(0),
                n_out: AtomicUsize::new(0),
                op: AtomicUsize::new(0),
                ep: AtomicUsize::new(0),
                hp: AtomicUsize::new(0),
                d_model: AtomicUsize::new(0),
                vocab: AtomicUsize::new(0),
                slots: (0..n_workers)
                    .map(|_| WorkerSlot {
                        best_val: AtomicU32::new(0),
                        best_idx: AtomicUsize::new(0),
                        _pad: [0u8; 64 - 4 - 8],
                    })
                    .collect(),
                seg_wp: [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)],
                seg_row_bytes: [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)],
                seg_sc_bits: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
                seg_nout: [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)],
                seg_op: [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)],
                n_segs: AtomicUsize::new(0),
                row_scales: AtomicUsize::new(0),
                act_scale_bits: AtomicU32::new(0),
                topk_k: AtomicUsize::new(0),
                topk_out: (0..n_workers).map(|_| UnsafeCell::new(Vec::new())).collect(),
                bap: AtomicUsize::new(0),
                bhp: AtomicUsize::new(0),
                b_count: AtomicUsize::new(0),
                b_act_sum: std::array::from_fn(|_| AtomicI32::new(0)),
                b_sc_bits: std::array::from_fn(|_| AtomicU32::new(0)),
                bop: AtomicUsize::new(0),
                topk_bout: (0..n_workers)
                    .map(|_| UnsafeCell::new((0..MAX_BATCH).map(|_| Vec::new()).collect()))
                    .collect(),
            }));
            for id in 0..n_workers {
                std::thread::spawn(move || worker_loop(pool, id));
            }
            pool
        })
    }
}

fn worker_loop(pool: &'static Pool, id: usize) {
    let mut last_gen = 0u64;
    loop {
        loop {
            let g = pool.generation.load(Ordering::Acquire);
            if g != last_gen {
                last_gen = g;
                break;
            }
            if pool.shutdown.load(Ordering::Relaxed) {
                return;
            }
            std::hint::spin_loop();
        }

        match pool.kind.load(Ordering::Relaxed) {
            KIND_MATMUL => unsafe {
                let wp = pool.wp.load(Ordering::Relaxed) as *const u8;
                let row_bytes = pool.row_bytes.load(Ordering::Relaxed);
                let ap = pool.ap.load(Ordering::Relaxed) as *const i8;
                let n_in = pool.n_in.load(Ordering::Relaxed);
                let n_out = pool.n_out.load(Ordering::Relaxed);
                let op = pool.op.load(Ordering::Relaxed) as *mut f32;
                let act_sum = pool.act_sum.load(Ordering::Relaxed);
                let sc = f32::from_bits(pool.sc_bits.load(Ordering::Relaxed));
                let acts = std::slice::from_raw_parts(ap, n_in);
                let chunk = pool.chunk.load(Ordering::Relaxed);
                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= n_out {
                        break;
                    }
                    let end = (start + chunk).min(n_out);
                    for row in start..end {
                        let rb = std::slice::from_raw_parts(wp.add(row * row_bytes), row_bytes);
                        let raw = ternary_dot_row(rb, acts, n_in);
                        *op.add(row) = (raw - act_sum) as f32 * sc;
                    }
                }
            },
            KIND_MATMUL_FUSED => unsafe {
                let n_segs = pool.n_segs.load(Ordering::Relaxed);
                let mut wps = [std::ptr::null::<u8>(); MAX_FUSED_SEGS];
                let mut rbs = [0usize; MAX_FUSED_SEGS];
                let mut scs = [0f32; MAX_FUSED_SEGS];
                let mut nouts = [0usize; MAX_FUSED_SEGS];
                let mut ops = [std::ptr::null_mut::<f32>(); MAX_FUSED_SEGS];
                for i in 0..n_segs {
                    wps[i] = pool.seg_wp[i].load(Ordering::Relaxed) as *const u8;
                    rbs[i] = pool.seg_row_bytes[i].load(Ordering::Relaxed);
                    scs[i] = f32::from_bits(pool.seg_sc_bits[i].load(Ordering::Relaxed));
                    nouts[i] = pool.seg_nout[i].load(Ordering::Relaxed);
                    ops[i] = pool.seg_op[i].load(Ordering::Relaxed) as *mut f32;
                }
                let ap = pool.ap.load(Ordering::Relaxed) as *const i8;
                let n_in = pool.n_in.load(Ordering::Relaxed);
                let act_sum = pool.act_sum.load(Ordering::Relaxed);
                let acts = std::slice::from_raw_parts(ap, n_in);
                let total_out: usize = nouts[..n_segs].iter().sum();
                let chunk = pool.chunk.load(Ordering::Relaxed);
                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= total_out {
                        break;
                    }
                    let end = (start + chunk).min(total_out);
                    for gi in start..end {
                        let mut seg = 0usize;
                        let mut local = gi;
                        while local >= nouts[seg] {
                            local -= nouts[seg];
                            seg += 1;
                        }
                        let row = std::slice::from_raw_parts(wps[seg].add(local * rbs[seg]), rbs[seg]);
                        let raw = ternary_dot_row(row, acts, n_in);
                        *ops[seg].add(local) = (raw - act_sum) as f32 * scs[seg];
                    }
                }
            },
            KIND_MATMUL_BATCHED => unsafe {
                let wp = pool.wp.load(Ordering::Relaxed) as *const u8;
                let row_bytes = pool.row_bytes.load(Ordering::Relaxed);
                let bap = pool.bap.load(Ordering::Relaxed) as *const i8;
                let n_in = pool.n_in.load(Ordering::Relaxed);
                let n_out = pool.n_out.load(Ordering::Relaxed);
                let bop = pool.bop.load(Ordering::Relaxed) as *mut f32;
                let b_count = pool.b_count.load(Ordering::Relaxed);
                let mut act_sum = [0i32; MAX_BATCH];
                let mut sc = [0f32; MAX_BATCH];
                for b in 0..b_count {
                    act_sum[b] = pool.b_act_sum[b].load(Ordering::Relaxed);
                    sc[b] = f32::from_bits(pool.b_sc_bits[b].load(Ordering::Relaxed));
                }
                let chunk = pool.chunk.load(Ordering::Relaxed);
                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= n_out {
                        break;
                    }
                    let end = (start + chunk).min(n_out);
                    for row in start..end {
                        let rb = std::slice::from_raw_parts(wp.add(row * row_bytes), row_bytes);
                        for b in 0..b_count {
                            let acts = std::slice::from_raw_parts(bap.add(b * n_in), n_in);
                            let raw = ternary_dot_row(rb, acts, n_in);
                            *bop.add(b * n_out + row) = (raw - act_sum[b]) as f32 * sc[b];
                        }
                    }
                }
            },
            KIND_TOPK_BATCHED => unsafe {
                let wp = pool.wp.load(Ordering::Relaxed) as *const i8;
                let bap = pool.bap.load(Ordering::Relaxed) as *const i8;
                let sp = pool.row_scales.load(Ordering::Relaxed) as *const f32;
                let d_model = pool.n_in.load(Ordering::Relaxed);
                let vocab = pool.n_out.load(Ordering::Relaxed);
                let k = pool.topk_k.load(Ordering::Relaxed);
                let b_count = pool.b_count.load(Ordering::Relaxed);
                let mut act_sum = [0i32; MAX_BATCH];
                let mut act_scale = [0f32; MAX_BATCH];
                for b in 0..b_count {
                    act_sum[b] = pool.b_act_sum[b].load(Ordering::Relaxed);
                    act_scale[b] = f32::from_bits(pool.b_sc_bits[b].load(Ordering::Relaxed));
                }
                let chunk = pool.chunk.load(Ordering::Relaxed);

                let out = &mut *pool.topk_bout[id].get();
                for b in 0..b_count {
                    out[b].clear();
                }

                let bias = _mm256_set1_epi8(-128i8);
                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= vocab {
                        break;
                    }
                    let end = (start + chunk).min(vocab);
                    for t in start..end {
                        let row = wp.add(t * d_model);
                        let row_scale = *sp.add(t);
                        for b in 0..b_count {
                            let mut acc = _mm256_setzero_si256();
                            let mut c = 0usize;
                            while c < d_model {
                                let wv = _mm256_loadu_si256(row.add(c) as *const __m256i);
                                let wu = _mm256_xor_si256(wv, bias);
                                let av = _mm256_loadu_si256(bap.add(b * d_model + c) as *const __m256i);
                                acc = crate::bitlinear::dpbusd(acc, wu, av);
                                c += 32;
                            }
                            let mut tmp = [0i32; 8];
                            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
                            let raw_u: i32 = tmp.iter().sum();
                            let signed_dot = raw_u - 128 * act_sum[b];
                            let dot = signed_dot as f32 * row_scale * act_scale[b];
                            topk_insert(&mut out[b], k, dot, t as i64);
                        }
                    }
                }
            },
            KIND_LMHEAD_BATCHED => unsafe {
                // Batched F16 lm_head argmax: b_count hidden states verified
                // against the real (unquantized) embedding table in one
                // pool sync -- same amortization as the int8 topk-batched
                // path (each row's F16 bytes read from DRAM once, reused
                // across the batch while hot in L1), but comparing this
                // path's speed against the int8+rescore path is exactly
                // the point of -fp16: does the int8 shortcut actually earn
                // its keep, or would plain batched F16 verify be enough?
                let ep = pool.ep.load(Ordering::Relaxed) as *const u8;
                let bhp = pool.bhp.load(Ordering::Relaxed) as *const f32;
                let d_model = pool.d_model.load(Ordering::Relaxed);
                let vocab = pool.vocab.load(Ordering::Relaxed);
                let b_count = pool.b_count.load(Ordering::Relaxed);
                let row_bytes = d_model * 2;
                let chunk = pool.chunk.load(Ordering::Relaxed);

                let out = &mut *pool.topk_bout[id].get();
                for b in 0..b_count {
                    out[b].clear();
                }

                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= vocab {
                        break;
                    }
                    let end = (start + chunk).min(vocab);
                    for t in start..end {
                        let row = std::slice::from_raw_parts(ep.add(t * row_bytes), row_bytes);
                        for b in 0..b_count {
                            let h = std::slice::from_raw_parts(bhp.add(b * d_model), d_model);
                            let dot = crate::model::lm_dot_row_f16c(row, h);
                            topk_insert(&mut out[b], 1, dot, t as i64);
                        }
                    }
                }
            },
            KIND_LMHEAD => unsafe {
                let ep = pool.ep.load(Ordering::Relaxed) as *const u8;
                let hp = pool.hp.load(Ordering::Relaxed) as *const f32;
                let d_model = pool.d_model.load(Ordering::Relaxed);
                let vocab = pool.vocab.load(Ordering::Relaxed);
                let h = std::slice::from_raw_parts(hp, d_model);
                let row_bytes = d_model * 2;
                let chunk = pool.chunk.load(Ordering::Relaxed);
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= vocab {
                        break;
                    }
                    let end = (start + chunk).min(vocab);
                    for t in start..end {
                        let row = std::slice::from_raw_parts(ep.add(t * row_bytes), row_bytes);
                        let dot = crate::model::lm_dot_row_f16c(row, h);
                        if dot > best_val {
                            best_val = dot;
                            best_idx = t;
                        }
                    }
                }
                pool.slots[id].best_val.store(best_val.to_bits(), Ordering::Relaxed);
                pool.slots[id].best_idx.store(best_idx, Ordering::Relaxed);
            },
            _ => unsafe {
                // KIND_TOPK: int8 x int8 VNNI dot per row (weight side biased
                // +128 via XOR 0x80 so vpdpbusd's unsigned-operand
                // requirement is satisfiable), maintaining this worker's own
                // local top-k via insertion sort -- see topk_insert below.
                let wp = pool.wp.load(Ordering::Relaxed) as *const i8;
                let ap = pool.ap.load(Ordering::Relaxed) as *const i8;
                let sp = pool.row_scales.load(Ordering::Relaxed) as *const f32;
                let d_model = pool.n_in.load(Ordering::Relaxed);
                let vocab = pool.n_out.load(Ordering::Relaxed);
                let act_sum = pool.act_sum.load(Ordering::Relaxed);
                let act_scale = f32::from_bits(pool.act_scale_bits.load(Ordering::Relaxed));
                let k = pool.topk_k.load(Ordering::Relaxed);
                let chunk = pool.chunk.load(Ordering::Relaxed);

                let acts = std::slice::from_raw_parts(ap, d_model);
                let out = &mut *pool.topk_out[id].get();
                out.clear();

                let bias = _mm256_set1_epi8(-128i8); // 0x80: XOR flips sign bit == +128 mod 256
                loop {
                    let start = pool.ctr.fetch_add(chunk, Ordering::Relaxed);
                    if start >= vocab {
                        break;
                    }
                    let end = (start + chunk).min(vocab);
                    for t in start..end {
                        let row = wp.add(t * d_model);
                        let mut acc = _mm256_setzero_si256();
                        let mut c = 0usize;
                        while c < d_model {
                            let wv = _mm256_loadu_si256(row.add(c) as *const __m256i);
                            let wu = _mm256_xor_si256(wv, bias);
                            let av = _mm256_loadu_si256(acts.as_ptr().add(c) as *const __m256i);
                            acc = crate::bitlinear::dpbusd(acc, wu, av);
                            c += 32;
                        }
                        let mut tmp = [0i32; 8];
                        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
                        let raw_u: i32 = tmp.iter().sum();
                        let signed_dot = raw_u - 128 * act_sum;
                        let dot = signed_dot as f32 * *sp.add(t) * act_scale;

                        topk_insert(out, k, dot, t as i64);
                    }
                }
            },
        }

        pool.completed.fetch_add(1, Ordering::Release);
    }
}

/// Sorted-descending top-k insertion, capped at k entries. Mirrors
/// bitlinear.mojo's _topk_insert.
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

#[inline]
fn wait_for_completion(pool: &Pool) {
    while pool.completed.load(Ordering::Acquire) < pool.n_workers {
        std::hint::spin_loop();
    }
}

/// Row-parallel BitLinear matmul dispatch: out[r] = (ternary_dot(row_r, acts)
/// - act_sum) * sc, for r in [0, n_out).
pub fn dispatch_matmul(
    packed_weights: &[u8],
    row_bytes: usize,
    acts: &[i8],
    act_sum: i32,
    sc: f32,
    n_in: usize,
    n_out: usize,
    out: &mut [f32],
) {
    let pool = Pool::global();
    let chunk = (n_out / (4 * pool.n_workers)).max(1);
    pool.wp.store(packed_weights.as_ptr() as usize, Ordering::Relaxed);
    pool.row_bytes.store(row_bytes, Ordering::Relaxed);
    pool.ap.store(acts.as_ptr() as usize, Ordering::Relaxed);
    pool.act_sum.store(act_sum, Ordering::Relaxed);
    pool.sc_bits.store(sc.to_bits(), Ordering::Relaxed);
    pool.n_in.store(n_in, Ordering::Relaxed);
    pool.n_out.store(n_out, Ordering::Relaxed);
    pool.op.store(out.as_mut_ptr() as usize, Ordering::Relaxed);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_MATMUL, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);
}

/// Fused row-parallel BitLinear matmul dispatch: several weight blocks
/// (up to MAX_FUSED_SEGS) sharing one activation vector, computed in a
/// single pool sync instead of one dispatch per block. Each tuple is
/// (packed_weights, row_bytes, sc, out).
pub fn dispatch_matmul_fused(segs: &mut [(&[u8], usize, f32, &mut [f32])], acts: &[i8], act_sum: i32, n_in: usize) {
    let pool = Pool::global();
    let n_segs = segs.len();
    assert!(n_segs <= MAX_FUSED_SEGS);
    let mut total_out = 0usize;
    for (i, (packed, row_bytes, sc, out)) in segs.iter_mut().enumerate() {
        pool.seg_wp[i].store(packed.as_ptr() as usize, Ordering::Relaxed);
        pool.seg_row_bytes[i].store(*row_bytes, Ordering::Relaxed);
        pool.seg_sc_bits[i].store(sc.to_bits(), Ordering::Relaxed);
        pool.seg_nout[i].store(out.len(), Ordering::Relaxed);
        pool.seg_op[i].store(out.as_mut_ptr() as usize, Ordering::Relaxed);
        total_out += out.len();
    }
    pool.n_segs.store(n_segs, Ordering::Relaxed);
    pool.ap.store(acts.as_ptr() as usize, Ordering::Relaxed);
    pool.act_sum.store(act_sum, Ordering::Relaxed);
    pool.n_in.store(n_in, Ordering::Relaxed);
    let chunk = (total_out / (4 * pool.n_workers)).max(1);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_MATMUL_FUSED, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);
}

/// Batched row-parallel BitLinear matmul: b_count activation sets against
/// one weight tensor in a single pool sync -- b_count separate
/// bitlinear_forward calls would each re-read all n_out*row_bytes weight
/// bytes from DRAM; this reads them once per row and reuses the (small,
/// L1-resident) row across the batch. `acts_flat` is [b_count][n_in]
/// contiguous, `act_sums`/`scs` are per-b, `out_flat` is [b_count][n_out].
pub fn dispatch_matmul_batched(
    packed_weights: &[u8],
    row_bytes: usize,
    acts_flat: &[i8],
    act_sums: &[i32],
    scs: &[f32],
    n_in: usize,
    n_out: usize,
    out_flat: &mut [f32],
) {
    let pool = Pool::global();
    let b_count = act_sums.len();
    assert!(b_count <= MAX_BATCH);
    pool.wp.store(packed_weights.as_ptr() as usize, Ordering::Relaxed);
    pool.row_bytes.store(row_bytes, Ordering::Relaxed);
    pool.bap.store(acts_flat.as_ptr() as usize, Ordering::Relaxed);
    pool.n_in.store(n_in, Ordering::Relaxed);
    pool.n_out.store(n_out, Ordering::Relaxed);
    pool.bop.store(out_flat.as_mut_ptr() as usize, Ordering::Relaxed);
    pool.b_count.store(b_count, Ordering::Relaxed);
    for b in 0..b_count {
        pool.b_act_sum[b].store(act_sums[b], Ordering::Relaxed);
        pool.b_sc_bits[b].store(scs[b].to_bits(), Ordering::Relaxed);
    }
    let chunk = (n_out / (4 * pool.n_workers)).max(1);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_MATMUL_BATCHED, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);
}

/// Batched int8-VNNI top-k scan: b_count hidden states against the
/// quantized embedding table in one pool sync. Returns one candidate-id
/// list per b (each worker's local top-k concatenated, same as
/// dispatch_topk_scan).
pub fn dispatch_topk_scan_batched(
    quant_codes: &[i8],
    row_scales: &[f32],
    acts_flat: &[i8],
    act_sums: &[i32],
    act_scales: &[f32],
    d_model: usize,
    vocab: usize,
    k: usize,
) -> Vec<Vec<i64>> {
    let pool = Pool::global();
    let b_count = act_sums.len();
    assert!(b_count <= MAX_BATCH);
    pool.wp.store(quant_codes.as_ptr() as usize, Ordering::Relaxed);
    pool.bap.store(acts_flat.as_ptr() as usize, Ordering::Relaxed);
    pool.row_scales.store(row_scales.as_ptr() as usize, Ordering::Relaxed);
    pool.n_in.store(d_model, Ordering::Relaxed);
    pool.n_out.store(vocab, Ordering::Relaxed);
    pool.topk_k.store(k, Ordering::Relaxed);
    pool.b_count.store(b_count, Ordering::Relaxed);
    for b in 0..b_count {
        pool.b_act_sum[b].store(act_sums[b], Ordering::Relaxed);
        pool.b_sc_bits[b].store(act_scales[b].to_bits(), Ordering::Relaxed);
    }
    let chunk = (vocab / (4 * pool.n_workers)).max(1);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_TOPK_BATCHED, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);

    let mut out: Vec<Vec<i64>> = (0..b_count).map(|_| Vec::with_capacity(pool.n_workers * k)).collect();
    for slot in &pool.topk_bout {
        let v = unsafe { &*slot.get() };
        for b in 0..b_count {
            out[b].extend(v[b].iter().map(|&(_, id)| id));
        }
    }
    out
}

/// Vocab-parallel lm_head argmax dispatch: returns argmax_t(row_t(embd) . h).
pub fn dispatch_lmhead(embd_raw: &[u8], h_normed: &[f32], d_model: usize, vocab: usize) -> i64 {
    let pool = Pool::global();
    let chunk = (vocab / (4 * pool.n_workers)).max(1);
    pool.ep.store(embd_raw.as_ptr() as usize, Ordering::Relaxed);
    pool.hp.store(h_normed.as_ptr() as usize, Ordering::Relaxed);
    pool.d_model.store(d_model, Ordering::Relaxed);
    pool.vocab.store(vocab, Ordering::Relaxed);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_LMHEAD, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);

    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0usize;
    for s in &pool.slots {
        let v = f32::from_bits(s.best_val.load(Ordering::Relaxed));
        if v > best_val {
            best_val = v;
            best_idx = s.best_idx.load(Ordering::Relaxed);
        }
    }
    let _ = f16_bits_to_f32; // silence unused-import if lmhead path inlined differently
    best_idx as i64
}

/// Batched F16 lm_head argmax: b_count hidden states verified against the
/// real embedding table (no int8 approximation at all) in one pool sync.
/// Returns one argmax id per b, exact.
pub fn dispatch_lmhead_batched(embd_raw: &[u8], h_normed_flat: &[f32], d_model: usize, vocab: usize, b_count: usize) -> Vec<i64> {
    let pool = Pool::global();
    assert!(b_count <= MAX_BATCH);
    let chunk = (vocab / (4 * pool.n_workers)).max(1);
    pool.ep.store(embd_raw.as_ptr() as usize, Ordering::Relaxed);
    pool.bhp.store(h_normed_flat.as_ptr() as usize, Ordering::Relaxed);
    pool.d_model.store(d_model, Ordering::Relaxed);
    pool.vocab.store(vocab, Ordering::Relaxed);
    pool.b_count.store(b_count, Ordering::Relaxed);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_LMHEAD_BATCHED, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);

    (0..b_count)
        .map(|b| {
            let mut best_val = f32::NEG_INFINITY;
            let mut best_id = 0i64;
            for slot in &pool.topk_bout {
                let v = unsafe { &*slot.get() };
                if let Some(&(val, id)) = v[b].first() {
                    if val > best_val {
                        best_val = val;
                        best_id = id;
                    }
                }
            }
            best_id
        })
        .collect()
}

/// Vocab-parallel int8xint8 (AVX-VNNI) top-k scan against the quantized
/// embedding table: returns each worker's local top-k candidate ids
/// concatenated (up to n_workers*k, some workers may return fewer near the
/// tail). d_model must be a multiple of 32 (true for D_MODEL=2560).
pub fn dispatch_topk_scan(
    quant_codes: &[i8],
    row_scales: &[f32],
    acts: &[i8],
    act_sum: i32,
    act_scale: f32,
    d_model: usize,
    vocab: usize,
    k: usize,
) -> Vec<i64> {
    let pool = Pool::global();
    let chunk = (vocab / (4 * pool.n_workers)).max(1);
    pool.wp.store(quant_codes.as_ptr() as usize, Ordering::Relaxed);
    pool.ap.store(acts.as_ptr() as usize, Ordering::Relaxed);
    pool.row_scales.store(row_scales.as_ptr() as usize, Ordering::Relaxed);
    pool.act_sum.store(act_sum, Ordering::Relaxed);
    pool.act_scale_bits.store(act_scale.to_bits(), Ordering::Relaxed);
    pool.n_in.store(d_model, Ordering::Relaxed);
    pool.n_out.store(vocab, Ordering::Relaxed);
    pool.topk_k.store(k, Ordering::Relaxed);
    pool.ctr.store(0, Ordering::Relaxed);
    pool.chunk.store(chunk, Ordering::Relaxed);
    pool.completed.store(0, Ordering::Relaxed);
    pool.kind.store(KIND_TOPK, Ordering::Relaxed);
    pool.generation.fetch_add(1, Ordering::Release);
    wait_for_completion(pool);

    let mut ids = Vec::with_capacity(pool.n_workers * k);
    for slot in &pool.topk_out {
        let v = unsafe { &*slot.get() };
        ids.extend(v.iter().map(|&(_, id)| id));
    }
    ids
}
