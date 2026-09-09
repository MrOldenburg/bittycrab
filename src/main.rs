use bittycrab::embd::quantize_embd;
use bittycrab::model::{bench_decode, bench_decode_topk, generate, generate_topk, load_model, profile_decode, VOCAB};
use bittycrab::tokenizer::{decode, encode, load_tokenizer};
use mimalloc::MiMalloc;

// Decode is allocation-heavy (each layer's rmsnorm/bitlinear/silu/residual
// allocates a fresh Vec<f32>, ~15-18 allocations/layer x 30 layers/token) --
// mimalloc's thread-local size-class caching handles that churn better than
// glibc's default allocator.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// mimalloc's own C API for forcing freed pages back to the OS instead of
// keeping them cached for reuse -- not exposed by the `mimalloc` crate
// wrapper, but the symbol is linked in since it's mimalloc's own runtime.
extern "C" {
    fn mi_collect(force: bool);
}

fn print_rss(label: &str) {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") || line.starts_with("VmHWM:") {
                println!("[mem @ {}] {}", label, line.trim());
            }
        }
    }
}

const PROMPTS: [&str; 15] = [
    "The capital of France is",
    "The capital of Japan is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "def fibonacci(n):",
    "import numpy as np\n",
    "The quick brown fox",
    "Once upon a time, there was a",
    "2 + 2 =",
    "The president of the United States lives in",
    "My favorite hobby is",
    "The chemical symbol for gold is",
    "In machine learning, a neural network",
    "She walked into the room and",
    "The speed of light is approximately",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mem_mode = args.iter().any(|a| a == "-mem" || a == "--mem");
    let fp16_mode = args.iter().any(|a| a == "-fp16" || a == "--fp16");
    let pq_mode = args.iter().any(|a| a == "-pq" || a == "--pq");
    let opq_mode = args.iter().any(|a| a == "-opq" || a == "--opq");
    let quant_only = args.iter().position(|a| a == "-quant-only").map(|p| args[p + 1].clone());
    if let Some(method) = quant_only {
        run_quant_only(&method);
    } else if mem_mode {
        run_mem_mode();
    } else if fp16_mode {
        run_fp16_mode();
    } else if opq_mode {
        run_opq_mode();
    } else if pq_mode {
        run_pq_mode();
    } else {
        run_default();
    }
}

/// -opq: same as -pq, but the scan table is Optimized Product Quantization
/// (see opq.rs) instead of plain PQ -- a learned rotation is applied before
/// splitting into subspaces, specifically to fix the failure mode plain PQ
/// hit (the true best token landing hundreds of ranks out even after heavy
/// overtraining): arbitrary contiguous-chunk subspaces don't align with the
/// embedding's real variance structure, so each subspace's k-means was
/// stuck with a needlessly hard clustering problem. Same pread-only
/// construction and runtime access as -pq (only the OPQ table, now
/// including the D_MODEL x D_MODEL rotation matrix, is meant to stay
/// resident), same disk cache (the rotation-learning step needs a
/// D_MODEL x D_MODEL SVD per outer iteration, which is the truly expensive
/// part -- not something to redo on every run).
fn run_opq_mode() {
    use bittycrab::embd::EmbdFileReader;
    use bittycrab::opq::{build_opq, load_opq, save_opq};
    use bittycrab::pld::generate_pld_opq;

    println!("=== -opq mode: rotation-learned (Optimized) product-quantized embedding scan ===");
    print_rss("start");
    let path = "/home/soldenb/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
    let m = load_model(path);
    print_rss("after load_model");
    let tok = load_tokenizer(&m.g);

    let pq_m: usize = std::env::var("OPQ_M").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let pq_k: usize = std::env::var("OPQ_CENTROIDS").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let train_samples: usize = std::env::var("OPQ_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let kmeans_iters: usize = std::env::var("OPQ_KMEANS_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let outer_iters: usize = std::env::var("OPQ_OUTER_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    println!(
        "OPQ config: m={} k={} train_samples={} kmeans_iters={} outer_iters={}",
        pq_m, pq_k, train_samples, kmeans_iters, outer_iters
    );

    let cache_path = format!(
        "/home/soldenb/Projects/bittycrab/opq_cache_m{}_k{}_s{}_ki{}_oi{}.bin",
        pq_m, pq_k, train_samples, kmeans_iters, outer_iters
    );
    let t0 = std::time::Instant::now();
    let opq = if let Ok(cached) = load_opq(&cache_path) {
        println!("loaded cached OPQ table from {}: {:.2}s", cache_path, t0.elapsed().as_secs_f64());
        cached
    } else {
        let built = build_opq(&m.g, path, VOCAB, pq_m, pq_k, train_samples, kmeans_iters, outer_iters);
        println!("build_opq: {:.2}s (no cache found at {})", t0.elapsed().as_secs_f64(), cache_path);
        if let Err(e) = save_opq(&built, &cache_path) {
            println!("warning: failed to save OPQ cache to {}: {}", cache_path, e);
        } else {
            println!("saved OPQ cache to {}", cache_path);
        }
        built
    };
    println!(
        "OPQ table: rotation {} bytes, codes {} bytes, codebooks {} bytes, norms {} bytes, total {:.2} MB",
        opq.rotation.len() * 4,
        opq.codes.len(),
        opq.codebooks.len() * 4,
        opq.norms.len() * 4,
        (opq.rotation.len() * 4 + opq.codes.len() + opq.codebooks.len() * 4 + opq.norms.len() * 4) as f64 / 1e6
    );
    print_rss("after build/load_opq (pread-only -- should NOT show the full F16 table resident)");

    let reader = EmbdFileReader::open(path, &m.g);
    let k: usize = std::env::var("OPQ_K").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let ngrams = [4usize, 3, 2];
    let max_draft = 7;
    let max_new = 20;

    println!("--- 15-prompt harness (-opq): PLD (OPQ scan) timing, k={} (no mmap touched yet) ---", k);
    let mut pld_outputs = Vec::with_capacity(PROMPTS.len());
    let mut total_tokens = 0usize;
    let mut total_secs = 0f64;
    for (i, &pt) in PROMPTS.iter().enumerate() {
        let ids = encode(&tok, pt, true);
        let t0 = std::time::Instant::now();
        let pld = generate_pld_opq(&m, &reader, &opq, &ids, max_new, k, &ngrams, max_draft);
        let elapsed = t0.elapsed().as_secs_f64();
        total_tokens += pld.len();
        total_secs += elapsed;
        println!("[{:2}] {:<45} {:.2} tok/s ({} tok in {:.3}s)", i, format!("{:?}", pt), pld.len() as f64 / elapsed, pld.len(), elapsed);
        pld_outputs.push((ids, pld));
    }
    println!(
        "--- -opq speed: overall {:.2} tok/s ({} tokens / {:.3}s) ---",
        total_tokens as f64 / total_secs, total_tokens, total_secs
    );
    print_rss("after -opq generation (true -opq-only footprint, before any mmap touch)");

    println!("--- validating against exact greedy (this step touches the mmap) ---");
    let mut all_exact = true;
    for (i, (ids, pld)) in pld_outputs.iter().enumerate() {
        let exact = generate(&m, ids, max_new);
        let matches = pld == &exact;
        all_exact &= matches;
        println!("[{:2}] exact={}", i, matches);
        if !matches {
            println!("      exact: {:?}", exact);
            println!("      pld  : {:?}", pld);
        }
    }
    println!("--- -opq accuracy: all_exact={} ---", all_exact);
    print_rss("end of -opq run (after validation touched the mmap)");
}

/// -pq: same PLD algorithm as the default int8 path, but the lm_head scan
/// uses Product Quantization instead of int8. Built entirely through
/// explicit positioned file reads (never mmap) for both construction and
/// every embedding-table access at runtime (input embedding + exact
/// rescore both go through `reader`), so nothing but the PQ table itself
/// (codes + codebooks + norms, a few MB) is ever meant to stay resident --
/// no permanent F16 residency, and no int8 table is ever built in this mode.
fn run_pq_mode() {
    use bittycrab::embd::EmbdFileReader;
    use bittycrab::pld::generate_pld_pq;
    use bittycrab::pq::{build_pq, load_pq, save_pq};

    println!("=== -pq mode: product-quantized embedding scan, PQ table only in memory ===");
    print_rss("start");
    let path = "/home/soldenb/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
    let m = load_model(path);
    print_rss("after load_model");
    let tok = load_tokenizer(&m.g);

    // Refined from the first pass (M=40, K=256, 20k samples, 10 iters),
    // which left the true best token ranked ~800th under PQ scoring --
    // finer subspaces (larger M) and more/better-trained centroids (more
    // samples, more iterations) should push that rank down toward
    // something a practical k can actually catch.
    let pq_m: usize = std::env::var("PQ_M").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let pq_k: usize = std::env::var("PQ_CENTROIDS").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let train_samples: usize = std::env::var("PQ_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let iters: usize = std::env::var("PQ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    println!("PQ config: m={} k={} train_samples={} iters={}", pq_m, pq_k, train_samples, iters);

    // A k-means build this size (M=128, 300 iters, full-vocab training) took
    // ~7 minutes -- pay that once, cache the ~20MB result to disk, and load
    // it directly on every subsequent run instead of retraining from
    // scratch each time. Cache filename encodes the hyperparameters so
    // different configs don't collide or silently load a stale table.
    let cache_path = format!(
        "/home/soldenb/Projects/bittycrab/pq_cache_m{}_k{}_s{}_i{}.bin",
        pq_m, pq_k, train_samples, iters
    );
    let t0 = std::time::Instant::now();
    let pq = if let Ok(cached) = load_pq(&cache_path) {
        println!("loaded cached PQ table from {}: {:.2}s", cache_path, t0.elapsed().as_secs_f64());
        cached
    } else {
        let built = build_pq(&m.g, path, VOCAB, pq_m, pq_k, train_samples, iters);
        println!("build_pq: {:.2}s (no cache found at {})", t0.elapsed().as_secs_f64(), cache_path);
        if let Err(e) = save_pq(&built, &cache_path) {
            println!("warning: failed to save PQ cache to {}: {}", cache_path, e);
        } else {
            println!("saved PQ cache to {}", cache_path);
        }
        built
    };
    println!(
        "PQ table: codes {} bytes, codebooks {} bytes, norms {} bytes, total {:.2} MB",
        pq.codes.len(),
        pq.codebooks.len() * 4,
        pq.norms.len() * 4,
        (pq.codes.len() + pq.codebooks.len() * 4 + pq.norms.len() * 4) as f64 / 1e6
    );
    print_rss("after build_pq (pread-only -- should NOT show the full F16 table resident)");

    let reader = EmbdFileReader::open(path, &m.g);

    let k: usize = std::env::var("PQ_K").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let ngrams = [4usize, 3, 2];
    let max_draft = 7;
    let max_new = 20;

    // Speed pass first, no mmap touched at all (generate_pld_pq only reads
    // the embedding table through `reader`) -- this is the honest -pq
    // footprint, same discipline as -mem.
    println!("--- 15-prompt harness (-pq): PLD (PQ scan) timing, k={} (no mmap touched yet) ---", k);
    let mut pld_outputs = Vec::with_capacity(PROMPTS.len());
    let mut total_tokens = 0usize;
    let mut total_secs = 0f64;
    for (i, &pt) in PROMPTS.iter().enumerate() {
        let ids = encode(&tok, pt, true);
        let t0 = std::time::Instant::now();
        let pld = generate_pld_pq(&m, &reader, &pq, &ids, max_new, k, &ngrams, max_draft);
        let elapsed = t0.elapsed().as_secs_f64();
        total_tokens += pld.len();
        total_secs += elapsed;
        println!("[{:2}] {:<45} {:.2} tok/s ({} tok in {:.3}s)", i, format!("{:?}", pt), pld.len() as f64 / elapsed, pld.len(), elapsed);
        pld_outputs.push((ids, pld));
    }
    println!(
        "--- -pq speed: overall {:.2} tok/s ({} tokens / {:.3}s) ---",
        total_tokens as f64 / total_secs, total_tokens, total_secs
    );
    print_rss("after -pq generation (true -pq-only footprint, before any mmap touch)");

    // Validate separately -- this uses the mmap-based exact path and WILL
    // fault the F16 table in from here on, so it's kept clearly apart from
    // the footprint measurement above.
    println!("--- validating against exact greedy (this step touches the mmap) ---");
    let mut all_exact = true;
    for (i, (ids, pld)) in pld_outputs.iter().enumerate() {
        let exact = generate(&m, ids, max_new);
        let matches = pld == &exact;
        all_exact &= matches;
        println!("[{:2}] exact={}", i, matches);
        if !matches {
            println!("      exact: {:?}", exact);
            println!("      pld  : {:?}", pld);
        }
    }
    println!("--- -pq accuracy: all_exact={} ---", all_exact);
    print_rss("end of -pq run (after validation touched the mmap)");
}

/// Isolated, single-method peak-RSS measurement: does ONLY load_model +
/// one quantization strategy, then exits immediately, so VmHWM reflects
/// that one method alone -- not contaminated by whatever ran before it in
/// the same process (mimalloc doesn't return freed pages to the OS by
/// default, so running three strategies back-to-back in one process makes
/// each one's peak look like the running maximum of all of them so far).
fn run_quant_only(method: &str) {
    use bittycrab::embd::{quantize_embd, quantize_embd_streaming, quantize_embd_streaming_windowed};
    let path = "/home/soldenb/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
    let m = load_model(path);
    print_rss("after load_model");
    let t0 = std::time::Instant::now();
    let qe = match method {
        "mmap" => quantize_embd(&m.g, VOCAB),
        "bigshard" => quantize_embd_streaming(path, &m.g, VOCAB),
        "windowed" => quantize_embd_streaming_windowed(path, &m.g, VOCAB, 4096),
        other => panic!("unknown -quant-only method: {}", other),
    };
    println!("{}: {:.3}s, {} codes", method, t0.elapsed().as_secs_f64(), qe.codes.len());
    print_rss(&format!("after {} (peak)", method));
}

/// -fp16: same PLD algorithm, same batched layers/priming/bonus-reuse as
/// the default int8 path, but the lm_head verify step reads the real F16
/// embedding table directly (batched, no int8 quantization or rescore at
/// all). This answers a question the rest of this session never actually
/// tested head-to-head: does the int8+rescore shortcut earn its keep once
/// PLD is already batching everything else, or would plain batched F16
/// verification have been just as fast?
fn run_fp16_mode() {
    use bittycrab::model::EOS_ID;
    use bittycrab::pld::generate_pld_fp16;

    println!("=== -fp16 mode: PLD with the real F16 lm_head, no int8 at all ===");
    print_rss("start");
    let path = "/home/soldenb/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
    let m = load_model(path);
    print_rss("after load_model");
    let tok = load_tokenizer(&m.g);

    let ngrams = [4usize, 3, 2];
    let max_draft = 7;
    let max_new = 20;

    println!("--- 15-prompt harness (-fp16): PLD (f16 verify) vs exact greedy ---");
    let mut all_exact = true;
    let mut total_tokens = 0usize;
    let mut total_secs = 0f64;
    for (i, &pt) in PROMPTS.iter().enumerate() {
        let ids = encode(&tok, pt, true);
        let exact = generate(&m, &ids, max_new);

        let t0 = std::time::Instant::now();
        let pld = generate_pld_fp16(&m, &ids, max_new, &ngrams, max_draft);
        let elapsed = t0.elapsed().as_secs_f64();

        let matches = pld == exact;
        all_exact &= matches;
        total_tokens += pld.len();
        total_secs += elapsed;

        let stopped_early = exact.last() == Some(&EOS_ID);
        println!(
            "[{:2}] {:<45} exact={:<5} {:.2} tok/s ({} tok in {:.3}s){}",
            i,
            format!("{:?}", pt),
            matches,
            pld.len() as f64 / elapsed,
            pld.len(),
            elapsed,
            if stopped_early { "  [EOS]" } else { "" },
        );
    }
    println!(
        "--- -fp16 harness summary: all_exact={}  overall {:.2} tok/s ({} tokens / {:.3}s) ---",
        all_exact,
        total_tokens as f64 / total_secs,
        total_tokens,
        total_secs
    );
    print_rss("end of -fp16 run");
}

/// -mem: same PLD algorithm, same layer weights (already zero-copy off the
/// mmap regardless), but the embedding table -- quantization source read,
/// input-token embedding, and exact rescore -- goes through explicit
/// positioned file reads instead of the mmap, so the ~656MB F16 table never
/// becomes permanently resident. Trade: some per-row syscall overhead.
fn run_mem_mode() {
    use bittycrab::embd::{quantize_embd_streaming_windowed, EmbdFileReader};
    use bittycrab::pld::generate_pld_mem;

    println!("=== -mem mode: no mmap residency for the F16 embedding table ===");
    print_rss("start");
    let path = "/home/soldenb/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
    let m = load_model(path);
    print_rss("after load_model");
    let tok = load_tokenizer(&m.g);
    print_rss("after load_tokenizer (128256 tokens + BPE merge table)");

    // Windowed (small-wave) quantization: measured in isolation, this cuts
    // peak RSS during startup by ~55-57% vs either mmap or the old
    // one-big-read-per-thread streamer (452MB vs ~1013-1030MB), for ~30-50ms
    // of extra startup time -- a clear win for -mem's whole point (keeping
    // the process's footprint small), so it's the real default here now,
    // not just a diagnostic comparison.
    let t0 = std::time::Instant::now();
    let qe = quantize_embd_streaming_windowed(path, &m.g, VOCAB, 4096);
    println!("quantize_embd_streaming_windowed: {:.2}s", t0.elapsed().as_secs_f64());
    print_rss("after quantize_embd_streaming_windowed (peak stays low throughout -- no big transient buffer to reclaim)");
    unsafe { mi_collect(true) };
    print_rss("after mi_collect(true) (should be a much smaller drop than before, since there's little to reclaim now)");

    let reader = EmbdFileReader::open(path, &m.g);

    let k = 4;
    let ngrams = [4usize, 3, 2];
    let max_draft = 7;
    let max_new = 20;

    // Speed pass first, with NO mmap-touching calls at all -- generate_pld_mem
    // only ever reads the embedding table through `reader` (pread). This is
    // the honest -mem footprint: the exact-reference comparison below uses
    // the plain mmap-based `generate()` for convenience, which would itself
    // fault the whole 656MB table in and defeat the point of this
    // measurement if it ran first -- so it's deliberately run AFTER, on its
    // own, clearly separated.
    println!("--- 15-prompt harness (-mem): PLD k={} timing (no mmap touched yet) ---", k);
    let mut pld_outputs = Vec::with_capacity(PROMPTS.len());
    let mut total_tokens = 0usize;
    let mut total_secs = 0f64;
    for (i, &pt) in PROMPTS.iter().enumerate() {
        let ids = encode(&tok, pt, true);
        let t0 = std::time::Instant::now();
        let pld = generate_pld_mem(&m, &reader, &qe, &ids, max_new, k, &ngrams, max_draft);
        let elapsed = t0.elapsed().as_secs_f64();
        total_tokens += pld.len();
        total_secs += elapsed;
        println!("[{:2}] {:<45} {:.2} tok/s ({} tok in {:.3}s)", i, format!("{:?}", pt), pld.len() as f64 / elapsed, pld.len(), elapsed);
        pld_outputs.push((ids, pld));
    }
    println!(
        "--- -mem speed: overall {:.2} tok/s ({} tokens / {:.3}s) ---",
        total_tokens as f64 / total_secs,
        total_tokens,
        total_secs
    );
    print_rss("after -mem generation (true -mem-only footprint, before any mmap touch)");

    // Now validate correctness -- this deliberately uses the mmap-based
    // exact path, which WILL fault the F16 table in from here on. Kept
    // separate so its RSS impact doesn't get blamed on -mem itself.
    println!("--- validating against exact greedy (this step touches the mmap) ---");
    let mut all_exact = true;
    for (i, (ids, pld)) in pld_outputs.iter().enumerate() {
        let exact = generate(&m, ids, max_new);
        let matches = pld == &exact;
        all_exact &= matches;
        println!("[{:2}] exact={}", i, matches);
        if !matches {
            println!("      exact: {:?}", exact);
            println!("      pld  : {:?}", pld);
        }
    }
    println!("--- -mem accuracy: all_exact={} ---", all_exact);
    print_rss("end of -mem run (after validation touched the mmap)");
}

fn run_default() {
    print_rss("start");
    let path = "/home/soldenb/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";
    let m = load_model(path);
    print_rss("after load_model (30 I2_S layers + norms; embedding is mmap, not counted here)");
    let tok = load_tokenizer(&m.g);

    let prompt_text = "The capital of France is";
    let prompt = encode(&tok, prompt_text, true);
    println!("prompt: {:?}", prompt_text);
    println!("prompt ids: {:?}", prompt);

    let gen = generate(&m, &prompt, 20);
    println!("generated ids: {:?}", gen);
    println!("generated text: {:?}", decode(&tok, &gen, true));
    println!("full: {:?}", format!("{}{}", prompt_text, decode(&tok, &gen, true)));

    let tps = bench_decode(&m, &prompt);
    println!("Steady-state decode (F16 lm_head): {:.2} tokens/sec", tps);

    let t0 = std::time::Instant::now();
    let qe = quantize_embd(&m.g, VOCAB);
    println!("quantize_embd: {:.2}s ({} bytes codes)", t0.elapsed().as_secs_f64(), qe.codes.len());
    print_rss("after quantize_embd (adds 328MB int8 codes + 513KB scales)");

    println!("--- k sweep (per-worker top-k candidates, {} workers) ---", 8);
    for &kk in &[1usize, 2, 4, 8, 16] {
        let gen_tk = generate_topk(&m, &qe, &prompt, 20, kk);
        let matches = gen_tk == gen;
        let tps_tk = bench_decode_topk(&m, &prompt, &qe, kk);
        println!(
            "k={:<3} matches_exact={:<5} {:.2} tokens/sec  ids={:?}",
            kk, matches, tps_tk, gen_tk
        );
    }

    // per-worker top-k. The single-prompt sweep earlier suggested k=2 was
    // safe; the full 15-prompt harness below proved that wrong (k=1 fails
    // 3/15, k=2 fails 1/15) -- k=4 is the actual minimum that holds exact
    // across all 15 prompts, so that's the real default.
    let k = 4;

    // phase breakdown of the topk lm_head itself
    {
        use bittycrab::embd::lm_head_argmax_topk_rescore_timed;
        use bittycrab::model::{rmsnorm, D_MODEL, RMS_EPS, VOCAB};
        let h = vec![0.01f32; D_MODEL]; // representative-sized dummy hidden state
        let hn = rmsnorm(&h, &m.output_norm, RMS_EPS);
        let embd_raw = m.g.f16_tensor_bytes("token_embd.weight");
        let iters = 50;
        let (mut tq, mut ts, mut tr) = (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut ncand = 0;
        for _ in 0..iters {
            let (_, q, s, r, n) = lm_head_argmax_topk_rescore_timed(embd_raw, &qe, &hn, D_MODEL, VOCAB, k);
            tq += q;
            ts += s;
            tr += r;
            ncand = n;
        }
        println!("--- topk lm_head phase breakdown ({} iters, {} candidates rescored) ---", iters, ncand);
        println!("quant_act_sum : {:.4} ms/call", tq.as_secs_f64() * 1000.0 / iters as f64);
        println!("int8 vnni scan: {:.4} ms/call", ts.as_secs_f64() * 1000.0 / iters as f64);
        println!("f16 rescore   : {:.4} ms/call", tr.as_secs_f64() * 1000.0 / iters as f64);
    }

    profile_decode(&m, &prompt);

    // prompt-lookup decoding: no second model, drafts from repeats in the
    // sequence itself, verified in one batched pass -- exactly matches
    // greedy output by construction.
    {
        use bittycrab::pld::{bench_decode_pld, generate_pld};
        let ngrams = [4usize, 3, 2]; // try longest/most-specific match first
        let max_draft = 7; // 1 real + up to 7 drafted = 8 = MAX_BATCH
        let gen_pld = generate_pld(&m, &qe, &prompt, 20, k, &ngrams, max_draft);
        println!("generated ids (pld): {:?}", gen_pld);
        println!("generated text (pld): {:?}", decode(&tok, &gen_pld, true));
        println!("pld matches exact: {}", gen_pld == gen);

        let tps_pld = bench_decode_pld(&m, &prompt, &qe, k, &ngrams, max_draft);
        println!("PLD decode: {:.2} tokens/sec (60-token budget)", tps_pld);
    }

    // full 15-prompt accuracy+speed harness, k=1 (no per-worker safety
    // margin) -- same prompt set used on the Mojo side's original int8
    // accuracy check, so this is checked against the real bar, not just
    // the one demo prompt.
    {
        use bittycrab::pld::generate_pld;
        use bittycrab::model::EOS_ID;

        let prompts = PROMPTS;

        let k1 = std::env::var("PLD_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize);
        let ngrams = [4usize, 3, 2];
        let max_draft = 7;
        let max_new = 20;

        println!("--- 15-prompt harness: PLD k={} vs exact greedy ---", k1);
        let mut all_exact = true;
        let mut total_tokens = 0usize;
        let mut total_secs = 0f64;
        for (i, &pt) in prompts.iter().enumerate() {
            let ids = encode(&tok, pt, true);
            let exact = generate(&m, &ids, max_new);

            let t0 = std::time::Instant::now();
            let pld = generate_pld(&m, &qe, &ids, max_new, k1, &ngrams, max_draft);
            let elapsed = t0.elapsed().as_secs_f64();

            let matches = pld == exact;
            all_exact &= matches;
            total_tokens += pld.len();
            total_secs += elapsed;

            let stopped_early = exact.last() == Some(&EOS_ID);
            println!(
                "[{:2}] {:<45} exact={:<5} {:.2} tok/s ({} tok in {:.3}s){}",
                i,
                format!("{:?}", pt),
                matches,
                pld.len() as f64 / elapsed,
                pld.len(),
                elapsed,
                if stopped_early { "  [EOS]" } else { "" },
            );
            if !matches {
                println!("      exact: {:?}", exact);
                println!("      pld  : {:?}", pld);
            }
        }
        println!(
            "--- harness summary: all_exact={}  overall {:.2} tok/s ({} tokens / {:.3}s) ---",
            all_exact,
            total_tokens as f64 / total_secs,
            total_tokens,
            total_secs
        );

        // isolate: is the k=1 mismatch a PLD verify-logic bug, or purely
        // the int8-scan accuracy risk (same failure with plain, non-batched
        // topk decoding at k=1)?
        println!("--- isolating k=1 mismatches: plain (non-PLD) topk k=1 on the same prompts ---");
        for &i in &[5usize, 8, 13] {
            let ids = encode(&tok, prompts[i], true);
            let exact = generate(&m, &ids, max_new);
            let plain_tk1 = generate_topk(&m, &qe, &ids, max_new, 1);
            println!("[{:2}] plain topk k=1 matches exact: {}", i, plain_tk1 == exact);
        }
    }

    print_rss("end of run (peak-ish -- VmHWM is the true high-water mark)");
}
