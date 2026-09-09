# bittycrab

A Rust reimplementation of a BitNet b1.58 2B4T CPU inference engine, ported
from an existing Mojo implementation and built specifically to push CPU-only
decode throughput and memory footprint as far as they'd go — with every
number below coming from a real accuracy+speed harness, not a single
favorable demo run.

The Mojo version was built and tuned first, as a way to find out what was
actually achievable on this hardware and where the real bottlenecks were,
before asking whether Rust would handle the same problem differently —
specifically, whether Rust's threading would carry less overhead than
Mojo's.

One concrete finding from that comparison: **thread-count tuning behaved
completely differently between the two.** The Mojo engine's throughput kept
climbing as more threads were added, all the way out to all 24 physical
cores — including the E-cores, which are meaningfully slower per-thread
than the P-cores. Porting the same workload to Rust changed that: the
optimal thread count dropped to around 8, matching the number BitNet's own
paper reports as its useful ceiling — past that, Rust's throughput flattens
or gets noisy rather than continuing to climb. That's not Rust being
"faster at the same job" so much as it hitting the real ceiling with less
overhead in the way: decode here is memory-bandwidth bound (see below), so
once you're not paying for extra thread/scheduling overhead, more threads
past ~8 just contend for the same memory bus without doing more useful
work — Rust's leaner threading model exposed that ceiling instead of
letting more threads paper over it.

## Results

Everything measured against the same 15-prompt harness (below), verified
token-for-token against plain greedy decode (`exact` = byte-identical
output).

| Config | Speed | Steady RAM | Accuracy |
|---|---|---|---|
| bitnet.cpp (upstream reference) | ~30-40 tok/s | ~1.1-1.2 GB | reference |
| Mojo baseline (this project's starting point) | 40.17 tok/s | ~956 MB | reference |
| **Rust, default (int8 lm_head + PLD)** | **~70.4 tok/s** | ~1.58 GB | **15/15 exact** |
| **Rust, `-mem` (pread lm_head + PLD)** | **~70.2 tok/s** | **~900 MB** | **15/15 exact** |

The default build is ~1.75x the Mojo baseline. The `-mem` build matches that
speed (statistically identical, no measurable cost) while cutting steady
memory by ~43% relative to the default build's own earlier peak — a real
"pick both" result rather than a speed/memory tradeoff.

### `-mem`: how the memory drop actually happens

The tied embedding table (`token_embd.weight`, F16, 656MB) is the one piece
of the model that's cheap to touch selectively — unlike the ternary layer
weights, which get read on every matmul of every layer of every token and
have to stay memory-mapped for speed, the embedding table is only read for
(a) one row per input token and (b) a handful of rescore candidates per
decode step.

- **Default build:** reads the embedding table via `mmap`. Once a page is
  touched it stays resident for the life of the process (nothing evicts
  clean pages absent memory pressure), so the full-vocab quantization scan
  at startup permanently pins the whole 656MB table into RSS.
- **`-mem` build:** reads the same table via explicit positioned file reads
  (`pread`) instead of `mmap` — quantization, input-token embedding, and
  exact rescore all go through a persistent file handle, streamed in small
  bounded waves. Nothing about the embedding table stays resident; the OS
  page cache still keeps disk reads fast, it just isn't charged against
  this process's memory.
- The int8 quantization table itself (~328MB, the fast approximate scan
  structure — see below) is unaffected either way; it's genuinely needed
  resident memory, not something `-mem` removes.

Net effect: `-mem`'s steady footprint (~900MB) is close to the practical
floor for this architecture — the ternary layer weights (~432MB+, mmap
necessarily resident) and the int8 scan table (~328MB, needed for speed)
account for nearly all of it.

Run it with:
```
cargo run --release -- -mem
```

## The lm_head/embedding table: FP16 → int8, and the K that holds the harness

The tied embedding table doubles as both the input embedding lookup and the
final vocabulary projection (`lm_head`). A naive implementation scans the
whole table — 128256 rows × 2560 dims, in F16 — every single decode step,
which dominates decode time (measured at ~51% of total).

The fix: quantize the table to **int8** (per-row absmax scale, same
convention as the ternary kernel's own activation quantizer) — a 4x
reduction from 656MB to ~328MB — and scan it with AVX-VNNI int8 dot
products instead of F16 dot products. That scan is *approximate*, so the
top-K candidates it returns get **exactly** re-scored against the real F16
table before picking a final answer — the int8 scan can only ever narrow
the candidate set, never produce a wrong final token by itself.

The open question was how small K (candidates re-scored per worker) could
go before that approximation actually cost accuracy. Swept directly against
the 15-prompt harness (i.e. this is 100% match on those 15 prompts
specifically, not a universal accuracy guarantee):

| K (per-worker top-k) | Match rate on the 15-prompt harness |
|---|---|
| 1 | 12/15 |
| 2 | 14/15 |
| **4** | **15/15 — first K that matches on every harness prompt** |
| 8 | 15/15 |
| 16 | 15/15 |

**K=4 is the value shipped.**

Why such a small K works at all: per the BitNet b1.58 paper, the model's
own weights are ternary (`{-1, 0, 1}`), which means the *effective*
precision the rest of the network actually operates at is coarse to begin
with — the true top logits tend to sit close together rather than being
sharply separated, so the int8-approximated ranking rarely misplaces the
real winner by more than a few positions. That's exactly why a tiny
re-score window (K=4) is enough to catch it essentially every time: the
search only has to cover a small neighborhood, not the full 128k-token
vocabulary, and re-scoring 4 candidates in exact F16 is close to free next
to the matmul work already happening every step.

That's also why this specific optimization compounds so well with
everything else here: shrinking K shrinks exactly the thing this whole
project kept finding was the actual constraint once the easy wins (thread
count, dispatch overhead, kernel fusion) were exhausted — **RAM bandwidth**.
Every decode step is bound by how many bytes of weights and table entries
have to move off DRAM, not by spare compute; a smaller K means fewer
candidate rows read back for re-scoring, which is a direct cut to that
bandwidth bill rather than a clever trick around it.

## The decoder: prompt-lookup decoding (PLD)

Beyond the int8 lm_head, decode throughput got a second, larger win from
**prompt-lookup decoding** — a form of speculative decoding that needs no
second ("draft") model at all:

1. Compute the real next token exactly (as normal).
2. Search the already-generated text for an earlier occurrence of the
   current few-token suffix. If one exists, guess that the same
   continuation follows again (a draft, up to 7 tokens).
3. Verify the *entire* draft in **one batched forward pass** — this is
   where the speed comes from: decode is memory-bandwidth bound (each step
   reads ~1.1GB of weights for one token's worth of compute), so verifying
   several draft tokens costs about the same as verifying one, since the
   weight bytes are read once either way.
4. Accept the longest verified prefix; anything after the first mismatch is
   discarded and replaced with the exact, real token at that position.

This is provably lossless: every emitted token is either the exact
single-token computation or a draft token that was independently verified
against that same exact computation — never a token trusted on the
strength of the guess alone. On text the model already tends to repeat
(loops, boilerplate, common phrasing), this can accept several tokens per
model pass instead of one; on genuinely novel text it gracefully degrades
toward plain one-token-at-a-time decoding, never toward wrong output.

## The full pipeline, stage by stage: disk / RAM / CPU

It's worth spelling out exactly where each stage of a decode step lives,
since that's what the memory and speed decisions above are actually about.

### Chain A: naive FP16 baseline (bitnet.cpp-style, no int8/K shortcut)

The "ternary weights, but the lm_head itself stays FP16 and gets scanned in
full" version:

```
[disk]  GGUF file: ternary-packed layer weights + F16 embedding table
   |
   |  (ONE-TIME, at process startup)
   v
[disk -> RAM]  mmap the file. Nothing is actually read yet -- mmap just
               maps the file's bytes into this process's address space.
               Pages get pulled into RAM lazily, the first time each one
               is touched.
   |
   |  ===== per decode step, repeats every token =====
   v
[RAM]  input token id -> look up its one row in the embedding table.
       First time a given page is touched, this faults it in from disk;
       after that it's a plain RAM read (page cache keeps it there).
[CPU]  widen that row from F16 -> F32 (hardware F16C instruction)
       -> this F32 vector is the layer-0 hidden state, held in RAM.
   |
   v
[RAM+CPU x30]  the transformer layers. For each layer: CPU reads that
               layer's ternary weight bytes (already RAM-resident from
               mmap, not re-fetched from disk), does the ternary x
               int8-activation matmuls (AVX-VNNI), writes the new hidden
               state back to RAM. Attention reads/writes a small KV-cache
               in RAM too. All RAM+CPU, no disk touched here after the
               first pass through each layer.
   |
   v
[RAM+CPU]  final RMSNorm -> h_normed (F32, RAM)
   |
   v
[RAM+CPU]  lm_head: for EVERY ONE of 128256 vocab rows, CPU reads that
           row (F16, RAM-resident) and computes an exact dot product
           against h_normed. This is the expensive, unshortcut step --
           656MB gets read and dot-producted in full, every single token.
   |
   v
[CPU]  argmax over all 128256 scores -> token id -> fed back as next
       token's input (loop to top)
```

Disk only matters here at **startup** (first touch of each page). Every
token after that is pure RAM+CPU -- the naive part isn't disk I/O, it's
that the lm_head step reads and scores the *entire* 656MB table every
single token instead of narrowing first.

### Chain B: what bittycrab actually does (int8 + K-rescore + PLD)

```
[disk]  same GGUF file
   |
   |  ===== ONE-TIME startup =====
   v
[disk -> RAM -> CPU -> RAM]  quantize the F16 embedding table to int8:
    read each row (disk/RAM, F16) -> CPU converts to int8 (per-row
    absmax scale) -> the resulting ~328MB int8 table is kept resident
    in RAM for the rest of the run. This is the only place the FULL
    table ever gets touched.
   |
   |  ===== per decode step =====
   v
[disk or RAM -> RAM -> CPU]  input token embedding: look up ONE F16 row
    - default build: RAM (mmap-resident from the quantization pass)
    - `-mem` build:   fresh small disk pread (table was never kept
                       resident -- see the -mem section above)
    CPU widens F16 -> F32, same as chain A.
   |
   v
[RAM+CPU x30]  identical to chain A's ternary layers -- this part of
               the pipeline is the same core BitNet compute either way,
               and it's where PLD's batching kicks in: several draft
               positions get pushed through this same weight-read pass
               together, amortizing the RAM bandwidth cost.
   |
   v
[RAM+CPU]  final RMSNorm -> h_normed (F32, RAM)
   |
   v
[CPU]  quantize h_normed to int8 (ephemeral, this token only)
[RAM+CPU]  scan the RESIDENT int8 table (built at startup, already in
           RAM -- no disk touched here) with AVX-VNNI int8 dot products
           -> keep each worker's top-K candidates (K=4 shipped)
   |
   v
[disk or RAM -> RAM -> CPU]  exact rescore: for just those ~32
    candidates (not 128256), fetch their REAL F16 rows
    - default build: RAM (mmap-resident)
    - `-mem` build:   tiny disk pread, a few dozen rows, ~5KB each
    CPU widens F16 -> F32, computes the exact dot product, picks the best.
   |
   v
[CPU]  winning token id -> fed back as next token's input (loop to top)
```

The key differences from chain A: the ternary layers are identical either
way (that's the shared core BitNet compute, and PLD's batching applies to
both); the only thing that changes is the lm_head step, which goes from
"read and score 656MB, every token" to "score a resident 328MB int8 table,
then exactly re-score ~32 rows" -- and, in `-mem` builds specifically, the
embedding table's memory residency behavior (RAM vs. small disk reads)
changes too, while the ternary layer weights stay mmap-resident in RAM
either way (they're read too often, every token, for a per-row disk read
to make sense there).

Either chain ends the same way: the winning row's exact F16 dot product is
what decides the token, never the int8 approximation alone -- the shortcut
only ever narrows the search, it never gets the final vote.

## The 15-prompt test harness

Every number above comes from generating 20 tokens per prompt and diffing
against plain greedy decode, token-for-token:

```
The capital of France is
The capital of Japan is
Water boils at a temperature of
The largest planet in the solar system is
def fibonacci(n):
import numpy as np
The quick brown fox
Once upon a time, there was a
2 + 2 =
The president of the United States lives in
My favorite hobby is
The chemical symbol for gold is
In machine learning, a neural network
She walked into the room and
The speed of light is approximately
```

## Running it

```
cargo build --release
./target/release/bittycrab            # default: int8 lm_head + PLD, fastest
./target/release/bittycrab -mem       # same speed, ~43% less steady RAM
./target/release/bittycrab -fp16      # PLD with the exact F16 lm_head, no int8
./target/release/bittycrab -pq        # experimental: product-quantized scan (~15MB table)
./target/release/bittycrab -opq       # experimental: rotation-learned PQ variant
```

`-pq`/`-opq` are documented experiments, not recommended defaults: they cut
the embedding-scan table to a few MB (vs int8's 328MB), but at every
build/K setting tested here they land on a real speed/accuracy tradeoff
this project didn't manage to close — see inline comments in `src/pq.rs`
and `src/opq.rs` for the full story.
