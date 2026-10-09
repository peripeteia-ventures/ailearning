---
{
  "slug": "serving-kv-cache",
  "title": "What happens at inference: prefill, decode, and the KV cache",
  "category": "inference",
  "summary": "Follow one request through a serving engine: a fast parallel prefill, a slow token-by-token decode, and the KV cache that makes decode possible. Learn to size that cache, why it caps concurrency, and how paging, prefix caching and preemption manage it.",
  "difficulty": "Systems",
  "minutes": 30,
  "prerequisites": ["attention-from-scratch", "transformer-foundations"],
  "learningObjectives": [
    "Explain why prefill is compute-bound and decode is memory-bandwidth-bound, using arithmetic intensity.",
    "Define TTFT, TPOT/ITL and throughput, and diagnose which phase a latency complaint points to.",
    "Explain what the KV cache stores, why it is safe to reuse, and why queries are not cached.",
    "Calculate KV cache bytes per token and per sequence from model geometry, and turn a GPU memory budget into a concurrency limit.",
    "Describe how PagedAttention blocks, prefix caching, and preemption manage KV memory, and the failure modes of each."
  ]
}
---

# Sections

## One request, two very different jobs {#two-phases}

When you send a prompt to a chat model, a program called a **serving engine** (vLLM, TensorRT-LLM, SGLang, TGI and friends) runs the model on a GPU and streams tokens back to you. From the outside it looks like one continuous thing. On the inside it's two jobs with completely different personalities, and nearly every serving decision you'll be asked about in an interview comes back to that split.

Quick refresher on the contract from **Text in, next token out: tokens, embeddings, and sampling**: the model takes a sequence of token IDs and produces **logits**, one score per vocabulary entry, for what the next token should be. You sample a token, append it, and go again. That's the **autoregressive loop**.

Here's the restaurant version. A table of eight sits down and hands you their whole order at once. The kitchen can fire all eight dishes in parallel, so the ovens are full and nobody's idle. That's **prefill**: the entire prompt is known up front, so the model processes all prompt tokens in one big parallel pass. Then dessert: the table orders one spoonful at a time, and each spoonful depends on how they liked the last one. The kitchen does a tiny amount of cooking per trip, and most of the time goes to walking back and forth to the pantry. That's **decode**: one new token per step, and each step can't begin until the previous token has been chosen.

```text
 prompt: "The capital of France is"          (5 tokens, all known)

 PREFILL  ── one parallel pass over all 5 positions ──►  logits → " Paris"
          (big matrix × matrix work, GPU busy)              ▲ first token

 DECODE   step 1: feed " Paris"  ──► logits → "."
          step 2: feed "."       ──► logits → "<eos>"      (stop)
          (one position per step, matrix × vector work, GPU mostly waiting on memory)
```

Note that prefill's output is already the first generated token. Prefill runs the prompt and samples token #1, and then decode produces token #2 onward.

Why does this matter? Because the two phases stress different parts of the hardware, they show up as different latency numbers, and they fail in different ways. A long prompt makes the *first* token slow. A busy server makes the *later* tokens slow. If you only report "average request time", you can't tell which one you have.

> 🎬 **Animation — prefill vs decode:** a 5-token prompt "The capital of France is" enters a 2-layer model drawn as two stacked boxes. Step 1 (prefill): all 5 token columns light up together and flow through both layers at once; a bar at the bottom labelled "GPU compute" fills to ~90%. The output "Paris" pops out. Step 2 (decode): only a single column (" Paris") enters and flows through; the compute bar drops to ~5% while a second bar labelled "memory reads" fills to 100%. Step 3: repeat for "." and then "<eos>", one column at a time, with a clock ticking at each step to show decode is sequential.

## Why decode is starved for memory bandwidth {#arithmetic-intensity}

To see why the phases feel so different, you need one idea from hardware: **arithmetic intensity**, which is the number of floating-point operations (FLOPs) you do per byte you move from memory.

A GPU has two speed limits. It can do a huge number of multiply-adds per second (compute), and it can move a large but much smaller number of bytes per second from its main memory, called **HBM** (high-bandwidth memory), into the compute units (bandwidth). Picture a chef who can chop incredibly fast but has to fetch every ingredient from a pantry down the hall. If each ingredient needs a lot of chopping, the chef is the bottleneck: you're **compute-bound**. If each ingredient needs one quick slice, the hallway is the bottleneck: you're **memory-bandwidth-bound**.

Let's put round, **illustrative** numbers on it, roughly the shape of a modern datacenter GPU: 1,000 TFLOP/s of bf16 compute (10¹⁵ FLOPs per second) and 3 TB/s of HBM bandwidth.

```formula
ridge point = peak FLOP/s ÷ peak bytes/s = 10¹⁵ ÷ (3 × 10¹²) ≈ 333 FLOPs per byte
```

The **ridge point** is the intensity at which both limits hit at the same time. Below it, bandwidth limits you. Above it, compute does.

Now take a model with N = 8 billion parameters stored in bf16 (2 bytes each), so 16 GB of weights. A forward pass costs about **2 FLOPs per parameter per token** (one multiply, one add), so 2N = 16 GFLOPs per token, ignoring the attention part for now.

**Decode, batch of 1.** To make one token, you have to stream all 16 GB of weights through the compute units, and you do 16 GFLOPs with them. Intensity is 16 × 10⁹ ÷ 16 × 10⁹ = **1 FLOP per byte**. That's about 333× below the ridge point. The time is set by bandwidth: 16 GB ÷ 3 TB/s ≈ **5.3 ms per token**, or at most ~190 tokens/s, and the compute units sit idle more than 99% of the time. Under the hood this is matrix × *vector* work: one token's activation vector times each weight matrix.

**Prefill, 2,000-token prompt.** You still read the 16 GB of weights once, but now you apply them to 2,000 tokens at once (matrix × *matrix*). FLOPs are 2,000 × 16 GFLOPs = 3.2 × 10¹³, so intensity is about **2,000 FLOPs per byte**, comfortably above the ridge. Now compute is the limit: 3.2 × 10¹³ ÷ 10¹⁵ ≈ **32 ms** at perfect utilization. Real kernels don't hit peak, so treat that as a floor.

```text
 intensity (FLOPs/byte, log scale)
   1         10        100   333   1,000   2,000
   │─────────│─────────│──────┼──────│──────│
   ▲                          ▲             ▲
 decode, batch 1          ridge point    prefill, 2,000 tokens
 (bandwidth-bound)                        (compute-bound)
```

**The consequence: batching is almost free for decode.** If 64 sequences decode together, you read the 16 GB of weights once and use them 64 times, so intensity jumps to ~64 FLOPs/byte and each step takes barely longer than a batch of 1. That's why serving engines batch many users' decode steps together, and why throughput (tokens/s summed across users) climbs steeply with batch size while each user's per-token speed barely changes at first. *We'll go through how engines form and reshape those batches in detail in* **Serving many users: batching, scheduling, and speculative decoding**.

But there's a catch that the weight-only math hides, and it's the star of this article. Every sequence also has to read its own **KV cache** on every decode step, and unlike weights, that data is *not* shared across the batch. With long contexts it can outweigh the weights. Hold that thought for the next few sections.

The honest senior caveat: "prefill is compute-bound, decode is bandwidth-bound" is a strong default, not a law. A tiny prompt's prefill can be bandwidth-bound, and a huge decode batch can approach the ridge. It depends on hardware, lengths, kernels and batch size.

> 🎬 **Animation — the roofline:** draw a log-log plot, x-axis "FLOPs per byte" (1 to 10,000), y-axis "achieved TFLOP/s". A diagonal line rises from the left (labelled "bandwidth limit: 3 TB/s") and meets a flat ceiling at 1,000 TFLOP/s at x ≈ 333 (the ridge). Step 1: drop a dot at x = 1 on the diagonal labelled "decode, batch 1 (≈3 TFLOP/s achieved)". Step 2: slide the dot right to x = 8, 32, 64 as "batch size" grows, showing throughput rising along the diagonal. Step 3: drop a second dot at x = 2,000 on the flat ceiling labelled "prefill, 2,000-token prompt".

## Measuring what the user actually feels {#latency-metrics}

Since the two phases behave differently, you measure them separately. These names come up constantly, so get them crisp.

- **TTFT (time to first token):** from when the request arrives to when the first output token comes out. It includes queueing (waiting for a slot), tokenization, and prefill. This is the "is it thinking?" pause the user stares at.
- **ITL (inter-token latency):** the gap between two consecutive streamed tokens. This is how smoothly the text flows.
- **TPOT (time per output token):** the average gap for one request, computed as (end-to-end latency − TTFT) ÷ (output tokens − 1). vLLM's metrics docs define it exactly this way. The −1 is there because the first token is already counted in TTFT.
- **End-to-end (E2E) latency:** arrival to last token.
- **Throughput:** total output tokens per second across *all* users (sometimes input + output tokens). This is what the finance team cares about, because it sets cost per token.

**Worked example.** A request waits 100 ms in the queue, prefills in 200 ms, and streams 200 tokens, finishing 5.3 s after arrival. TTFT = 0.3 s. TPOT = (5.3 − 0.3) ÷ 199 = 5.0 ÷ 199 ≈ **25.1 ms per token**, about 40 tokens/s. That's comfortably faster than people read.

```text
 arrival         first token                                   last token
   │── queue ──│── prefill ──│ t │ t │ t │ t │ ... │ t │ t │
   0         0.1 s         0.3 s ◄─ ITL gaps (≈25 ms each) ─► 5.3 s
   └──────── TTFT ─────────┘
   └─────────────────────── E2E latency ─────────────────────┘
```

**How to read symptoms:**

| Symptom | Likely cause | Phase |
|---|---|---|
| High TTFT, normal TPOT | Long prompts, or requests queueing for memory | Queue / prefill |
| Normal TTFT, high TPOT | Too many long contexts decoding together; big prefills interrupting decode | Decode |
| Periodic stalls mid-stream | Sequences being preempted and resumed (covered below) | Decode |
| Great throughput, angry users | Batch so large that each step is slow | Tradeoff |

That last row is the central tension. Throughput and per-user latency pull against each other, and a serving team usually sets **SLOs** (service-level objectives, e.g. "p95 TTFT under 1 s, p95 TPOT under 50 ms") and then maximizes throughput subject to them. Always look at percentiles (p50, p95, p99), not averages, because a few preempted requests can hide inside a nice mean.

> 🎬 **Animation — TTFT vs TPOT timeline:** a horizontal timeline for one request. Step 1: a grey "queue" bar from 0 to 0.1 s. Step 2: an orange "prefill" bar from 0.1 to 0.3 s, and a bracket labelled "TTFT = 0.3 s". Step 3: small blue ticks appear one by one every ~25 ms, each a token, with a bracket over two neighbouring ticks labelled "ITL". Step 4: a bracket from 0.3 s to 5.3 s labelled "199 gaps → TPOT = 5.0 ÷ 199 ≈ 25 ms". Step 5: a second request's timeline appears below it with a long prefill (TTFT 2 s) but the same tick spacing, to show the two metrics move independently.

## The KV cache: keep the keys and values, skip the recomputation {#kv-cache}

Now the core idea. Recall from **Attention from scratch: how tokens talk to each other** that inside every attention layer each token gets three vectors, made by multiplying its hidden state by the learned matrices W_Q, W_K and W_V:

- a **query** (Q): "what am I looking for?"
- a **key** (K): "what do I contain, as a label others can match?"
- a **value** (V): "what do I hand over if you pick me?"

The current token compares its query against every earlier token's key, turns the scores into weights with softmax, and takes a weighted average of the values. Because of the **causal mask**, a token only ever looks at itself and tokens *before* it.

Now think about decode naively. To generate token 1,001 you'd run the whole 1,000-token history through the model again to get every token's K and V at every layer. Then for token 1,002 you'd do it all again for 1,001 tokens. That's absurdly wasteful, and here's why: **token 37's key and value at layer 5 never change.** They depend only on the tokens at positions 1 to 37 (causal mask) and on fixed weights. Nothing you generate later can reach back and alter them. So compute them once, store them, and reuse them forever. That stored pile is the **KV cache**.

The analogy: you're doing long division on paper. When you add a new line, you don't recopy and redo every line above it. You glance up at the scratch work that's already there. The KV cache is the model's scratch work.

```text
 layer ℓ's KV cache for one sequence (rows = token positions, cols = features)

            K cache                    V cache
 pos 1   [k₁ ........]             [v₁ ........]
 pos 2   [k₂ ........]             [v₂ ........]
 ...           ...                       ...
 pos t   [k_t .......]             [v_t .......]
 ─────────────────────────────────────────────── decode step t+1:
 pos t+1 [k_{t+1} ...]  ◄─ append  [v_{t+1} ...]  ◄─ append
          ▲ the new token's q_{t+1} is scored against ALL rows k₁…k_{t+1},
            then used to average ALL rows v₁…v_{t+1}. Then q is thrown away.
```

So one decode step at each layer does this: compute q, k, v for **just the new token**, append k and v to the cache, then attend over the whole cache. Prefill's job, in cache terms, is to fill in rows 1..T for every layer in one parallel pass.

**Why not cache queries too?** A query is only used by its own token, at the moment that token is processed. Token 37's query was needed to compute token 37's output, and no future token will ever look at it again. Keys and values are what *future* tokens read. So caching Q would buy nothing.

**How much work does it save?** Say you have a 1,000-token prompt and generate 1,000 tokens. Without a cache, generating token number t means pushing all t tokens through the full model, so the total is 1,000 + 1,001 + … + 1,999 = 1,000 × (1,000 + 1,999) ÷ 2 = **1,499,500 token-passes**. With a cache, every token passes through the model exactly once: **2,000 token-passes**. That's about 750× less projection and MLP work.

Three things the cache is *not*, because people conflate them:

1. **It doesn't make attention free.** Each decode step still reads the entire cache, so a step at position 100,000 reads 100,000 rows per layer. The cache removes *recomputation*, not *reading*. This is exactly the per-sequence memory traffic I warned about in the arithmetic-intensity section.
2. **It isn't an answer cache.** It holds intermediate tensors for one exact token sequence, not responses or anything semantic.
3. **It isn't optional bookkeeping.** For a running sequence it is *required* state. Throw it away and the only way back is to recompute it.

> 🎬 **Animation — KV cache append:** a 2-layer model with a 4-token prompt "the cat sat on". Each layer has two tall grids labelled "K" and "V". Step 1 (prefill): four rows appear in both grids of both layers at once, each tagged with its token. Step 2 (decode "the"): a single new token enters; at each layer a small "q" vector appears, draws lines to all 5 K rows (line thickness = attention weight), then a weighted blend of the V rows flows upward; a 5th row is appended to K and V in both layers; the "q" fades out. Step 3: repeat for one more token, with the older rows glowing "reused" and only the new row glowing "computed".

## How big is the cache? The formula and a worked example {#kv-size}

The cache holds one key vector and one value vector, per token, per layer, per KV head. Write down the dimensions and multiply.

```formula
KV bytes per token = 2 × L × h_kv × d_head × s
KV bytes total     = (KV bytes per token) × Σᵢ Tᵢ
```

- **2** — one K and one V.
- **L** — number of transformer layers. Each layer has its own cache.
- **h_kv** — number of key/value heads. In classic multi-head attention this equals the number of query heads h, so h_kv · d_head = d, the model width.
- **d_head** — width of each head.
- **s** — bytes per element: 2 for fp16/bf16, 1 for 8-bit.
- **Σᵢ Tᵢ** — the total cached tokens across all live sequences i. If every sequence has the same length T and there are B of them, this is just B × T.

That second line is worth staring at. KV cost depends on **resident tokens**, meaning tokens whose K and V are currently sitting in GPU memory, not on request count. One 32K-token request costs the same as thirty-two 1K-token requests.

A note on h_kv. Many modern models let several query heads *share* one key/value head, which shrinks h_kv (and the cache) without shrinking h. That's **grouped-query attention (GQA)**, and the extreme version where all query heads share a single KV head is **multi-query attention (MQA)**. For sizing, the rule is simple: **use KV heads, not query heads.** *We'll go through why these variants work and what they cost in quality in* **Making attention cheaper: GQA, FlashAttention, and long context**.

**Worked example (hypothetical model).** L = 32 layers, h = 32 query heads, h_kv = 8 KV heads, d_head = 128, bf16 (s = 2).

```text
 per token  = 2 × 32 × 8 × 128 × 2
            = 2 × 32 = 64
              64 × 8 = 512
              512 × 128 = 65,536
              65,536 × 2 = 131,072 bytes = 128 KiB

 one 8,192-token sequence = 131,072 × 8,192 = 1,073,741,824 bytes = exactly 1 GiB
 16 such sequences        = 16 GiB
```

If the same model used full multi-head attention (h_kv = 32), it would be 4× bigger: 512 KiB per token and 4 GiB per 8K sequence. That factor of four is why GQA became so popular.

**Sanity check against a published number.** The PagedAttention paper says OPT-13B needs about 800 KB of KV cache per token. OPT-13B has 40 layers and width d = 5,120 with full multi-head attention, so 2 × 40 × 5,120 × 2 bytes = 819,200 bytes = 800 KiB. The formula checks out.

**Things the simple formula leaves out** (the interviewer follow-ups):

- **Sliding-window layers** only keep the last W tokens, so for those layers, replace T with min(T, W). Models that mix windowed and full layers need a per-layer sum.
- **Quantized KV** (e.g. 8-bit) halves s but adds scale metadata. *More on this in* **Quantization: spending fewer bits per weight**.
- **Multi-GPU:** with tensor parallelism the KV heads are usually split across GPUs, but if h_kv is smaller than the GPU count, heads get *replicated*, so you can't just divide the global total by the GPU count.
- **Allocation overhead:** the engine allocates in blocks, so partially filled blocks cost real memory (see the paging section).

## Memory, not compute, caps how many users you can serve {#memory-budget}

Put the last two sections together and you get the single most important serving fact: **a GPU runs out of KV memory long before it runs out of compute for decode.** Batching makes decode cheap per token, but every extra sequence in the batch needs its own cache, and the cache grows by one row per layer every step.

Think of a parking garage. The model weights are the building itself, a fixed cost. Each active conversation is a car that gets *longer* every second it's parked. The garage fills up based on total car length, not the number of cars.

**Worked budget (illustrative).** An 80 GiB GPU serving our hypothetical 8-billion-parameter model from the last section:

| Slice | GiB | Notes |
|---|---|---|
| Weights | 15 | 8 × 10⁹ params × 2 bytes = 16 × 10⁹ bytes ≈ 14.9 GiB |
| Activations, workspaces, runtime | 9 | Illustrative; measure this on your engine |
| Safety margin | 8 | Headroom for spikes and fragmentation |
| **KV pool** | **48** | Everything left over |

At 128 KiB per token, 48 GiB holds 48 × 8,192 = **393,216 token slots** (since 1 GiB ÷ 128 KiB = 8,192).

Now the trap. Suppose every request has an 8,192-token prompt. That's 1 GiB each, so 48 requests exactly fill the pool with prompts alone. The moment any of them generates a token, there's nowhere to put it. If each request may also generate up to 2,048 tokens, the worst-case footprint per sequence is 10,240 tokens × 128 KiB = **1.25 GiB**, and the safe count is floor(48 ÷ 1.25) = floor(38.4) = **38 sequences**.

```text
 KV pool: 48 GiB = 393,216 token slots
 ┌──────────────────────────────────────────────────────────────┐
 │ 48 × 8K prompts ████████████████████████████████████████████ │  full at step 0!
 └──────────────────────────────────────────────────────────────┘
 ┌──────────────────────────────────────────────────────────────┐
 │ 38 × (8K prompt ███ + 2K growth ░)  ██████████████████████░░░░│  fits worst case
 └──────────────────────────────────────────────────────────────┘
```

Two lessons interviewers love:

1. **"Max concurrent requests" is the wrong knob on its own.** Thirty-eight 10K-token sequences and four hundred 1K-token chats use about the same memory. Admission should look at tokens (or blocks), not just request count.
2. **Reserving the worst case is safe but wasteful.** Most answers stop long before 2,048 tokens, so reserving the maximum for everyone leaves memory idle. Allocating incrementally lets you admit more, but then you need a plan for when growth outruns memory. That plan is preemption, which comes last in this article.

Now tie it back to bandwidth. At 38 sequences × ~9K tokens average, each decode step reads ~15 GiB of weights *plus* ~43 GiB of KV cache (38 × 9,216 tokens × 128 KiB ≈ 42.75 GiB). The per-sequence KV reads now dominate step time, which is why long-context decode gets slower per token as the batch fills. Batching amortizes weights, not caches.

> 🎬 **Animation — the filling garage:** an 80 GiB bar split into coloured segments: weights 15 (dark), runtime 9 (grey), margin 8 (hatched), KV pool 48 (empty). Step 1: request blocks of width 1 GiB drop into the KV pool one at a time with a counter "requests: 1, 2, … 48" until it's exactly full. Step 2: each block tries to grow a sliver (the first generated token) and a red "OUT OF MEMORY" flash appears. Step 3: reset; this time each request drops in as 1 GiB solid plus 0.25 GiB dotted "reserved growth", and the counter stops at 38 with a small leftover gap of 0.5 GiB.

## Paging the cache: blocks instead of one big slab {#paged-attention}

The obvious way to store a sequence's cache is one contiguous tensor sized for the maximum length, say 8,192 tokens. It's simple and it's terrible. A chat that ends after 500 tokens has reserved 8,192, so 7,692 slots (94%) sit empty the whole time. The PagedAttention paper measured existing systems of the time and found only **20.4%–38.2%** of KV memory actually held token states. The rest was lost to three kinds of waste:

- **Reservation:** slots held for future tokens that might never come.
- **Internal fragmentation:** over-provisioning for the maximum possible length.
- **External fragmentation:** free memory broken into gaps that are too small or awkward to fit a new request's slab.

**PagedAttention** (the idea behind vLLM, from Kwon et al., 2023) borrows the fix operating systems use for RAM: **virtual memory with pages**. Split the KV pool into fixed-size **blocks**, each holding the K and V for, say, 16 tokens (vLLM's default block size in the paper; TensorRT-LLM's docs list 128 tokens as its default). A sequence gets blocks one at a time as it grows, and they can live anywhere in memory. A per-sequence **block table** maps "logical block 0, 1, 2…" to "physical block 7, 2, 11…". The attention kernel follows the table to find each block.

The library analogy: a long book whose chapters are shelved wherever there's free space, plus an index card saying chapter 1 is on shelf 7, chapter 2 on shelf 2, and so on. Nobody needs a single empty stretch of shelf long enough for the whole book.

```text
 sequence A (35 tokens, block size 16)

 block table A                 physical KV pool (any free block will do)
 ┌─────────┬──────────┐        [0 free][1 B ][2 A ][3 free][4 B ][5 A ][6 free][7 A ]
 │ logical │ physical │                     ▲                   ▲             ▲
 │    0    │    2     │ ────────────────────┘                   │             │
 │    1    │    7     │ ────────────────────────────────────────┼─────────────┘
 │    2    │    5     │ ────────────────────────────────────────┘
 └─────────┴──────────┘
 blocks 2 and 7 hold 16 tokens each; block 5 holds 3 tokens (13 slots empty)
```

**Worked example.** Block size 16, 128 KiB per token, so one block = 16 × 128 KiB = **2 MiB** (across all layers). A 33-token sequence needs ceil(33 ÷ 16) = 3 blocks = 6 MiB allocated, versus 33 × 128 KiB = 4,224 KiB = 4.125 MiB of real state. The 15 empty slots in the tail block waste 15 × 128 KiB = 1.875 MiB. So paging doesn't eliminate waste. It caps it at under one block per sequence, instead of "whatever you over-reserved". That's a big deal for long sequences and still noticeable for many tiny ones.

**The block-size tradeoff.** Smaller blocks mean less tail waste and finer-grained sharing, but bigger block tables, more lookups, and less efficient memory access in the kernel. Larger blocks are the opposite.

**What paging does *not* change:** the math. Attention still reads every past token; it just finds them through a table. Paging is a storage layout, not an approximation.

The bonus is that blocks make **sharing** easy. Two sequences can point their block tables at the same physical block. The engine keeps a **reference count** per block (how many sequences point at it) and frees a block only when the count hits zero. If a sequence needs to *write* into a shared block, for example when two samples from the same prompt diverge mid-block, the engine does **copy-on-write**: it copies the block to a private one first and then writes. The paper uses exactly this for parallel sampling and beam search.

> 🎬 **Animation — block table in action:** a physical pool drawn as a row of 12 numbered boxes, each with 16 small slots. Step 1: sequence A's prefill of 20 tokens claims physical block 7 (16 slots filled) and block 2 (4 slots filled); A's block table on the left shows "0→7, 1→2". Step 2: sequence B arrives and claims blocks 4 and 9, showing blocks are non-adjacent. Step 3: A decodes 12 more tokens; slots in block 2 fill up, then A grabs free block 11 and the table gains "2→11". Step 4: B finishes; blocks 4 and 9 turn back to "free" with no gap left behind. Step 5: zoom into block 11 showing 0 of 16 used and label "tail waste < 1 block".

## Prefix caching: never prefill the same thing twice {#prefix-caching}

Now that blocks can be shared, a big optimization falls out. Lots of requests start the same way: the same long system prompt, the same few-shot examples, the same document in a multi-turn chat about it. If the KV blocks for that prefix are still in memory, a new request can **point at them** and skip prefilling those tokens. That's **prefix caching** (vLLM calls it automatic prefix caching).

**Worked example.** A 2,000-token system prompt, 40 concurrent sequences, our 128 KiB/token model, block size 16.

| | Memory for the prefix | Prefill work per new request |
|---|---|---|
| No sharing | 40 × 2,000 × 128 KiB = 10,240,000 KiB ≈ **9.77 GiB** | 2,000 tokens |
| Shared prefix | 2,000 ÷ 16 = 125 blocks × 2 MiB = **250 MiB** | 0 tokens for the prefix |

That's roughly 40× less memory for the prefix, and on our illustrative GPU it skips about 2 × 8 × 10⁹ × 2,000 = 3.2 × 10¹³ FLOPs (≈32 ms at peak) of prefill per request, straight off TTFT.

**When is sharing actually correct?** This is the part to get right, because a wrong cache hit produces silently wrong output. A cached block is only valid if *every* input that fed into those K and V values matches:

- the **exact token IDs** of that block, *and of everything before it* (because each token's K and V depend on all earlier tokens through the layers below);
- the same **model weights**, including any adapter such as a LoRA fine-tune;
- any other inputs mixed in, like images in a multimodal prompt.

vLLM enforces the "everything before it" rule neatly. Each full block's hash combines the **parent block's hash**, the block's own tokens, and **extra keys** (adapter IDs, image hashes, a cache salt). Because each hash chains through its parent, a match on block 5 implies blocks 0–4 matched too. vLLM (and TensorRT-LLM) only cache **full blocks**, so a 2,010-token shared prefix with 16-token blocks reuses 125 blocks (2,000 tokens) and recomputes the last 10.

```text
 request 1: [SYS PROMPT ........................][user: "summarise X"]
 request 2: [SYS PROMPT ........................][user: "translate Y"]
             └── identical blocks → shared ─────┘└─ diverges → private ┘

 request 3: [2025-06-01 12:00:03][SYS PROMPT ......][user: ...]
             └ timestamp first → every later block's hash differs → zero reuse
```

**Practical consequences you should say out loud in an interview:**

- **Put stable content first.** A timestamp or user ID at the *start* of the prompt kills reuse for everything after it. Put variable content at the end.
- **Render the chat template identically.** The template is the formatting that wraps each message with role markers before tokenization. One extra space changes the tokens and breaks the match.
- **A matching suffix is useless.** If two prompts differ at token 3 and agree afterwards, the later tokens' K and V still differ, because they were computed with a different history.
- **Replicas don't share caches.** Behind a load balancer, the next turn of a conversation may land on a different GPU with a cold cache. Cache-aware or sticky routing fixes that. *We'll get to routing in* **Serving many users: batching, scheduling, and speculative decoding**.

**Security: the cache is a side channel.** A cache hit makes TTFT faster. So an attacker sharing the server can *guess* a prefix ("Patient John Smith has…"), send it, and time the response. A fast first token suggests someone else recently sent that prefix. vLLM's security guidance describes exactly this attack and the fix: a **cache salt**, a secret mixed into the first block's hash, so only requests carrying the same salt can share blocks. Treat the salt as a secret, scope it to a trust boundary (per tenant, meaning per customer or team sharing the deployment), and set it server-side from the authenticated identity, never from a client-supplied field. Isolation costs you some sharing, so measure capacity with it turned on.

> 🎬 **Animation — chained block hashes:** three requests drawn as rows of blocks. Step 1: request 1 prefills 4 blocks; under each block its hash is drawn as "h₀ = H(tokens₀)", "h₁ = H(h₀, tokens₁)", and so on, with arrows chaining each hash to the next. Step 2: request 2 arrives with the same first 3 blocks; blocks 0–2 light up green "HIT, refcount 2" and only block 3 is computed fresh; a TTFT bar next to it shrinks by ~75%. Step 3: request 3 has one changed token in block 0; its h₀ differs, the red mismatch propagates through the chain, and all 4 blocks are computed fresh. Step 4: add a "salt: tenant-B" tag to request 2's first hash and show its hits turn red, illustrating isolation.

## When memory runs out: preemption and eviction {#preemption-eviction}

With incremental allocation, you're making a bet: that not everyone will grow to the max at once. Sometimes the bet loses. A decode step needs a new block and the pool is empty. The engine has three moves, and it's important to keep them apart because they cost very different things.

**1. Evict inactive cached blocks (cheap).** Blocks kept around only for possible prefix reuse, with reference count zero, can simply be dropped. vLLM keeps these in a free queue in **LRU** (least-recently-used) order and reclaims from the old end. The only cost is that a future request might have to prefill again. TensorRT-LLM is explicit that "launching new requests take priority over possible reuse", and it can also **offload** evicted reusable blocks to CPU memory instead of discarding them, which pays off only if the GPU–CPU link is fast.

**2. Preempt an active sequence (expensive).** If there's nothing inactive left, the engine pauses a *running* sequence and takes its blocks. You cannot just drop some of a live sequence's history, because full attention needs every past K and V. Silently deleting rows changes the model's output. So preemption is **all-or-nothing** per sequence (that's the paper's policy), and the paused sequence gets its state back one of two ways:

| Recovery | How | Cost |
|---|---|---|
| **Swap** | Copy its blocks to CPU RAM, copy back later | PCIe/interconnect transfer both ways, CPU memory |
| **Recompute** | Discard blocks; later re-run prefill on prompt + tokens generated so far | GPU FLOPs, but prefill is fast and parallel |

Recompute is often cheaper than it sounds, because rebuilding the cache is a *prefill*, which is compute-bound and efficient, while swapping is a slow copy. The PagedAttention paper measured recomputation overhead at no more than 20% of swapping's latency in their setup, and current vLLM docs say its V1 engine defaults to `RECOMPUTE` for that reason. Either way the user sees a stall mid-stream, which shows up as a latency spike in ITL.

**3. Don't admit (the calm option).** The best preemption is the one you avoided. **Admission control** means deciding whether a request may start at all, based on free blocks and a growth allowance. Queue new arrivals (with a deadline) or reject them before memory is so tight that the engine thrashes, preempting and recomputing the same sequences over and over.

```text
 need a new block, pool is empty
        │
        ├─► any refcount-0 cached blocks?  ── yes ─► evict LRU (maybe offload) ─► done
        │                                           cost: future prefill
        no
        ├─► preempt a running sequence (all its blocks)
        │        ├─ swap to CPU   → cost: transfer time
        │        └─ recompute     → cost: redo prefill later
        │   user sees: a mid-stream stall
        │
 and upstream, always: admission control so this path stays rare
```

**Rules of thumb to defend in an interview:**

- **Reference counts decide reclaimability, not "request finished".** A finished request's prefix blocks may still be used by another sequence, or kept on purpose for reuse.
- **A high cache-hit rate isn't automatically good.** If retained prefixes crowd out new requests and cause queueing, you've optimized the wrong number. Judge prefix caching by saved prefill work *and* by queueing, preemptions and latency percentiles.
- **Plain LRU can be flushed.** A burst of one-off giant prompts can push out a hot system prompt. Per-tenant quotas or retention priorities help.
- **Benchmark cold and warm.** A benchmark whose second run hits a shared prefix will look great, and it can hide an admission policy that collapses the day prompts change.
- **Watch preemptions as a metric.** A rising preemption count is your early warning that the KV pool is over-committed. The fixes are more memory for KV (e.g. a higher memory-utilization setting), lower concurrency, shorter output caps, or a smaller cache per token (GQA, KV quantization).

> 🎬 **Animation — three responses to pressure:** a KV pool of 10 blocks, fully occupied: 3 grey "cached, refcount 0" blocks and 7 coloured blocks owned by sequences A, B, C. Step 1: sequence A needs a block; the oldest grey block fades out ("evicted: LRU") and turns into A's colour. Step 2: two more requests need blocks; the other grey blocks go. Step 3: B needs a block and no grey blocks remain; all of C's blocks slide out of the pool into a box labelled "preempted, will recompute", and C's output stream on the right freezes with a stall icon. Step 4: a new request D waits outside the pool behind a gate labelled "admission: queue". Step 5: A finishes, its blocks free up, C is re-prefilled back into the pool, and its stream resumes.

# Interview

## Question

You run a model with 32 layers, 8 KV heads of dimension 128, and a bf16 KV cache on an 80 GiB GPU, leaving 48 GiB for KV. The team set "max concurrent requests = 48" because prompts are about 8,192 tokens and "that's 1 GiB each". They allow up to 2,048 output tokens. Under load, streams stall, p99 inter-token latency explodes, and logs show constant preemptions. Walk through what's happening and what you'd change.

## Answer

Start with bytes per token: 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB. An 8,192-token prompt is exactly 1 GiB, so 48 prompts fill the 48 GiB pool before a single output token is generated. Every decode step then needs a new block that doesn't exist, so the engine preempts running sequences. With recompute-style preemption, a victim's whole prompt gets re-prefilled later, which takes more GPU time and blocks the others, and the cycle repeats. That's the thrashing that shows up as stalls and a p99 ITL blow-up.

The worst-case footprint per sequence is 8,192 + 2,048 = 10,240 tokens × 128 KiB = 1.25 GiB, so a conservative raw bound is floor(48 ÷ 1.25) = 38 sequences, before block tail waste and runtime variance. I'd go a bit lower in practice, then tune using measured behaviour.

Fixes, in order. First, make admission **token- or block-aware** rather than a request count: admit only if free blocks cover the prompt plus a growth allowance. Second, decide on reservation. Either reserve the full output allowance (safe, less utilization) or reserve a partial allowance based on the observed output-length distribution, and accept a *measured, bounded* preemption rate. Third, evict refcount-zero prefix blocks before ever preempting live sequences. Fourth, if a big shared system prompt exists, turn on prefix caching, but don't count on it in the worst-case plan, because a cold start or a template change removes the benefit. Fifth, if memory is still the binding constraint, reduce KV per token (KV quantization, or a GQA/MQA model with fewer KV heads) or add GPUs.

Finally, verify with cold-cache and warm-cache load tests. Look at p95/p99 TTFT and TPOT, queue time, preemption count, and block occupancy, not average request time.

## Follow-ups

- The product wants 4 parallel samples per prompt (n = 4). How do logical and physical KV demand change, and where does copy-on-write come in?
- When is swapping a preempted sequence to CPU cheaper than recomputing it?
- Why does per-token decode latency rise as the batch fills with long contexts, even though weights are read only once per step?
- How would you set up prefix caching in a multi-tenant deployment without creating a timing side channel?
- A teammate proposes caching queries too, "for symmetry". What do you say?

# Pitfalls

- Sizing concurrency from weight memory alone and forgetting that each sequence's KV cache grows every step. The pool can be full at step zero.
- Using query heads instead of KV heads in the size formula for a GQA/MQA model, which overestimates the cache by h ÷ h_kv (4× in the example).
- Saying "the KV cache makes decode O(1)". It removes recomputation, but every step still reads the whole cache, so per-step cost grows with context length.
- Thinking paging approximates attention or shortens context. It only changes where blocks live; the math is identical.
- Assuming a matching suffix or "similar" prompt can reuse cached KV. Only an exact, identical prefix, with the same model and adapter, is valid.
- Treating unreferenced cached blocks and a live sequence's blocks as equally evictable. Dropping live history changes the output; it needs preemption with swap or recompute.
- Reporting only warm-cache throughput or average latency, which hides cold-start TTFT, preemption stalls and p99 behaviour.

# Checklist

- Explain prefill vs decode, and compute arithmetic intensity for batch-1 decode and a long prefill.
- Define TTFT, ITL, TPOT and throughput, and compute TPOT from E2E latency, TTFT and token count.
- Explain what the KV cache stores, why keys and values are reusable under the causal mask, and why queries aren't cached.
- Compute KV bytes per token and per sequence from L, h_kv, d_head and dtype, and check the result against a known model.
- Turn a GPU memory budget into a safe concurrency number that includes output growth.
- Describe block tables, tail waste, reference counts and copy-on-write.
- State the exact conditions for prefix-cache reuse, how chained block hashes enforce them, and how cache salts isolate tenants.
- Distinguish eviction of inactive blocks, preemption (swap vs recompute) and admission control.

# Sources

- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023)](https://arxiv.org/abs/2309.06180) — Prefill vs memory-bound generation, 20.4%–38.2% utilization in prior systems, OPT-13B at ~800 KB/token, block tables, block size 16, reference counts, copy-on-write, all-or-nothing preemption with swap or recompute.
- [vLLM: Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/) — Block hashes built from parent hash, block tokens and extra keys; full-block-only caching; LRU free queue with reference counting; cache_salt.
- [vLLM: Security guide](https://docs.vllm.ai/en/latest/usage/security/) — The TTFT timing side channel on shared prefix caches and per-tenant secret cache salts.
- [vLLM: Metrics design](https://docs.vllm.ai/en/latest/design/metrics/) — Definitions of TTFT, inter-token latency, TPOT and end-to-end latency, plus prefix-cache hit counters.
- [vLLM: Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) — V1 defaults to RECOMPUTE preemption; gpu_memory_utilization as a lever against frequent preemption.
- [TensorRT-LLM: KV cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html) — Full-block-only reuse, default 128-token blocks, LRU eviction with new requests prioritized over reuse, offloading to host memory.

# Flashcards

## phases

**Q:** What are prefill and decode, and why do they stress the GPU differently?

Prefill processes the whole known prompt in one parallel pass (matrix × matrix), writes K and V for every prompt token at every layer, and produces the first output token. It does many FLOPs per byte of weights read, so it's usually compute-bound. Decode produces one token per step per sequence (matrix × vector). It reads all the weights to do very little math per sequence, so it's usually memory-bandwidth-bound. They also map to different metrics: prefill drives TTFT, decode drives TPOT/ITL.

## arithmetic-intensity

**Q:** Why is batch-1 decode bandwidth-bound? Give the back-of-envelope.

An N-parameter model costs ~2N FLOPs per token and, in bf16, must read 2N bytes of weights per step. So intensity is ~1 FLOP/byte, far below a GPU's ridge point of hundreds of FLOPs/byte. Time is set by bytes ÷ bandwidth (e.g. 16 GB ÷ 3 TB/s ≈ 5.3 ms/token for 8B params, illustrative numbers). Batching B sequences reuses the same weight reads, raising intensity to ~B, which is why decode batching is nearly free until KV reads or compute take over.

## ttft-tpot

**Q:** Define TTFT and TPOT, and compute TPOT for E2E = 5.3 s, TTFT = 0.3 s, 200 output tokens.

TTFT is arrival to first token: queueing + prefill. TPOT is the average gap between later tokens: (E2E − TTFT) ÷ (tokens − 1) = 5.0 ÷ 199 ≈ 25.1 ms. The −1 is because the first token is counted in TTFT. High TTFT points at queueing or long prompts; high TPOT points at a heavy decode batch or interference.

## no-query-cache

**Q:** Why does the KV cache store keys and values but not queries?

Under the causal mask, a token's K and V at a layer depend only on earlier tokens and fixed weights, so they never change and every future token reads them. A query is used only once, by its own token, at the step that token is processed; no later token ever looks at it. Caching Q would use memory for nothing.

## formula

**Q:** What's the KV cache size formula?

Bytes per token = 2 × L × h_kv × d_head × s (the 2 is K and V; L layers; h_kv KV heads; d_head per-head width; s bytes per element). Total = bytes per token × the sum of cached tokens across all live sequences. Cost scales with resident tokens, not request count. Adjust for sliding-window layers, quantization metadata and multi-GPU layout.

## gqa

**Q:** Which head count goes into the KV formula for a GQA or MQA model?

The KV head count h_kv, because several query heads share one stored K/V head. Using the query head count overestimates the cache by h ÷ h_kv (e.g. 32 ÷ 8 = 4×). Why GQA/MQA work and what they trade off is covered in **Making attention cheaper: GQA, FlashAttention, and long context**.

## example

**Q:** 32 layers, 8 KV heads, d_head 128, bf16: how much KV per token and per 8,192-token sequence?

2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token. Times 8,192 tokens = 1,073,741,824 bytes = exactly 1 GiB per sequence. With full MHA (32 KV heads) it would be 4 GiB.

## growth

**Q:** Why do 48 one-GiB prompts fail in a 48 GiB KV pool, and what's a safe count with 2,048 output tokens allowed?

The prompts alone fill the pool, so the very first generated token has nowhere to go and the engine starts preempting. Worst-case per sequence is 10,240 tokens × 128 KiB = 1.25 GiB, so floor(48 ÷ 1.25) = 38 sequences, before block waste and runtime overhead.

## paging

**Q:** What does PagedAttention's block table do?

It maps a sequence's logical KV blocks (0, 1, 2…) to physical fixed-size blocks anywhere in the GPU pool, like OS virtual memory pages. Sequences grow one block at a time instead of reserving a contiguous max-length slab, which removes reservation waste and external fragmentation. The attention math is unchanged; the kernel just follows the table.

## fragmentation

**Q:** Does paging eliminate all KV memory waste?

No. Each sequence can waste up to one partially filled tail block. E.g. 33 tokens with 16-token blocks use 3 blocks (48 slots), wasting 15 slots = 1.875 MiB at 128 KiB/token. Smaller blocks waste less but cost more table lookups and less efficient kernels.

## references

**Q:** Why might a finished request's blocks not be freed?

Blocks are reference-counted. A shared prefix block may still be referenced by other live sequences, and even at refcount zero the engine may keep it for future prefix-cache hits until it's evicted. "Request finished" doesn't equal "memory reclaimable".

## identity

**Q:** When can cached KV blocks be reused for a new request?

Only when the block's tokens and every token before it are identical, and the model, adapter and other inputs (e.g. images) match, because each K/V depends on the entire preceding history. vLLM enforces this by hashing each full block together with its parent's hash plus extra keys. A matching suffix after a changed earlier token can't be reused, and a timestamp at the start of the prompt breaks reuse for everything after it.

## isolation

**Q:** What's the security risk of a shared prefix cache, and the mitigation?

A cache hit lowers TTFT, so an attacker can guess a prefix, time the response, and infer whether another user recently sent it. The mitigation is a secret cache salt mixed into the first block's hash and scoped to a trust boundary (e.g. per tenant), set server-side from authenticated identity.

## eviction

**Q:** How does evicting cached blocks differ from preempting a sequence?

Evicting refcount-zero blocks just discards reusable state; the only cost is a possible future prefill. Preempting takes all the blocks of a running sequence, which must then be swapped to CPU or recomputed via prefill, and the user sees a stall. You can't drop part of a live sequence's history, because full attention needs all of it.

## metrics

**Q:** Why isn't a high prefix-cache hit rate enough to call a serving policy good?

Retained prefixes occupy memory that active requests might need, so a high hit rate can coexist with queueing and preemptions. Judge it by saved prefill work together with queue time, preemption count, block occupancy, throughput and TTFT/TPOT percentiles, on both cold and warm traffic.
