---
{
  "slug": "efficient-attention",
  "title": "Making attention cheaper: GQA, FlashAttention, and long context",
  "category": "architecture",
  "summary": "Attention costs you three different things: arithmetic, memory traffic, and KV-cache bytes. Learn which lever (GQA, FlashAttention, RoPE scaling, sliding windows) pulls on which cost, do the sizing arithmetic, and defend the tradeoffs.",
  "difficulty": "Advanced",
  "minutes": 30,
  "prerequisites": ["serving-kv-cache", "transformer-foundations"],
  "learningObjectives": [
    "Separate attention's three costs (FLOPs, memory traffic, KV-cache size) and say which one binds in prefill, decode, and training.",
    "Derive MHA, MQA, and GQA tensor shapes and calculate KV-cache bytes and decode bandwidth for a concrete model.",
    "Explain how FlashAttention's tiling and online softmax compute exact attention with far less memory traffic, and why it is still quadratic.",
    "Explain how RoPE encodes relative position, how context extension (interpolation, YaRN) works, and why a bigger context number does not prove usable long context.",
    "Choose between dense, sliding-window, and sparse attention and defend the choice with a falsifiable evaluation."
  ]
}
---

# Sections

## Attention sends you three different bills {#three-costs}

Quick recap from **What happens at inference: prefill, decode, and the KV cache**. When a model answers you, it runs two phases. **Prefill** reads your whole prompt in one big parallel pass. It's compute-bound: the GPU's math units are busy. **Decode** then produces the answer one token at a time. Each step does very little math but has to read the model's weights and the **KV cache** from memory, so it's memory-bandwidth-bound. The KV cache is where we keep the key and value vectors of every token seen so far, for every layer, so a new token only computes its own query, key and value and then attends over the stored history instead of recomputing it.

Attention is the part of the transformer whose cost grows with context length, so it's the natural place to look for savings. The trap is that "efficient attention" isn't one thing. It's at least three different bills, and each trick pays down a different one:

| Bill | What it is | When it hurts | Lever in this article |
|---|---|---|---|
| **FLOPs** | Multiply-adds for `QKᵀ` and `P·V` | Long prefill, training | Sliding-window / sparse attention |
| **Memory traffic** | Moving the big `T × T` score table in and out of GPU memory | Prefill and training with a naive kernel | FlashAttention |
| **KV-cache size** | Bytes of stored K and V, which you re-read every decode step | Decode, concurrency | MQA / GQA, sliding windows, KV quantization |

Think of a restaurant kitchen. FLOPs are how much chopping the chefs do. Memory traffic is how many trips they make to the walk-in fridge. The KV cache is how much counter space each table's order takes up while it's still being cooked. Faster chefs don't free up counter space, and fewer fridge trips don't reduce the chopping. Keep the three separate on the whiteboard and half the interview is already won.

Here's where each bill comes from, using the shared notation: `B` sequences, `T` tokens, `h` query heads of width `d_head`, and `h_kv` key/value heads.

```text
 Q : [B, T_q, h,    d_head]     one query per position per head
 K : [B, T_k, h_kv, d_head]     stored keys  (the KV cache)
 V : [B, T_k, h_kv, d_head]     stored values
 S = QKᵀ : [B, h, T_q, T_k]     one score per (query, key) pair per head
                     ▲    ▲
          prefill:   T    T    →  T² scores per head  (quadratic)
          decode:    1    T    →  T scores per head   (linear per step)
```

In prefill, `T_q = T_k = T`, so scores grow as T². In decode, `T_q = 1` because only the newest token asks a question, and `T_k` is the whole history. Each decode step is linear in history, but you repeat it once per generated token.

How much does the quadratic part actually matter? Take a 7B-class model with `d = 4096` and `L = 32` layers. Per layer, the two attention matmuls cost about `4 · T² · d` FLOPs, halved by the causal mask. The weight matmuls cost about `2 · N` FLOPs per token (`N` = parameter count).

| Prompt length | Attention FLOPs (causal) | Weight-matmul FLOPs | Attention share |
|---|---|---|---|
| 8,192 tokens | 1.76 × 10¹³ | 1.15 × 10¹⁴ | ~13% |
| 131,072 tokens | 4.50 × 10¹⁵ | 1.84 × 10¹⁵ | ~71% |

These are back-of-envelope counts, not measurements. At 8k tokens attention is a minor cost. At 128k it dominates. That's why long context is where all of this starts to matter.

> 🎬 **Animation — three bills:** a single transformer layer drawn as a box. Three meters sit beside it labelled "FLOPs", "HBM trips" and "KV bytes". Step 1: a 4-token prompt enters (prefill) and a 4 × 4 score grid lights up; the FLOPs and HBM-trip meters rise and the KV meter adds 4 slots. Step 2: decode one token; a 1 × 5 score row lights up; FLOPs barely move, the KV meter adds 1 slot, and an arrow shows the whole KV stack being read. Step 3: grow the prompt to 8 tokens; the grid becomes 8 × 8 (4× the cells) while the KV meter only doubles. The caption reads "different tricks shrink different meters".

## Sharing keys and values across heads: MHA, MQA, GQA {#head-sharing}

Recall from **Attention from scratch: how tokens talk to each other** that attention is split into `h` heads. Each head is a smaller attention running side by side, with its own view of the tokens. In classic **multi-head attention (MHA)**, every query head has its own key head and value head. So the cache stores `h` keys and `h` values per token per layer.

Here's the analogy. Query heads are students and KV heads are textbooks. MHA buys every student their own copy. **Multi-query attention (MQA)**, from Shazeer's 2019 paper, makes the whole class share one copy: a single K head and a single V head serve all the query heads. **Grouped-query attention (GQA)** puts the students in small groups that share a copy each: `h_kv` KV heads, with `h / h_kv` query heads per group. Crucially, every student still writes their own answer. Each query head still computes its own attention pattern and its own output. Only the thing being looked up is shared.

```text
 query heads:   q0  q1  q2  q3  q4  q5  q6  q7
                │   │   │   │   │   │   │   │
 MHA (h_kv=8)  kv0 kv1 kv2 kv3 kv4 kv5 kv6 kv7    8 KV heads cached

 GQA (h_kv=2)   └───────┬──────┘  └──────┬───────┘
                       kv0              kv1       2 KV heads cached

 MQA (h_kv=1)   └───────────────┬────────────────┘
                               kv0                1 KV head cached
```

A concrete model makes this real. Mistral 7B's published config has `d = 4096`, `h = 32` query heads of `d_head = 128`, and `h_kv = 8`, so each KV head serves 32 / 8 = 4 query heads. Its shapes, ignoring biases:

```text
 W_Q : [4096, 4096]   → Q : [B, T, 32, 128]
 W_K : [4096, 1024]   → K : [B, T,  8, 128]    1024 = 8 × 128
 W_V : [4096, 1024]   → V : [B, T,  8, 128]
 attention output     :     [B, T, 32, 128] → concat → [B, T, 4096] → W_O
```

The output is still 32 heads wide. Sharing K and V never merges query outputs.

```formula
group size = h / h_kv        KV bytes ∝ h_kv   (not h)
```

Here `h` is the number of query heads and `h_kv` the number of KV heads. GQA assumes `h` divides evenly by `h_kv`. MHA is the special case `h_kv = h`, and MQA is `h_kv = 1`.

Why not always use MQA? Quality. One shared KV head gives every query head the same keys to compare against, which squeezes the model's capacity to track different kinds of relationships. The GQA paper (Ainslie et al., 2023) found GQA reaches quality close to MHA with speed close to MQA. They also showed you can **uptrain** an existing MHA checkpoint into GQA: mean-pool each group's K and V projection matrices into one, then continue pretraining for about 5% of the original compute. Note what that implies. GQA is a property of the trained weights, not a runtime flag. You can't flip `num_key_value_heads` in a config file of an MHA model and expect sane output.

One implementation gotcha that interviewers love: the kernel should *index* the shared KV head for each group, not physically copy it out to 32 heads. A `repeat_interleave` is fine as a slow reference implementation, but if it materializes a 32-head K and V in memory for every step, you've given back the savings you trained for.

> 🎬 **Animation — students and textbooks:** 8 query-head circles (q0–q7) in a row over a shelf of KV "books". Step 1 (MHA): 8 books, one arrow from each circle to its own book; a counter shows "8 KV heads cached". Step 2 (GQA): books merge into 2; q0–q3 draw arrows to book 0 and q4–q7 to book 1; the counter drops to 2. Step 3 (MQA): one book, all 8 arrows converge; the counter shows 1. Throughout, 8 separate output boxes stay under the circles, to show outputs never merge.

## Doing the KV arithmetic {#kv-arithmetic}

Now turn head counts into bytes, because bytes decide how many users fit on a GPU. For every layer and every token, you store one key and one value vector per KV head:

```formula
KV bytes = 2 × L × h_kv × d_head × s × (total retained tokens)
```

`2` counts K and V. `L` is layers, `h_kv` KV heads, `d_head` head width, and `s` bytes per element (2 for bf16). "Total retained tokens" is the sum over all live sequences of how many tokens each is actually holding, which is `B × T` if they're all the same length. Query heads don't appear. The most common interview slip is plugging in `h` instead of `h_kv`.

**Per-token cost** for the Mistral-7B-like shape (`L = 32`, `d_head = 128`, bf16):

| Variant | h_kv | Bytes per token | Per 8,192-token sequence | 8 sequences × 8,192 tokens |
|---|---|---|---|---|
| MHA | 32 | 2·32·32·128·2 = 524,288 (512 KiB) | 4 GiB | 32 GiB |
| GQA | 8 | 131,072 (128 KiB) | 1 GiB | 8 GiB |
| MQA | 1 | 16,384 (16 KiB) | 128 MiB | 1 GiB |

Check the middle row: 2 × 32 × 8 × 128 × 2 = 131,072 bytes. Times 65,536 tokens (8 × 8,192) gives 8,589,934,592 bytes = exactly 8 GiB (1 GiB = 2³⁰ bytes).

**Capacity.** Say that after weights (~14 GB for 7B parameters in bf16), activations workspace, and framework overhead, you have 12 GiB left for cache. With GQA that ideally fits 12 of those 8k-token sequences. With MHA it fits 3. That's a 4× difference in concurrency, which translates directly into cost per token.

**Bandwidth.** Decode reads the weights plus every live sequence's cache on every step. As an illustrative lower bound, assume 2 TB/s of memory bandwidth and a batch of 8 sequences at 8k tokens:

```text
 per decode step  =  weights  +  KV cache          ÷ 2 TB/s
 MHA :  14.0 GB  +  34.4 GB  = 48.4 GB   →  ~24 ms
 GQA :  14.0 GB  +   8.6 GB  = 22.6 GB   →  ~11 ms
 MQA :  14.0 GB  +   1.1 GB  = 15.1 GB   →  ~7.5 ms
```

These are bandwidth floors under simple assumptions, not benchmarks. But notice the shape. With MHA at long context, the cache outweighs the model. GQA roughly halves the step time here. Going further to MQA helps much less, because weights now dominate. So cache reduction buys you less when contexts are short or batches are small.

**Three things the formula doesn't tell you:**

1. **Allocator reality.** The formula counts logical payload. Real servers allocate the cache in fixed-size blocks (PagedAttention, from the serving article), so you also pay for partially filled blocks, metadata, and room for outputs that haven't been generated yet. A request admitted at 8,192 tokens might grow to 10,000. Admission control has to reserve for growth, which we'll go through in detail in **Serving many users: batching, scheduling, and speculative decoding**.
2. **Tensor-parallel replication.** Tensor parallelism splits a layer's heads across GPUs, which we'll go through properly in **Distributed training: fitting a training run onto a cluster**. If you have 8 KV heads and split across 16 GPUs, each KV head has to live on 2 GPUs. The logical cache is unchanged, but the physical bytes double. The GQA paper points out that MQA wastes memory exactly this way under sharding.
3. **Other byte-shavers stack.** Storing K and V in 8-bit halves `s`, at some quality risk (covered in **Quantization: spending fewer bits per weight**). Prefix caching shares identical prompt prefixes, but only count it if your allocator really deduplicates them.

> 🎬 **Animation — filling a GPU:** an 80 GB bar split into "weights 14 GB", "workspace", and a "KV budget 12 GiB" region. Step 1 (MHA): 8k-token request blocks, each 4 GiB wide, drop in; only 3 fit and a 4th bounces off in red. Step 2 (switch to GQA): each block shrinks to 1 GiB and 12 fit. Step 3: one block grows as it generates tokens and pushes into a hatched "growth reserve" strip, showing why you don't fill to 100%.

## Why a naive attention kernel is slow even when the math is fine {#memory-wall}

Now the second bill, memory traffic. A GPU has two kinds of memory that matter here:

- **HBM** (high-bandwidth memory): the tens of gigabytes you think of as "GPU memory". Big, but every read or write costs time.
- **SRAM**: tiny on-chip scratchpads next to the compute units, measured in kilobytes to low megabytes per compute block, and much faster.

The analogy is a warehouse and a desk. The warehouse (HBM) holds everything. The desk (SRAM) holds a few pages. Work gets done at the desk. The question is how many trips you make.

A textbook attention implementation does this, per head:

```text
 1. read Q, K from HBM  → compute S = QKᵀ / √d_head  → WRITE S (T×T) to HBM
 2. read S               → softmax                    → WRITE P (T×T) to HBM
 3. read P, V            → O = P·V                    → write O (T×d_head)
```

That `T × T` table gets written and read twice. How big is it? At `T = 8,192` in bf16, one head's score table is 8,192² × 2 bytes = 128 MiB. With 32 heads that's 4 GiB per layer for a single sequence, just for scratch. Meanwhile the actual inputs (Q, K, V for one head) are only 8,192 × 128 × 2 bytes = 2 MiB each. So the scratch table is 64× bigger than any one input, and it makes four trips (write S, read S, write P, read P). The chefs are fast. They're just walking to the fridge all day.

The operations in steps 1–2 (scaling, masking, softmax) have very little math per byte, so they run at the speed of memory, not compute. Profilers show attention as "memory-bound" even though the matmuls themselves are fine. The fix isn't fewer FLOPs. It's fewer trips.

## FlashAttention: tiles and a running softmax {#flashattention}

**FlashAttention** (Dao et al., 2022) never writes the `T × T` table to HBM at all. It **tiles** the computation: load a block of queries onto the desk, stream blocks of K and V past it one at a time, and keep a running result for each query row. Only the final output `O` goes back to the warehouse.

```text
            K,V blocks stream past →
          ┌──────┬──────┬──────┬──────┐
 Q block  │ tile │ tile │ tile │ tile │   each tile: scores, exp, accumulate
 (in SRAM)│  1   │  2   │  3   │  4   │   all in SRAM, nothing T×T hits HBM
          └──────┴──────┴──────┴──────┘
                                        → write O once
```

The obstacle is softmax. Each row's softmax divides by a sum over *every* key in the row, so it looks like you need the whole row before you can normalize anything. The trick is **online softmax**: keep three running numbers per query row and fix up the past whenever the future surprises you.

- `m`: the largest score seen so far (subtracted before `exp` for numerical safety)
- `l`: the running sum of `exp(score − m)`
- `u`: the running sum of `exp(score − m) · value`, the not-yet-normalized output

When a new block arrives with its own max `m_b`, sum `l_b` and weighted sum `u_b`:

```formula
m′ = max(m, m_b)
l′ = e^(m − m′) · l  +  e^(m_b − m′) · l_b
u′ = e^(m − m′) · u  +  e^(m_b − m′) · u_b
final output O = u / l
```

The `e^(m − m′)` factor rescales old sums that were computed relative to an out-of-date max. It's like re-basing a running total when the exchange rate changes.

**Worked example.** One query row sees four keys with scores `[1, 3, 2, 4]` and (scalar, for simplicity) values `[10, 20, 30, 40]`. Tiles hold two keys each.

```text
 Tile 1 (scores 1,3; values 10,20)
   m1 = 3
   l1 = e^(1−3) + e^(3−3)          = 0.1353 + 1       = 1.1353
   u1 = 0.1353·10 + 1·20           =                    21.353
 Tile 2 (scores 2,4; values 30,40)
   m2 = 4
   l2 = e^(2−4) + e^(4−4)          = 0.1353 + 1       = 1.1353
   u2 = 0.1353·30 + 1·40           =                    44.060
 Merge: m′ = 4, rescale tile 1 by e^(3−4) = 0.3679
   l′ = 0.3679·1.1353 + 1.1353     = 1.5530
   u′ = 0.3679·21.353 + 44.060     = 51.916
   O  = 51.916 / 1.5530            = 33.43
```

Check against a full softmax: `exp(score − 4)` = `[0.0498, 0.3679, 0.1353, 1]`, which sums to 1.5530. The weighted values sum to 0.498 + 7.358 + 4.060 + 40 = 51.916, giving 33.43. It matches. Now the wrong way: normalize each tile separately (18.81 and 38.81) and average them, and you get 28.81. That's wrong, because the two tiles had different denominators.

> 🎬 **Animation — online softmax on one row:** a row of four score cells [1, 3, 2, 4] with value chips [10, 20, 30, 40] beneath. Step 1: tile 1 (first two cells) slides onto a "desk" panel; show m=3, l=1.1353, u=21.353. Step 2: tile 2 slides in; its local stats m=4, l=1.1353, u=44.060 appear. Step 3: the old l and u shrink by a ×0.3679 "rescale" arrow and add to tile 2's; show l′=1.5530, u′=51.916. Step 4: O = 33.43 appears with a green check next to a "full softmax = 33.43" box. Step 5: a red ghost path shows averaging 18.81 and 38.81 into 28.81 with a cross.

**Training.** The backward pass needs the attention probabilities, and naive training stores the whole `T × T` table for that. FlashAttention stores only `O` and the per-row statistics, then *recomputes* the scores tile by tile during backward. More FLOPs, fewer bytes, less wall-clock time. The paper reports speedups such as about 3× on GPT-2 training at 1K context, and memory for attention that grows linearly with `T` instead of quadratically. That's what made long-context training practical.

**What FlashAttention is not:**

- **Not approximate.** It computes the same masked softmax attention. Results can differ in the last bits because the additions happen in a different order, so it's "exact" rather than "bit-identical".
- **Not linear-time.** Every query still meets every allowed key. The FLOPs are still quadratic in `T`. Only the memory traffic and scratch storage changed.
- **Not a KV-cache fix.** The persistent cache is exactly the same size. FlashAttention and GQA are complementary, not alternatives.

The follow-ups got faster on newer hardware without changing the math. FlashAttention-2 reworked how work is split across GPU threads for roughly 2× over v1, reaching 50–73% of the A100's theoretical peak. FlashAttention-3 targets H100s with asynchronous data movement and FP8 support. In practice you'll usually get these kernels through your framework's fused attention call rather than writing them yourself.

> 🎬 **Animation — warehouse vs desk:** left: a big "HBM" warehouse; right: a small "SRAM" desk. Step 1 (naive): an 8 × 8 score grid is built on the desk, then trucked whole to the warehouse, back to the desk for softmax, back to the warehouse, and back again for P·V; a trip counter reaches 4 big trucks. Step 2 (Flash): a 2-row Q block sits on the desk while 2-column K/V blocks arrive one at a time; each is used and discarded; only the final 2-row O strip is trucked back. The trip counter shows small trucks only, and the FLOP counter is identical in both runs.

## How RoPE puts position into attention {#rope}

Now the long-context part. Attention by itself is order-blind: shuffle the tokens and the scores don't care. **The transformer block: assembling the full model** introduced **RoPE** (rotary position embeddings, from the RoFormer paper) as the way most modern decoders fix this. Here's the mechanism in enough detail to reason about extending it.

Split each query and key vector into pairs of coordinates. Treat each pair as a little 2-D arrow. At position `p`, rotate every pair by an angle `p · θ_i`, where each pair `i` has its own speed `θ_i`. Then take the dot product as usual.

The magic is that the dot product of two rotated arrows only depends on the *difference* of their angles. Picture two clock hands. Advance both by an hour and the angle between them is unchanged, and that angle is what the score sees.

**Tiny example.** One pair, `q = k = (1, 0)`, speed `θ = 30°` per position.

```text
 query at p=5  → rotated 150°      key at p=3 → rotated 90°
 angle between = 60°  → score = cos 60° = 0.5

 shift both by 10 positions:
 query at p=15 → 450° ≡ 90°        key at p=13 → 390° ≡ 30°
 angle between = 60°  → score = 0.5   (same! only p − t matters)
```

```formula
q̃_p = R(p·θ) q     k̃_t = R(t·θ) k     q̃_pᵀ k̃_t = qᵀ R((t − p)·θ) k
```

`R(α)` is a 2-D rotation by angle `α`, applied independently to each coordinate pair with its own `θ_i`. The speeds are usually `θ_i = base^(−2i / d_head)` with `base` typically 10,000. So the first pairs spin fast and resolve nearby order, and the last pairs spin very slowly and distinguish far-apart positions, like the second, minute and hour hands of a clock.

Two consequences to keep straight. First, RoPE changes only *how position is represented*. It stores no extra tokens and removes no attention pairs, so it touches none of the three bills. Second, the formula happily accepts position 100,000 even if the model only ever saw positions up to 4,096. The math is defined out there. Whether the model *behaves* is a separate question.

> 🎬 **Animation — clock hands:** a unit circle with a blue query arrow and an orange key arrow, both starting at 0°. Step 1: rotate the query to 150° (p=5) and the key to 90° (p=3); shade the 60° gap and show "score = 0.5". Step 2: advance both by 300° (10 positions × 30°); the arrows land at 90° and 30°, the shaded 60° gap is unchanged, and the score stays 0.5. Step 3: beside it, three small clocks labelled "fast pair", "medium pair", "slow pair" tick at different speeds for the same position change.

## Stretching the context window, and why a bigger number isn't proof {#context-extension}

Suppose a model was trained on 2,048-token sequences and you want 8,192. Just feeding longer inputs (**extrapolation**) means the fast-and-slow clock hands land at angle combinations the model never saw. Chen et al. (2023) showed that extrapolation can produce catastrophically large attention scores that wreck the model.

**Position interpolation (PI)** does the opposite. It squeezes the new range into the old one by dividing positions by the scale factor `s = new length / trained length`. With `s = 4`, token 6,000 is rotated as if it were at position 1,500. Every angle is now one the model has seen. The cost is that neighbouring tokens are only a quarter of a position apart, so fine-grained order is blurrier. The PI paper extended LLaMA models up to 32,768 tokens with a short fine-tune (within 1,000 steps). The fine-tune is part of the recipe, not optional.

```text
 trained range     |0 ─────────────── 2048|
 extrapolate       |0 ─────────────── 2048 ─ ─ ─ ─ ─ ─ 8192|  unseen angles ✗
 interpolate (÷4)  |0 ─── 8192 squeezed into ─── 2048|          seen angles, finer spacing
```

Later recipes such as **YaRN** scale the pairs *unevenly*. They leave the fast pairs (local word order) mostly alone and stretch the slow pairs (long-range position), and they report reaching long contexts with about 10× fewer tokens and 2.5× fewer steps of fine-tuning than earlier methods. Many serving stacks expose a "rope scaling" setting. It only helps if the checkpoint was trained or fine-tuned for that exact scaling. Otherwise you've just changed the geometry under the model's feet.

**Fits vs works.** A model that *accepts* 128k tokens and a model that *uses* 128k tokens are different claims. "Lost in the Middle" (Liu et al., 2023) found a U-shaped curve: models retrieved relevant facts best when they sat at the start or end of the context, and noticeably worse when they sat in the middle. And note that extension also doesn't shrink any bill. The KV cache at 128k is 16× the cache at 8k, and prefill attention FLOPs are 256× larger.

How would you validate an extension? Build an evaluation grid:

| Axis | Values |
|---|---|
| Context length | original length, 2×, 4×, target |
| Evidence position | start, middle, end |
| Task type | exact retrieval, multi-hop (two facts combined), with distractors |
| Regression guard | original short-context benchmarks |

Also check the plumbing. Position IDs, scaling settings and masks must match between prefill and decode, and a prefix cache built under one RoPE setting is garbage under another. A successful memory allocation proves capacity. Stable accuracy across that grid proves usable context.

> 🎬 **Animation — squeeze vs stretch:** a ruler marked 0–2048 in green ("trained"). Step 1 (extrapolate): the ruler extends to 8192 and the new region glows red with a spike icon labelled "unseen angles". Step 2 (interpolate): the 8192 ruler is compressed like an accordion to fit 0–2048; tick spacing becomes 4× denser, with a label "position 6000 → 1500". Step 3: a heatmap grid with context length on the x-axis and evidence position (start/middle/end) on the y-axis fills in, showing a darker middle band to illustrate "lost in the middle" (illustrative shading, not data).

## Sliding windows and sparse attention: seeing fewer tokens on purpose {#sliding-window}

The last lever cuts the pairs themselves. **Sliding-window attention** lets each token attend only to the last `W` tokens, including itself. It's like reading a long book while only being able to glance back at the last few pages.

```text
 causal, full (T=6)          sliding window W=3
 pos  0 1 2 3 4 5            pos  0 1 2 3 4 5
  0   ■                       0   ■
  1   ■ ■                     1   ■ ■
  2   ■ ■ ■                   2   ■ ■ ■
  3   ■ ■ ■ ■                 3     ■ ■ ■
  4   ■ ■ ■ ■ ■               4       ■ ■ ■
  5   ■ ■ ■ ■ ■ ■             5         ■ ■ ■
  21 pairs  (~T²/2)           15 pairs  (≤ T·W)
```

```formula
allowed keys for query t = { j : max(0, t − W + 1) ≤ j ≤ t }       total pairs ≤ T × W
```

`t` is the query position, `j` a key position, and `W` the window size including the current token. Work is now linear in `T` for a fixed `W`.

The big win is in the cache. If every layer uses the window, keys older than `W` can never be read again, so a **rolling buffer** can overwrite them: store token `i` in slot `i mod W`. Mistral 7B does exactly this with `W = 4,096`. The paper reports an 8× cache reduction at 32k tokens without quality loss. Checking with our per-token number: 32,768 × 128 KiB = 4 GiB per sequence, versus 4,096 × 128 KiB = 512 MiB. That's 8×.

**Reach through layers.** Information can still travel further than `W`, because layer 2 attends to tokens that already looked back `W` in layer 1. With 32 layers and `W = 4,096`, Mistral's paper cites a theoretical span of about 131k tokens (32 × 4,096). But "theoretical" is doing heavy lifting. Old information has to survive being compressed into hidden states layer after layer, like a message passed down a line of people. The gist gets through, but an exact account number from 100k tokens ago probably won't.

**Sparse patterns** generalize this idea. **Longformer** combined local windows with a few **global** tokens that attend to, and are attended by, everything. They act as shortcuts, like a table of contents. Its pattern was designed for encoders that look both directions, so don't copy it blindly into a causal decoder, where nothing may look at the future. Other patterns use dilated windows (skipping every few positions) or block patterns. Many recent models mix layers, with most local and a few full, to get most of the savings while keeping some direct long-range access. Global layers still need a full-length cache, so the rolling-buffer savings apply only to the local ones.

The engineering fine print:

- **Fewer pairs doesn't mean faster.** Irregular sparsity can defeat GPU tiling, so 50% fewer edges may give 0% speedup. Measure the kernel.
- **It's an architecture choice, not a serving hack.** Evicting old cache entries from a model trained with dense attention changes its behaviour. It's no longer the same model, even if it stopped running out of memory.
- **Test the distance you care about.** Place critical evidence beyond `W` and check whether answers survive.

> 🎬 **Animation — the window slides:** a 10 × 10 causal mask grid (lower triangle lit). Step 1: dim everything outside a width-3 diagonal band, and show the pair counter drop from 55 to 27. Step 2: beneath it, a ring buffer with 3 slots; tokens 0, 1, 2 fill it, token 3 overwrites slot 0, and token 4 overwrites slot 1, with a label "slot = i mod 3". Step 3: add one global column (token 0 visible to all rows) lit in gold, and show its cache slot pinned outside the ring. Step 4: a three-layer stack where a highlighted path hops back 3 tokens per layer, reaching 9 tokens back, with the signal fading along the way.

## Putting the levers together, and defending the choice {#combining}

These techniques compose because they hit different bills:

| Lever | Bill it cuts | Changes model behaviour? | Needs retraining? |
|---|---|---|---|
| GQA / MQA | KV bytes (and decode bandwidth) | Yes, a different architecture | Yes (or uptraining) |
| FlashAttention | Memory traffic, scratch memory | No (exact, up to rounding) | No |
| RoPE scaling (PI, YaRN) | None, it enables length | Yes | Usually a fine-tune |
| Sliding window / sparse | FLOPs and KV bytes | Yes | Yes |
| KV quantization | KV bytes | Slightly | Sometimes calibration |

A typical modern decoder uses GQA for the cache, a FlashAttention-style kernel everywhere, a RoPE base or scaling chosen for its target context, and maybe some local layers. When you describe that stack, be precise about which parts are "free" and which are checkpoint properties that need quality evidence. Calling the whole thing "lossless" is a red flag to interviewers.

To defend a design, make it falsifiable:

1. **State the binding bill.** Is this workload long-prompt prefill (FLOPs and traffic), high-concurrency chat (KV bytes), or training (activation memory)?
2. **Do the arithmetic** with real `h_kv`, dtype, retained lengths and TP layout, plus headroom for growth.
3. **Benchmark realistic distributions**, not one prompt length. Report TTFT (time to first token), inter-token latency, throughput, peak memory, and failure rate across concurrency levels. Exclude warm-up and compile time.
4. **Check numerics** against a simple reference attention, with causal masks and unequal query/key lengths.
5. **Say what would make you reject it**, e.g. "if middle-position retrieval at 32k drops more than X points versus the 8k baseline".

And resist the tempting inference that a 4× smaller cache means 4× faster. We saw above that with weights dominating, it can be much less. What a smaller cache reliably buys you is *room*: more concurrent sequences, which the scheduler can turn into throughput.

# Interview

## Question

You're serving a 32-layer decoder with 32 query heads of width 128 in bf16. Traffic is 8 concurrent requests that each hold about 8,192 tokens of context, and you have 12 GiB of GPU memory left for the KV cache. The current checkpoint uses standard multi-head attention and you're running out of memory. Walk me through what's happening and what you'd do, without promising speed or quality you haven't tested.

## Answer

First the arithmetic. Per token, the cache stores K and V for every layer and every KV head: 2 × 32 layers × 32 heads × 128 × 2 bytes = 512 KiB. Eight requests at 8,192 tokens is 65,536 tokens, so 32 GiB. That's nearly 3× the 12 GiB budget, so only about 3 requests fit, before even counting block overhead and output growth.

The biggest structural fix is a checkpoint with fewer KV heads. With GQA at 8 KV heads, the per-token cost drops to 128 KiB and the workload needs 8 GiB, leaving 4 GiB for allocator overhead and generated tokens. I'd use a model that was trained or uptrained with GQA. I wouldn't just edit the head count in the config of the MHA model, because sharing is baked into the weights. The GQA paper's uptraining recipe (mean-pool KV projections, then about 5% more pretraining) is an option if we must keep this model family, but it's a new model that needs a full eval.

Independently, I'd make sure we're on a FlashAttention-style fused kernel. It doesn't shrink the cache, but it removes the `T × T` scratch traffic in prefill and keeps attention memory linear in `T`. I'd also check the GQA kernel indexes shared KV heads rather than expanding them.

Then I'd measure. Prefill and decode separately, TTFT and inter-token latency across concurrency, peak memory, and task quality against the MHA baseline. I'd expect decode to get faster, because each step reads less cache, but weights are about 14 GB per step, so the gain is well under 4×.

If contexts keep growing, next options are KV-cache quantization, admission limits, or a model trained with sliding-window layers. Each needs its own quality check. What I wouldn't do is treat RoPE scaling as a memory fix. It changes the positions the model can handle, not the bytes it stores.

## Follow-ups

- With tensor parallelism across 16 GPUs and 8 KV heads, what is the per-GPU cache size, and why?
- If most requests share a 2,000-token system prompt, how does prefix caching change the calculation, and what must the allocator do for that to be real?
- Describe a workload where cutting the KV cache by 4× gives almost no latency improvement.
- How would you test whether a sliding-window model loses evidence that sits 20,000 tokens back?
- Why is FlashAttention's advantage larger in training than in single-token decode?

# Pitfalls

- Using query heads `h` instead of KV heads `h_kv` in the cache formula. This overstates the GQA cache by the group size.
- Calling FlashAttention "approximate", or claiming it makes attention linear. It's exact up to rounding and still does quadratic FLOPs. What it cuts is memory traffic and scratch storage.
- Thinking FlashAttention shrinks the KV cache. The persistent cache is untouched, and GQA or quantization is what shrinks it.
- Expanding GQA's shared K/V to all query heads in memory (e.g. `repeat_interleave`) in the hot path and quietly losing the savings.
- Setting a larger RoPE scaling factor or max length in a config and calling it "long-context support" without fine-tuning or a length-by-position evaluation.
- Treating the logical KV formula as an admission guarantee. Block rounding, metadata, tensor-parallel replication and output growth all add to it.
- Applying a rolling-buffer cache to a model trained with dense attention, or to its global layers. That changes the model's behaviour.
- Assuming a 4× smaller cache gives 4× faster decode when weight reads dominate each step.

# Checklist

- Name attention's three costs and say which one binds in prefill, decode, and training.
- Write Q, K, V and score shapes with separate `h` and `h_kv`.
- Compute KV bytes per token and per workload for MHA, GQA and MQA, and check the result against a GiB budget.
- Estimate a decode step's bandwidth floor from weights plus cache.
- Walk through online softmax on a two-tile example and show why averaging tile outputs is wrong.
- Explain why FlashAttention is exact, still quadratic, and irrelevant to KV cache size.
- Explain RoPE's relative-position property with the clock-hand picture, and contrast interpolation with extrapolation.
- Design an evaluation grid (length × evidence position × task type) for a context extension.
- Compute sliding-window pair counts and rolling-buffer cache size, and explain why multi-layer reach isn't direct retrieval.
- State which efficiency changes preserve the model's output and which require quality validation.

# Sources

- [Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)](https://arxiv.org/abs/1911.02150) — Introduces multi-query attention: keys and values shared across heads to cut decode memory bandwidth.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) — Defines grouped-query attention, mean-pooling conversion, uptraining with 5% of pretraining compute, and notes on sharding waste.
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825) — Architecture table (32 heads, 8 KV heads, head_dim 128, window 4096), rolling buffer cache with 8× saving at 32k, and the ~131k theoretical span.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — Tiling between HBM and SRAM, online softmax, recomputation in backward, exactness, and reported speedups.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691) — About 2× over v1 and 50–73% of theoretical A100 throughput.
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)](https://arxiv.org/abs/2407.08608) — H100-specific asynchrony and FP8 support, 1.5–2.0× over FlashAttention-2.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — Defines RoPE, which encodes absolute position by rotation and yields relative-position dependence in attention scores.
- [Extending Context Window of Large Language Models via Positional Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595) — Position interpolation, extending LLaMA models to 32,768 tokens within 1,000 fine-tuning steps, and the instability of extrapolation.
- [YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) — A RoPE extension method reporting 10× fewer tokens and 2.5× fewer training steps than prior methods.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172) — U-shaped retrieval performance, with the middle of a long context used worst.
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150) — Local windowed attention plus task-motivated global attention, scaling linearly with sequence length.

# Flashcards

## budgets

**Q:** What three costs should you separate when someone says "efficient attention"?

FLOPs (the `QKᵀ` and `P·V` arithmetic), memory traffic (moving the `T × T` score table between HBM and the compute units), and KV-cache size (persistent K/V bytes re-read every decode step).

They're paid down by different levers. Sliding windows cut FLOPs, FlashAttention cuts traffic, and GQA/MQA cut cache bytes. So an improvement in one doesn't imply an improvement in the others, or in end-to-end latency.

## shapes

**Q:** Write the Q, K and V shapes for grouped-query attention.

Q is `[B, T_q, h, d_head]`. K and V are `[B, T_k, h_kv, d_head]`, with `h_kv < h`.

Groups of `h / h_kv` query heads share each K/V head, but each query head still computes its own scores and output, so the output is still `h` heads wide.

## mqa

**Q:** How do MHA, MQA and GQA differ?

MHA gives each query head its own KV head (`h_kv = h`). MQA shares a single KV head across all query heads (`h_kv = 1`). GQA sits in between (`1 < h_kv < h`).

MQA gives the smallest cache but risks quality. GQA gets close to MHA quality with close to MQA speed. In all three, query outputs stay separate.

## group

**Q:** With 32 query heads and 8 KV heads, how many query heads share each KV head, and does that shrink the attention output?

Four (32 / 8). The output doesn't shrink. All 32 query heads still produce their own results, which are concatenated to width `h · d_head` before `W_O`. Only the stored K/V is shared.

## cache-formula

**Q:** What's the KV-cache size formula, and which head count goes in it?

`2 × L × h_kv × d_head × bytes_per_element × total_retained_tokens`. The 2 counts K and V, and total retained tokens is `B × T` for equal lengths.

It uses `h_kv`, not the query head count. Using `h` overstates a GQA cache by the group size.

## worked

**Q:** Size the KV cache for 32 layers, 8 KV heads, d_head 128, bf16, and 8 sequences of 8,192 tokens.

Per token: 2 × 32 × 8 × 128 × 2 = 131,072 bytes (128 KiB). Tokens: 8 × 8,192 = 65,536. Total: 8,589,934,592 bytes = 8 GiB. MHA with 32 KV heads would be 32 GiB, and MQA 1 GiB.

This is logical payload only, excluding weights, workspace and allocator overhead.

## allocation

**Q:** Why isn't the logical KV payload enough for admission control?

Real allocation adds partially filled blocks, metadata, possible tensor-parallel replication of KV heads, and room for tokens not yet generated. Admit requests to 100% of the payload estimate and they'll fail or get preempted later as outputs grow.

## flash-exact

**Q:** In what sense is FlashAttention "exact"?

It computes the same masked softmax attention as the textbook version, just tiled with a running softmax, and never stores the full score matrix. Because additions happen in a different order, floating-point rounding can differ slightly. It's mathematically exact, not bit-identical.

## online

**Q:** In online softmax, why can't you normalize each tile and average the results?

Each tile's softmax divides by its own denominator, but the true softmax divides by one sum over all keys. You keep a running max `m`, sum `l` and weighted sum `u`, rescale the old ones by `e^(m − m′)` when the max changes, and divide `u / l` only at the end.

In the toy row with scores [1, 3, 2, 4] and values [10, 20, 30, 40], this gives 33.43, while averaging tiles gives a wrong 28.81.

## quadratic

**Q:** Does FlashAttention make attention linear in sequence length?

No. Every query still meets every allowed key, so FLOPs remain quadratic. What becomes linear is the extra *memory*, because the `T × T` table is never materialized. The speedup comes from fewer HBM trips, not less math.

## flash-io

**Q:** Why can FlashAttention be faster even though it does *more* FLOPs in training?

Naive attention writes and re-reads a `T × T` score table in HBM (128 MiB per head at 8k tokens in bf16), and those steps are memory-bound. FlashAttention keeps tiles in on-chip SRAM and recomputes scores during the backward pass instead of storing them. Trading cheap arithmetic for expensive memory trips cuts wall-clock time.

## rope

**Q:** Why doesn't RoPE give you unlimited usable context?

RoPE rotates query/key coordinate pairs by position, so the score depends on relative distance, and the formula accepts any position. But the model has only learned the angle patterns it saw in training. Past that length it may break down. Extension needs a recipe (interpolation, YaRN), usually a fine-tune, and an evaluation across lengths and positions.

## context-extension

**Q:** How does position interpolation differ from extrapolation?

Extrapolation feeds positions beyond the trained range, producing unseen rotation angles and possibly exploding attention scores. Interpolation divides positions by the scale factor (e.g. 6,000 → 1,500 at 4×), so all angles are familiar, at the cost of finer spacing. It is followed by a short fine-tune.

## window

**Q:** When can a sliding-window model keep a bounded KV cache?

When its layers were trained with a causal window `W`. Keys older than `W` can never be read again, so a rolling buffer (slot `i mod W`) overwrites them. Mistral 7B's `W = 4,096` gives 8× less cache at 32k tokens.

Global or full-attention layers still need the whole history, and forcing eviction on a dense-trained model changes its behaviour.

## reach

**Q:** Stacked window layers give a "theoretical span" of `L × W`. Why isn't that the same as global attention?

Distant information only arrives indirectly, hopping through hidden states layer by layer, and it gets compressed at every hop. Summaries can survive, but exact details (an ID from 100k tokens back) often don't. Global attention reads the old token directly.

## benchmark

**Q:** Why might cutting the KV cache 4× give little latency improvement?

Each decode step also reads all the weights (~14 GB for a 7B bf16 model), and other kernels, communication and scheduling take time too. When contexts are short or batches small, weights dominate. The reliable win is capacity (more concurrent sequences), and end-to-end speed has to be measured.
