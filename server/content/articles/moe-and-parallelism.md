---
{
  "slug": "moe-and-parallelism",
  "title": "Mixture of experts: more parameters, same compute per token",
  "category": "architecture",
  "summary": "How a router sends each token to a few expert MLPs, why total and active parameters answer different questions, and what load balancing, the all-to-all and serving memory cost you in exchange.",
  "difficulty": "Advanced",
  "minutes": 30,
  "prerequisites": ["transformer-foundations", "optimization-generalization"],
  "learningObjectives": [
    "Explain how a router with top-k gating turns one MLP into many experts, and walk a token through score, select, dispatch, compute and combine.",
    "Calculate total and active parameters, weight memory, and per-token compute for a real MoE (Mixtral 8x7B) from its config.",
    "Calculate expert capacity, overflow and all-to-all payload for a routing group, and separate bandwidth cost from per-message latency.",
    "Defend when an MoE beats a dense model, and when it doesn't, in terms of training compute, serving memory, batching and communication."
  ]
}
---

# Sections

## Big model, small bill: the idea behind sparse layers {#dense-vs-sparse}

Quick recap of where we are. A transformer is a stack of `L` identical blocks. Each block has two halves: **attention**, where tokens read from each other, and an **MLP** (also called the feed-forward network, FFN), a small two-layer network that runs on every token position independently. In a modern model the MLP is usually a SwiGLU with three weight matrices, and it holds roughly two thirds of the block's parameters. (We built all of this in **The transformer block: assembling the full model**.)

In a **dense** model, every token goes through every weight. If the model has 47 billion parameters, each token pays for 47 billion multiply-adds (well, about 2 FLOPs per parameter per token for the forward pass: one multiply and one add). Make the model bigger and every single token gets more expensive. Capacity (how much the model can store) and cost per token are welded together.

A **mixture-of-experts (MoE)** layer cuts that weld. Instead of one MLP per block, you keep, say, 8 of them, called **experts**. A tiny learned **router** looks at each token and picks a couple of experts to run. The other experts sit idle for that token. So the model can *store* 8 MLPs' worth of knowledge but each token only *pays* for 2.

Analogy: a hospital. A dense model is one general practitioner who sees every patient and has to know everything. An MoE is a clinic with eight doctors and a triage nurse. The nurse (router) glances at each patient and sends them to two doctors. Each patient costs two consultations, not eight. But you still pay salaries and office rent for all eight doctors, and the nurse has to keep any one doctor from getting a queue out the door. Hold on to all three of those facts: cheap per patient, expensive to house, and a queueing problem. They are the whole article.

```text
 DENSE BLOCK                          MoE BLOCK (8 experts, top-2)

 token ─► attention ─► MLP ─► out     token ─► attention ─► router ──┐
                       (all                                         │ picks 2
                        weights                     ┌───┬───┬───┬───┼───┬───┬───┐
                        used)                       │E0 │E1 │E2 │E3 │E4 │E5 │E6 │E7
                                                    └─▲─┴───┴───┴─▲─┴───┴───┴───┘
                                                      │  run only │
                                                      └── these ──┘ ─► weighted sum ─► out
```

One thing that trips people up: an "expert" is **not** a topic specialist that someone designed. Nobody assigns "the maths expert" or "the French expert". Each expert is just an ordinary MLP with its own weights, and the router learns which one helps, token by token, through training. When the Mixtral authors looked at which experts their router picked, they found no obvious split by topic (maths vs biology vs philosophy text looked similar). What they did see was a pattern tied to syntax: things like indentation in code, or particular words such as `self` in Python, kept going to the same experts, and neighbouring tokens often shared experts. Specialisation emerges, and it's usually stranger than "one expert per subject".

Attention is normally *not* split into experts. Every token still runs full attention in every block. MoE replaces only the MLP half.

> 🎬 **Animation — dense vs sparse block:** Left panel: a single wide MLP box; three tokens ("the", "cat", "sat") flow in and every weight in the box lights up for each token, with a counter "weights touched per token: 100%". Right panel: the same total width split into 8 narrower boxes labelled E0–E7, plus a small "router" diamond. Step 1: token "the" arrives, the router shows 8 bars, the top two (E2, E5) glow and only those two boxes light. Step 2: "cat" arrives and lights E0 and E5. Step 3: "sat" lights E3 and E6. Counter on the right reads "weights stored: 8 experts, weights touched per token: 2 experts (25% of the expert weights)". Finish by highlighting that over the three tokens, 6 of the 8 experts were used at least once.

## The router: how a token picks its experts {#router-top-k}

The router is embarrassingly small: one matrix `W_r` of shape `d × E`, where `d` is the hidden width and `E` is the number of experts. For each token, you take the token's hidden vector `x` (the thing flowing down the residual stream, a row of `d` numbers) and multiply:

```formula
logits = x · W_r                        (one score per expert, shape E)
G(x)   = softmax( TopK(logits, k) )     (non-top-k logits set to −∞ first)
y      = Σ over chosen i of  G(x)_i · Expert_i(x)
```

- `x` is one token's hidden vector (length `d`).
- `W_r` is the router's weight matrix (`d × E`). For Mixtral, that's 4,096 × 8 = 32,768 numbers per layer: nothing.
- `TopK(·, k)` keeps the `k` largest scores and sets the rest to −∞, so the softmax gives them weight exactly 0.
- `G(x)_i` is the **gate weight** for expert `i`: how much its output counts. The chosen ones sum to 1.
- `Expert_i(x)` is expert `i`'s MLP output (length `d`), and `y` is the MoE layer's output for this token, which gets added back onto the residual stream as usual.

That exact form is what Mixtral uses. Other models differ in small ways: some take the softmax over *all* experts and then keep the top-k values without renormalising, and Switch Transformers uses `k = 1` and multiplies the single expert's output by its softmax probability. When you read a codebase, check which convention it uses. It changes the scale of the output.

**Tiny worked example.** Four experts, top-2. Suppose the router matmul for one token gives:

```text
 expert:        E0      E1      E2      E3
 logit:        2.0     1.0     0.5    -1.0
 exp(logit):   7.389   2.718   1.649   0.368      (sum = 12.124)
 full softmax: 0.609   0.224   0.136   0.030      (for reference)

 TopK(k=2) keeps E0, E1:
 exp:          7.389   2.718     —       —        (sum = 10.107)
 gate G:       0.731   0.269     0       0
```

So `y = 0.731 · E0(x) + 0.269 · E1(x)`. E2 and E3 never run for this token. Notice the gates are softmax over just the two survivors, which is why they're larger than the full-softmax values (0.609 and 0.224).

**The five steps in a real layer.** For a whole batch, you don't loop over tokens. You do this:

1. **Score**: one matmul, `(tokens × d) · (d × E)` → a `tokens × E` score table.
2. **Select**: top-k per row, and keep the gate weights.
3. **Dispatch**: shuffle token vectors so each expert gets a contiguous pile of just its tokens.
4. **Compute**: each expert runs its MLP on its pile as one ordinary dense matmul.
5. **Combine**: send outputs back, multiply by gates, sum per token, and put everything back in the original order.

```text
 tokens (rows = positions, cols = d features)
 ┌────────┐        score + top-2           dispatch (group by expert)
 │ t0 ░░░ │  ─►  t0→E1,E3  t1→E0,E1  ─►   E0: [t1, t3]        compute each
 │ t1 ░░░ │      t2→E3,E2  t3→E0,E2       E1: [t0, t1]    ─►  pile as a     ─► combine:
 │ t2 ░░░ │                               E2: [t2, t3]        dense matmul      unshuffle,
 │ t3 ░░░ │                               E3: [t0, t2]                        weight by gate,
 └────────┘                                                                    sum per token
```

Here's a subtle training point. Choosing "which experts" is a discrete decision, and you can't take a gradient through "pick index 3". What *does* get a gradient is the gate value: because `y` is multiplied by `G(x)_i`, and `G` comes out of a softmax over the router logits, the loss can nudge `W_r` to raise or lower each chosen expert's score. Experts that help get their scores pushed up, so they're more likely to be picked next time. That's the entire learning signal for the router. (Gradients and backprop are covered in **How a model learns: loss, gradients, and optimizers**.)

Also note that routing is decided on the *contextual* hidden vector, not on the raw word. The word "bank" can go to different experts in different sentences, and at different layers. There's no fixed "token → expert" table.

> 🎬 **Animation — one token through the router:** Show a single token vector x (4 coloured cells). Step 1: x multiplies a 4×4 router matrix W_r, producing four logits [2.0, 1.0, 0.5, −1.0] as bars labelled E0–E3. Step 2: the two tallest bars (E0, E1) stay coloured, and E2, E3 are crossed out and relabelled "−∞". Step 3: softmax over the survivors turns them into gates 0.731 and 0.269, shown as pie slices. Step 4: x is copied into E0 and E1 (two MLP boxes), each emitting an output vector. Step 5: the outputs are scaled by 0.731 and 0.269 and summed into y, which drops back onto a horizontal "residual stream" line.

## Total vs active parameters: two numbers, two bills {#total-vs-active}

Every MoE model gets described with two numbers, and interviewers love to check you know which one controls what.

- **Total parameters**: every weight in the model, all experts included. This decides **memory**: how many GPUs you need just to hold the thing.
- **Active parameters**: the weights one token actually uses in a forward pass (shared parts plus `k` experts per layer). This roughly decides **compute per token**: FLOPs ≈ 2 × active parameters.

```formula
P_total  = P_shared + L_moe × E × P_expert
P_active ≈ P_shared + L_moe × k × P_expert
```

`P_shared` is everything every token uses (attention, embeddings, norms, routers). `L_moe` is the number of MoE layers, `E` is the experts per layer, `k` is the experts chosen per token, and `P_expert` is the parameters in one expert.

**Toy version first.** Say a model has 2B shared parameters, and summed across its layers the experts come to 16 groups of 0.5B each, with top-2. Total = 2 + 16 × 0.5 = **10B**. Active = 2 + 2 × 0.5 = **3B**. Storing it in bf16 (2 bytes per weight) takes 20 GB, while a 3B dense model would need only 6 GB. Same compute per token, over three times the memory.

**Now a real one: Mixtral 8x7B.** Its published config is `d = 4096`, `L = 32` layers, 32 query heads and 8 key/value heads of width 128 (grouped-query attention, see **Making attention cheaper: GQA, FlashAttention, and long context**), SwiGLU expert hidden width 14,336, vocabulary 32,000, 8 experts, and top-2. Every layer is an MoE layer. Let's count.

```text
 One expert (SwiGLU = 3 matrices of d × 14336):
   3 × 4096 × 14336                        = 176,160,768   (~0.176B)

 Attention per layer:
   W_Q  4096 × 4096                        =  16,777,216
   W_K  4096 × 1024   (8 kv heads × 128)   =   4,194,304
   W_V  4096 × 1024                        =   4,194,304
   W_O  4096 × 4096                        =  16,777,216
                                   sum     =  41,943,040
   × 32 layers                             = 1,342,177,280  (~1.34B)

 Embedding + separate output head:
   2 × 32000 × 4096                        =   262,144,000  (~0.26B)
 Routers: 32 × 4096 × 8                    =     1,048,576  (tiny)

 TOTAL  = 32 × 8 × 176,160,768  (45.10B experts)
        + 1.342B + 0.262B + 0.001B         ≈ 46.7B
 ACTIVE = 32 × 2 × 176,160,768  (11.27B experts)
        + 1.342B + 0.262B + 0.001B         ≈ 12.9B
```

(Norm weights add a few hundred thousand more, which is noise.) These land on the ~47B total and ~13B active that the Mixtral paper reports. Two lessons fall out.

First, **"8x7B" is not 56B.** The name suggests eight 7B models glued together, but only the MLPs are copied. Attention and embeddings (about 1.6B) are shared, so the total is 46.7B.

Second, **the two bills really are different.** In bf16, 46.7B parameters is about **93 GB** of weights, which doesn't fit on a single 80 GB GPU before you've even allocated a KV cache. But each token's forward pass costs about 2 × 12.9B ≈ **26 GFLOPs**, the same as a 13B dense model, and about 3.6× less than a 46.7B dense model (~93 GFLOPs).

| Question | Governed by | Mixtral 8x7B |
|---|---|---|
| How many GPUs to hold the weights? | total params × bytes | ~93 GB in bf16 |
| FLOPs per token (forward)? | ≈ 2 × active params | ~26 GFLOPs |
| Weight bytes read per decode step at batch 1? | active params × bytes | ~26 GB |
| Weight bytes read per decode step at a large batch? | nearly total params × bytes | ~93 GB (shared across the batch) |

That last row is the one people miss, and it gets its own section below.

The actual *latency* depends on neither number alone. It also depends on communication, memory bandwidth, attention cost and scheduling. Active parameters tell you the arithmetic, not the wall clock.

> 🎬 **Animation — Mixtral's parameter ledger:** A stacked horizontal bar for "total" (46.7B): a thin grey segment "attention 1.34B", a thinner "embeddings 0.26B", then 8 coloured segments "experts, 5.64B each summed over 32 layers". Below it, an "active" bar: the same grey segments plus only 2 of the 8 coloured segments, reading 12.9B. Step 1: build the total bar segment by segment with the arithmetic shown. Step 2: fade 6 expert segments to outline and slide the rest into the active bar. Step 3: put two gauges next to them: "memory: 93 GB (total × 2 bytes)" pointing at the top bar, and "compute: ~26 GFLOPs/token (2 × active)" pointing at the bottom bar. Step 4: a crossed-out label "8 × 7B = 56B" with the note "only MLPs are copied".

## Keeping the experts busy: load balancing {#load-balancing}

Left alone, routers collapse. Here's the feedback loop. Early in training, expert 3 happens to be slightly better. The router sends it more tokens, so it gets more gradient updates, so it gets better, so it gets more tokens. A few rounds later, two experts handle nearly everything and six are dead weight you're still paying to store. That's the rich-get-richer failure, and it's the core training problem in MoE.

It hurts twice. Quality suffers, because you've paid for 8 experts and effectively trained 2. And speed suffers, because once experts live on different GPUs (next sections), the GPU holding the popular expert does all the work while the others wait.

**The classic fix: an auxiliary loss.** Add a small extra term to the training loss that is lowest when tokens spread evenly. Switch Transformers defines it for `N` experts over a batch as:

```formula
L_aux = α · N · Σ_i ( f_i · P_i )
```

- `f_i` is the **fraction of tokens actually sent** to expert `i` (hard counts from the top-1 choice).
- `P_i` is the **average router probability** given to expert `i` across the batch (soft, and differentiable).
- `N` is the number of experts. Multiplying by `N` makes a perfectly even split score exactly 1, whatever `N` is.
- `α` is a small weight so this doesn't swamp the real loss. Switch used α = 10⁻².

Why the product? `f_i` can't be differentiated (it's a count), but `P_i` can. The gradient flows through `P_i`, weighted by `f_i`: experts that are *already overloaded* get their probability pushed down hardest.

**Worked numbers**, 4 experts, top-1:

```text
 Balanced:   f = [0.25, 0.25, 0.25, 0.25]   P = [0.25, 0.25, 0.25, 0.25]
             Σ f·P = 4 × 0.0625 = 0.25      → 4 × 0.25 = 1.0   (× α)

 Collapsed:  f = [1.0, 0, 0, 0]              P = [0.7, 0.1, 0.1, 0.1]
             Σ f·P = 1.0 × 0.7 = 0.7        → 4 × 0.7  = 2.8   (× α)
```

The collapsed router pays 2.8× the penalty, and the gradient says: lower expert 0's probability.

Two cautions. It's a *pressure*, not a guarantee: any single batch can still be lopsided. And it's a *tradeoff*: the aux loss pulls the router away from what's best for the language-modelling loss, so too large an `α` costs quality.

**The newer fix: bias-based balancing.** DeepSeek-V3 (671B total, 37B active, with 256 routed experts plus 1 **shared expert** per MoE layer, and 8 routed experts chosen per token) mostly drops the auxiliary loss. Each expert gets a bias number that is added to its score *only when choosing the top-k*. After each training step, overloaded experts have their bias lowered by a small amount γ, and underloaded ones have it raised. The gate weights that multiply the expert outputs still come from the original, un-biased scores, so balancing doesn't distort the math of the output. They keep a very small sequence-level balance loss as a safety net, and they report they didn't need to drop any tokens during training. The shared expert is the other trick: one expert that every token always goes through, which learns the common stuff so the routed experts don't all have to relearn it.

**What to monitor** in practice: tokens per expert per layer, max-to-mean load, the overflow rate (next section), and router entropy (how spread out the probabilities are). Don't trust entropy alone: probabilities can look nicely spread while the top-1 choice keeps landing on the same expert.

> 🎬 **Animation — router collapse and the balancing push:** Show 4 expert bins with token counts as bar heights over "training steps". Step 1 (no aux loss): bars start roughly equal at 25 each; expert 0 grows a little, a curved arrow labelled "more tokens → more gradient → better" loops back, and over several steps E0 reaches 90 while the others shrink toward 3 each. Step 2: rewind and enable "aux loss". Each step shows f_i and P_i numbers above the bars and a downward arrow on the tallest bar sized by f_i. The bars settle around 22–28, jittering but never collapsing. Step 3: a caption compares Σ f·P × N: 2.8 (collapsed) vs 1.0 (balanced).

## Capacity, overflow and dropped tokens {#capacity-and-dropping}

Balancing on average isn't enough, because hardware wants fixed shapes. Many MoE implementations give each expert a fixed number of **slots** per routing group, called its **expert capacity**. Think of checkout lanes with a fixed number of spots in each line: if lots of shoppers pick lane 4, some won't fit, even while lane 2 is empty.

```formula
C = ceil( c · T · k / E )
```

`T` is the number of tokens in the routing group (for example one batch shard), `k` the experts per token, `E` the number of experts, and `c` the **capacity factor**, a headroom multiplier (1.0 means exactly the average, and 1.25 gives 25% slack). Libraries define this slightly differently, so check yours.

**Worked example.** `T` = 4,096 tokens, `E` = 16, top-2, `c` = 1.25:

```text
 assignments requested     = 4096 × 2          = 8,192
 average per expert        = 8192 / 16         =   512
 capacity per expert  C    = ceil(1.25 × 512)  =   640
 total slots reserved      = 16 × 640          = 10,240   (25% padding)

 hot expert receives 900   → 900 − 640 = 260 assignments overflow
```

There are 2,048 spare slots in total across the other experts, but that doesn't help. Slots are *per expert*. A token that wanted expert 7 can't be served by expert 12's empty seats unless you explicitly reroute it.

**What happens to overflow?** In GShard and Switch Transformers, an overflowing token **skips that expert**: its contribution is zero, and the token's vector just continues to the next layer along the residual connection (the skip path that adds each sublayer's output back to its input). So "dropped token" doesn't mean the token disappears from the sequence. It means one layer's MLP work was skipped for it. Switch reported that this typically affected under 1% of tokens. That's small, but it's not free: it's a silent change to the computation, and it can get worse on out-of-distribution traffic where routing is more lopsided.

So capacity factor is a genuine three-way trade:

| Capacity factor | Memory and compute | Dropped tokens | Notes |
|---|---|---|---|
| Low (1.0) | least padding | more drops | cheapest; quality risk under skew |
| High (2.0) | up to 2× expert buffer, and padded matmuls waste FLOPs | few drops | buys safety with hardware |
| Dropless | variable-size piles, no padding | none | needs special kernels; hot experts still become stragglers |

**Dropless MoE.** MegaBlocks reframes expert compute as *block-sparse* matrix multiplication, which handles experts that receive different numbers of tokens without padding them to a fixed `C`. So it never drops tokens. Its authors reported training up to 40% faster than the Tutel MoE library. Dropless removes the quality tax, but not the physics: if one expert gets 3× the tokens, whichever GPU owns it takes 3× as long, and everyone waits.

> 🎬 **Animation — capacity overflow:** 16 vertical bins (experts), each with a dashed line at 640 slots and a dotted line at the mean of 512. Step 1: 8,192 token-dots rain in, mostly evenly, but expert 7 fills to 900. Step 2: dots above the 640 line in expert 7 (260 of them) turn red and slide sideways onto a horizontal "residual stream" arrow that bypasses the MoE layer. Step 3: highlight the empty space in the other 15 bins with the label "2,048 free slots, wrong experts". Step 4: switch to a "dropless" mode where expert 7's bin stretches to 900 with no red dots, and a clock above it runs 1.4× longer than the others.

## Expert parallelism and the all-to-all {#expert-parallelism}

Once the model is too big for one GPU (Mixtral already is, in bf16), you have to spread it out. The general tools for this are data parallelism (copies of the model on different batches), tensor parallelism (splitting individual matmuls across GPUs) and pipeline parallelism (different layers on different GPUs). We'll go through those in detail in **Distributed training: fitting a training run onto a cluster**. MoE adds one more that's specific to it.

**Expert parallelism (EP)** puts *different experts on different GPUs*. With 8 experts and 4 GPUs, GPU 0 holds E0 and E1, GPU 1 holds E2 and E3, and so on. Attention is usually replicated or split by the other methods. So the weights stay put and **tokens travel to the weights**. It's the opposite of what you'd do in a dense model, where you'd split the matrices.

The travelling happens through a collective called **all-to-all**: every GPU sends a different chunk of data to every other GPU. (A *collective* is a communication step that a group of GPUs performs together.) An MoE layer needs two of them in the forward pass:

```text
 BEFORE (each GPU has its own batch of tokens, colour = chosen expert's GPU)

 GPU0: [a→0 b→2 c→1 d→3]     all-to-all #1       GPU0: [a  e  i  m]   run E0,E1
 GPU1: [e→0 f→1 g→3 h→2]    ───── dispatch ───►  GPU1: [c  f  j  n]   run E2,E3  ...
 GPU2: [i→0 j→1 k→2 l→3]                         GPU2: [b  h  k  o]   run E4,E5
 GPU3: [m→0 n→1 o→2 p→3]                         GPU3: [d  g  l  p]   run E6,E7

                             all-to-all #2
         outputs go home   ◄───── combine ─────  (reverse permutation)
```

Every GPU talks to every other GPU, twice per MoE layer, plus two more times in the backward pass during training. That's why EP lives or dies on the network.

**Sizing the traffic.** Forward-pass bytes for one MoE layer, across the whole EP group:

```formula
payload ≈ 2 · T · k · d · b      bytes   (dispatch + combine)
```

`T` is the tokens in the group, `k` the experts per token, `d` the hidden width, and `b` the bytes per number. The 2 is for there and back.

**Worked example (prefill or training, big batch).** `T` = 8,192 tokens, `k` = 2, `d` = 4,096, bf16 (`b` = 2), and 8 GPUs in the EP group:

```text
 dispatch   = 8192 × 2 × 4096 × 2 B     = 134,217,728 B  = 128 MiB
 + combine  (same size)                                  = 256 MiB total
 per GPU    = 256 / 8                                    =  32 MiB
 remote     = 7/8 of that (uniform routing)              =  28 MiB  = 29,360,128 B
 at 25 GB/s effective link bandwidth (assumed):
   29,360,128 / 25e9                                     ≈ 1.17 ms per MoE layer
 × 32 MoE layers                                         ≈ 37.6 ms per forward pass
```

That's a floor, not a prediction. It ignores contention, uneven routing and padding. But it tells you EP traffic is comparable to real compute time, so you either hide it (overlap with other work) or keep it on fast links. DeepSeek-V3 did both: it caps each token at 4 nodes' worth of experts to limit slow cross-node traffic, and schedules computation so the all-to-all is almost entirely overlapped with compute.

**Worked example (decode, small batch).** Now 64 sequences each generating one token: `T` = 64.

```text
 per GPU:  8 tokens × 2 experts × 4096 × 2 B  = 131,072 B dispatched
 remote:   7/8 → 114,688 B, split across 7 peers ≈ 16 KiB each
 transfer at 25 GB/s:    114,688 / 25e9 ≈ 4.6 µs
```

If each all-to-all has a fixed startup cost of around 10 µs (an illustrative number for launch plus synchronisation), you're spending more time on the *handshake* than on the *bytes*. A cost model that captures this:

```formula
t_comm ≈ n_messages · α + bytes / B_eff
```

`α` is the fixed per-message latency and `B_eff` the effective bandwidth. For prefill, the second term dominates, and a faster network helps. For low-batch decode, the first term dominates, and doubling bandwidth barely helps. The fixes are to use fewer, larger messages (batch more), use fewer EP ranks, or keep the EP group inside one server's fast NVLink domain.

**EP is not tensor parallelism.** TP slices one matrix across GPUs, and every GPU does part of every token's math. EP gives whole experts to GPUs, and each GPU does all the math for a subset of tokens. They combine: a giant expert can itself be tensor-sharded. In practice EP groups are often carved out of the data-parallel dimension (attention is replicated across the group, experts are split across it), so when you describe a layout, draw which GPU owns which weights instead of just multiplying parallelism degrees. The next section covers what happens when one of those GPUs fails.

> 🎬 **Animation — the all-to-all:** Four GPU boxes in a row, each holding four coloured token squares (colour = the GPU that owns the chosen expert) and labelled with its two experts. Step 1 (dispatch): squares fly along a full mesh of arrows so each GPU ends up holding only squares of its own colour; a byte counter ticks up to "128 MiB group-wide". Step 2: each GPU's pile goes through its expert boxes (a short "compute" bar). Step 3 (combine): squares fly back to their home GPU and slot into their original positions. Step 4: replay at decode scale: only two tiny squares per GPU, and a timeline shows a long grey "startup 10 µs" segment followed by a sliver "transfer 4.6 µs".

## Serving an MoE: memory, batching, and failure {#serving-moe}

At inference time the MoE bargain looks different from how it looks in training, and this is where senior interviews dig in. A quick primer on inference: generation has a **prefill** phase (process the whole prompt at once, lots of tokens, compute-bound) and a **decode** phase (one new token per sequence per step, which is limited by how fast you can read weights from GPU memory). We'll go through this properly in **What happens at inference: prefill, decode, and the KV cache**.

**1. All experts must be resident.** You don't know which experts the next token will want, so every expert must be in GPU memory (or you pay to page it in, which is painfully slow). Memory is set by total parameters. Mixtral needs ~93 GB of bf16 weights plus KV cache. A 13B dense model with similar compute per token needs ~26 GB.

**2. At batch 1, MoE decode is cheap.** Decode speed is roughly "bytes of weights read ÷ memory bandwidth". At batch 1 you read only the active weights: ~12.9B × 2 B ≈ 25.8 GB per token, versus ~93 GB for a dense 46.7B model. At an illustrative 3 TB/s of HBM bandwidth, that's about 8.6 ms vs 31 ms per token. That's the headline MoE win: big-model quality at small-model decode speed.

**3. As the batch grows, the batch touches every expert.** Each token picks `k` of `E` experts. If routing is roughly uniform and independent, the chance that a given expert is used by at least one token in a batch of `B` is:

```formula
P(expert used) ≈ 1 − (1 − k/E)^B
```

```text
 Mixtral (k=2, E=8):     B = 1  → 25%     B = 4  → 68%     B = 16 → 99%
 DeepSeek-V3-like
   (k=8, E=256):         B = 1  → 3%      B = 32 → 64%     B = 256 → ~100%
```

With Mixtral at batch 16, essentially every expert's weights get read every step. So per step you read nearly the *total* parameters, and the "only 13B active" saving on memory traffic has evaporated. (It's shared across 16 tokens, so it's still efficient per token, but the step itself is as heavy as a dense 47B model's.)

**4. And each expert sees only a handful of tokens.** At batch `B`, each expert processes about `B · k / E` tokens on average. For Mixtral at `B` = 16 that's 4 tokens per expert. For a 256-expert model at `B` = 32, it's 1. Tiny matmuls leave most of a GPU's compute idle. This is why large MoE deployments run big decode batches and wide expert parallelism: DeepSeek-V3's report describes decoding with 320-way expert parallelism, where each GPU hosts roughly one expert so that each expert's pile gets big enough to be efficient. They also add **redundant experts** (extra copies of the hottest experts on other GPUs) to spread the load. Batching and scheduling are covered in **Serving many users: batching, scheduling, and speculative decoding**.

**5. Stragglers set the pace.** Every step waits for the slowest GPU. If the router sends a burst of tokens to one expert, its GPU becomes the straggler and everyone else idles. Hot-expert skew hits tail latency, not just throughput.

**6. Failure is replica-wide.** If the GPU holding E5 dies, the model can't compute the next token for anyone whose router picks E5, which could be anyone. Quietly "routing around" the missing expert changes the model's behaviour, which is a silent quality regression. The safe baseline is to stop sending new requests to that replica, retry in-flight requests on a healthy replica, and restore it before it rejoins. Treat any degraded-routing fallback as a separately evaluated model.

> 🎬 **Animation — batch size vs experts touched:** A grid of 8 expert boxes for Mixtral. Step 1: batch 1: one token lights 2 boxes; a gauge "weights read this step: 12.9B" and "tokens per expert: 1". Step 2: batch 4: four tokens, lines to their two experts each, 5–6 boxes lit; gauge rises to ~34B. Step 3: batch 16: all 8 lit; gauge reads ~46.7B, but a second gauge "weights read per token" falls to ~2.9B. Step 4: a side panel with 256 tiny boxes (DeepSeek-like, top-8): batch 32 lights about 64% of them, and each lit box shows "~1 token", with a small matmul that is mostly empty.

## MoE vs dense: when the trade is worth it {#moe-vs-dense}

Now we can argue it properly. MoE doesn't make anything free. It **moves cost from FLOPs to memory and communication**.

| | Dense | MoE (same active params) |
|---|---|---|
| Training FLOPs per token | baseline | ~same |
| Quality at that training compute | baseline | usually higher (more stored capacity) |
| Weight memory | 1× | E/k-ish× larger (Mixtral: 3.6× total/active) |
| Decode at batch 1 | baseline | similar speed, more memory |
| Decode at large batch | baseline | reads nearly all weights each step; needs big batches to feed the experts |
| Communication | TP/PP/DP only | adds 2 all-to-alls per MoE layer per pass |
| Training stability | well understood | router collapse, balancing tuning, some instability |
| Fine-tuning | straightforward | more prone to overfitting (Switch used higher dropout inside experts) |

The evidence for the "usually higher quality" row: Switch Transformers reported reaching the quality of T5-Base in one seventh of the time, with the same compute per token, using 64 experts. Mixtral reports matching or beating Llama 2 70B on most benchmarks they tested with ~13B active parameters. Both are the authors' own measurements on their setups, not universal laws.

**When MoE wins:**
- You're **compute-bound in training** and want the best model for a fixed FLOP budget.
- You serve at **high throughput** on a cluster that holds all the weights, with enough traffic to batch.
- You have fast interconnect (NVLink-class inside a node) for the all-to-all.

**When dense wins:**
- **Memory is the constraint**, for example on-device, a single GPU, or a small shop. A 13B-active MoE that needs 93 GB loses to a 13B dense model that fits in 26 GB.
- **Low, bursty traffic** where you can't batch enough to keep experts fed, and each step pays all-to-all latency.
- You want the simplest thing to fine-tune, quantise and debug.

Middle paths exist too. You can **distil** a big MoE into a small dense model: Switch reported keeping about 30% of the sparse model's quality gain with about 1/20th of the parameters. And you can quantise MoE weights harder, since memory is the bill (see **Quantization: spending fewer bits per weight**).

Here's the one-sentence version for an interview. **MoE decouples capacity from per-token compute: you pay for total parameters in memory, for active parameters in FLOPs, and for routing in communication and load balance, so it wins when compute is the scarce resource and you have the memory, the interconnect and the traffic to keep every expert busy.**

> 🎬 **Animation — where the cost went:** Three sliders labelled "FLOPs/token", "memory", "communication". Step 1: a dense 47B model sets all three high, medium and low. Step 2: switching to Mixtral 8x7B drops FLOPs/token to about 28% (12.9/46.7), leaves memory at 100%, and pushes communication up with an "all-to-all ×2 per layer" tag. Step 3: a dense 13B model: FLOPs match Mixtral, memory drops to 28%, communication low, and a quality meter drops a notch. Caption: "MoE sits where compute is cheap and memory is expensive."

# Interview

## Question

You're serving a mixture-of-experts model with ~47B total and ~13B active parameters on 8 GPUs with expert parallelism. At low concurrency its per-token latency is *worse* than a 13B dense model's, even though the FLOPs per token are about the same. Explain why, and how you'd investigate and fix it.

## Answer

First I'd rule out apples-to-oranges: same precision, same prompt and output lengths, same attention setup, and comparable quality targets. Then I'd separate the three ledgers. **Compute** per token is similar (~2 × 13B FLOPs). **Memory** is not: the MoE holds ~93 GB of bf16 weights, which forces it across several GPUs where the dense 13B fits on one. And **communication** is new: every MoE layer does two all-to-alls in the forward pass, one to dispatch tokens to expert-owning GPUs and one to bring outputs back. With 32 MoE layers, that's 64 collectives per decode step.

At low concurrency those messages are tiny. With a handful of tokens per step, each GPU sends a few kilobytes per peer, so transfer time is a few microseconds but each collective pays a fixed startup and synchronisation cost. The step time is `n_messages · α + bytes / bandwidth`, and the `α` term dominates. Also, each expert processes one or two tokens, so its matmuls are tiny and the GPU is mostly idle. And every step waits for the slowest GPU, so any routing skew shows up directly as latency.

To investigate, I'd profile per-layer time split into attention, expert compute and collective time. I'd log tokens per expert per step and the max-to-mean load, and check the message sizes and whether the all-to-all crosses node boundaries. To tell a hot expert from a congested network interface, I'd compare per-GPU compute time (hot expert means one GPU computes longer) against per-link transfer time (congestion means comms take longer while compute is balanced).

Fixes, in rough order. Shrink the EP group, or keep it inside one NVLink domain so each collective is cheaper. If memory allows, run more independent replicas with fewer GPUs each rather than one wide replica. Batch more aggressively within the latency SLO so messages and expert matmuls get bigger. Overlap the all-to-all with attention or shared-expert compute. Replicate hot experts. Quantise weights to fit on fewer GPUs. Anything that changes routing (capacity limits, dropping experts on failure) needs a quality eval, because it changes the model's function. If the product really is low-concurrency and latency-bound, the honest answer may be that a dense model is the better fit.

## Follow-ups

- What changes when prefill dominates the workload instead of decode?
- How would you tell a hot expert apart from a congested network link in a profile?
- When does replicating an expert help, and what does it cost in memory and, during training, in gradient synchronisation?
- Why did DeepSeek-V3 limit each token to experts on at most 4 nodes?
- How does an auxiliary balancing loss differ from bias-based balancing, and why might the latter hurt quality less?

# Pitfalls

- Treating active parameters as the memory requirement. Memory is set by total parameters: every expert must be resident because you don't know who the next token will pick.
- Assuming decode stays as cheap as the active count at any batch size. At moderate batch sizes the batch touches nearly every expert, so each step reads close to all the weights.
- Reading "8x7B" as 56B, or thinking each expert is a full 7B model. Only the MLPs are replicated; attention and embeddings are shared.
- Thinking the load-balancing loss guarantees an even split in every batch. It's an average pressure; individual batches still skew, and capacity limits or dropless kernels have to handle the rest.
- Assuming a dropped token vanishes from the sequence. It skips that expert's contribution and rides the residual connection. That's a silent change to the computation that needs a quality check, not a crash.
- Assuming spare slots on one expert absorb another expert's overflow. Capacity is per expert; global free space doesn't help a local hot spot.
- Counting group-wide all-to-all bytes as per-GPU bytes, or assuming more bandwidth fixes small-batch decode when per-message latency dominates.
- Calling experts "topic specialists". They're learned MLPs chosen per token by context, and observed specialisation is often syntactic.

# Checklist

- Walk a token through score → top-k → dispatch → expert compute → combine, including the gate-weight arithmetic.
- Compute total and active parameters, bf16 weight memory and FLOPs per token for Mixtral 8x7B from its config.
- Write the Switch auxiliary loss, say what f_i and P_i are, and why only P_i carries gradient.
- Compute expert capacity and overflow for a given T, k, E and capacity factor, and explain what happens to overflow.
- Estimate all-to-all bytes per GPU per layer, and say whether bandwidth or per-message latency will dominate.
- Explain why batch size changes how many experts' weights get read per decode step.
- Argue for or against MoE for a given workload in terms of memory, compute, communication and traffic shape.

# Sources

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538) — Origin of the modern sparsely-gated MoE layer: up to thousands of expert feed-forward networks, a trainable gate choosing a sparse combination, models up to 137B parameters.
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668) — Top-2 gating, expert capacity, local group dispatch, overflowed tokens passed via residual connections, AllToAll communication, 600B-parameter model.
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph, Shazeer)](https://arxiv.org/abs/2101.03961) — Top-1 routing, the α·N·Σ f_i·P_i auxiliary loss with α = 10⁻², capacity factor, token dropping (typically under 1%), 7× pre-training speedup over T5-Base, expert dropout, and distillation results.
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) — Architecture table (d = 4096, 32 layers, 8 experts, top-2, SwiGLU experts), Softmax(TopK) gating, ~47B total and ~13B active parameters, routing analysis showing no clear topic specialisation, expert parallelism notes.
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437) — 671B total and 37B active; 1 shared + 256 routed experts with 8 active; auxiliary-loss-free bias balancing; node-limited routing (at most 4 nodes); no token dropping; EP32 prefill and EP320 decode deployment with redundant experts.
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Gale et al., 2022)](https://arxiv.org/abs/2211.15841) — Block-sparse formulation for dropless MoE without padding; reported up to 40% faster than Tutel.

# Flashcards

## total-active

**Q:** Why do total and active parameters answer different questions?

Total parameters decide **memory**: every expert has to sit in GPU memory because you don't know which ones the next token will pick. Active parameters (shared weights plus the k chosen experts per layer) decide **compute per token**: forward FLOPs ≈ 2 × active.

Neither is a latency number. Real speed also depends on all-to-all communication, memory bandwidth, attention, and how evenly tokens spread across experts.

## expert-identity

**Q:** Is an MoE expert a specialist in a named topic, like maths or French?

No. An expert is an ordinary MLP with its own weights, and the router learns per token, from the contextual hidden vector, which experts to use. Nobody assigns topics.

Some specialisation does emerge, but it's often syntactic rather than topical. Mixtral's routing analysis found no obvious topic split, and instead saw patterns like indentation tokens or Python's `self` going to consistent experts.

## parameter-example

**Q:** A model has 2B shared parameters and, summed over layers, 16 expert groups of 0.5B each, with top-2 routing. What are total and active parameters, and bf16 weight memory?

Total = 2 + 16 × 0.5 = **10B**. Active = 2 + 2 × 0.5 = **3B** (ignoring the tiny router).

At 2 bytes per weight, the weights need **20 GB**, versus 6 GB for a 3B dense model with the same compute per token. Same FLOPs, more than 3× the memory.

## mixtral-arithmetic

**Q:** Why is Mixtral "8x7B" about 47B total and 13B active, not 56B and 14B?

Only the MLPs are replicated into 8 experts. Each SwiGLU expert is 3 × 4096 × 14336 ≈ 0.176B per layer, so 8 experts × 32 layers ≈ 45.1B. Attention (~1.34B with GQA) and embeddings plus the output head (~0.26B) are **shared**, which gives ≈ 46.7B total.

Active = 2 experts × 32 layers × 0.176B ≈ 11.3B plus the same ~1.6B shared ≈ 12.9B. In bf16 that's ~93 GB of weights but only ~26 GFLOPs per token.

## router-topk

**Q:** Router logits for 4 experts are [2.0, 1.0, 0.5, −1.0] with top-2 and Mixtral-style gating. What are the gates, and what is the output?

Keep E0 and E1, set the others to −∞, then softmax the survivors: e² = 7.389 and e¹ = 2.718, sum 10.107. The gates are **0.731** and **0.269**.

Output y = 0.731 · E0(x) + 0.269 · E1(x). E2 and E3 never run. Gradients reach the router through these gate values, since the top-k choice itself isn't differentiable.

## auxiliary

**Q:** What does the Switch Transformers load-balancing loss use, and why that form?

L_aux = α · N · Σ f_i · P_i, where f_i is the fraction of tokens actually sent to expert i (hard counts), P_i is the mean router probability for expert i (soft), N is the number of experts, and α = 10⁻² in the paper.

f_i isn't differentiable, so the gradient flows through P_i, scaled by f_i: overloaded experts get pushed down hardest. A perfect split scores 1 (× α). It encourages balance on average but doesn't guarantee an even split in every batch.

## aux-loss-free

**Q:** How does DeepSeek-V3 balance experts without relying on an auxiliary loss?

Each expert gets a bias that is added to its score **only for choosing the top-k**. After each step, overloaded experts' biases are lowered by γ and underloaded ones raised.

The gate weights that scale expert outputs still come from the original scores, so balancing steers traffic without distorting the output or fighting the main loss through an extra gradient. A tiny sequence-level balance loss remains as a safety net, and they report no token dropping during training.

## capacity

**Q:** Compute expert capacity for T = 4096, k = 2, E = 16, capacity factor 1.25. What happens if one expert gets 900 assignments?

C = ceil(1.25 × 4096 × 2 / 16) = ceil(1.25 × 512) = **640** slots per expert.

The hot expert overflows by 900 − 640 = **260**, even though there are 2,048 free slots across the other experts. Capacity is per expert, so global spare room doesn't help a local hot spot.

## dropping

**Q:** Does "token dropping" delete a token from the sequence?

No. The token skips that expert's contribution for that layer (it gets zero from it), and its vector continues to the next layer through the residual connection.

It's still a silent change to the model's computation, so it needs quality evaluation, especially on out-of-distribution traffic where routing is more skewed. Switch reported typical drop rates under 1%.

## dropless

**Q:** Does dropless MoE (like MegaBlocks) eliminate load imbalance?

No. Block-sparse kernels let each expert process however many tokens it receives without padding or dropping, which removes the quality cost of dropping and the FLOP waste of padding.

But a hot expert still does more work, so the GPU that owns it becomes a straggler and everyone waits. It also still needs memory for its bigger pile. Dropless fixes correctness and padding, not physics.

## expert

**Q:** How is expert parallelism different from tensor parallelism?

Expert parallelism puts **whole, different experts** on different GPUs and moves tokens to them with all-to-all collectives. Each GPU does all the math for a subset of tokens.

Tensor parallelism **splits one matrix** across GPUs, so every GPU does part of the math for every token. They can be combined: a large expert can itself be tensor-sharded. EP groups are often carved out of the data-parallel dimension, so draw the rank map rather than multiplying parallelism degrees.

## payload

**Q:** Estimate forward all-to-all traffic for one MoE layer with T = 8192, k = 2, d = 4096, bf16, 8 GPUs.

Group-wide: 2 · T · k · d · b = 2 × 8192 × 2 × 4096 × 2 B = **256 MiB** (dispatch plus combine).

Per GPU: 32 MiB, of which about 7/8 (28 MiB) is remote under uniform routing. At an assumed 25 GB/s that's about 1.17 ms per layer, as a lower bound. It excludes contention, padding and backward traffic, and group-wide bytes are not per-GPU bytes.

## latency

**Q:** Why might doubling network bandwidth barely speed up low-concurrency MoE decode?

With only a few tokens per step, each all-to-all message is a few kilobytes. Time ≈ n_messages · α + bytes / bandwidth, and the fixed per-message startup α dominates.

Bandwidth only shrinks the second term. Better fixes: fewer, larger messages (batch more), fewer EP ranks, keeping EP inside one fast NVLink domain, or overlapping communication with compute.

## batch-touches-experts

**Q:** Why does MoE decode stop reading "only the active weights" as batch size grows?

Each token uses k of E experts, so the chance a given expert is used by at least one of B tokens is about 1 − (1 − k/E)^B. For Mixtral (k = 2, E = 8), batch 16 gives 1 − 0.75¹⁶ ≈ 99%.

So each step reads nearly all the weights, like a dense model of the total size, though the cost is shared across the batch. Meanwhile each expert gets only about B·k/E tokens, so its matmuls are small and inefficient.

## failure

**Q:** Why can losing one expert-owning GPU take down a whole model replica?

Any token's router might pick the expert on that GPU, so the replica can't correctly compute the next token for anyone. Silently routing around the missing expert changes the model's function, which is a silent quality regression.

The safe baseline: stop admitting new requests to that replica, retry in-flight work on a healthy replica, and restore it before it rejoins. Treat any degraded-routing fallback as a separate model that needs its own eval.

## moe-vs-dense

**Q:** When is a dense model a better choice than an MoE with the same active parameters?

When **memory** is the binding constraint (one GPU, on-device, a small cluster): the MoE needs memory for its total parameters, so a 13B-active MoE needs ~93 GB where a 13B dense model needs ~26 GB.

Also when traffic is **low or bursty**, so you can't batch enough to feed the experts and every step pays all-to-all latency, and when simplicity of fine-tuning, quantisation and debugging matters. MoE wins when training compute is the scarce resource and you have the memory, interconnect and traffic to keep experts busy.
