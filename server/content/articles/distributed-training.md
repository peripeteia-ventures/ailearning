---
{
  "slug": "distributed-training",
  "title": "Distributed training: fitting a training run onto a cluster",
  "category": "training",
  "summary": "Why a 7B model won't train on one GPU, and how data, ZeRO/FSDP, tensor, pipeline, sequence and expert parallelism split the memory and the work across a cluster without drowning it in communication.",
  "difficulty": "Systems",
  "minutes": 40,
  "prerequisites": ["transformer-foundations", "optimization-generalization"],
  "learningObjectives": [
    "Calculate the per-GPU memory of a mixed-precision AdamW run (weights, gradients, optimizer state, activations) and show how ZeRO/FSDP stages 1–3 shrink it.",
    "Explain what data, tensor, pipeline, sequence/context and expert parallelism each split, which collective each one needs, and how often it runs.",
    "Estimate pipeline bubbles, all-reduce traffic and MFU with back-of-envelope numbers.",
    "Map parallelism dimensions onto hardware (NVLink inside a node, InfiniBand/Ethernet across nodes) and defend the placement.",
    "Design a checkpointing and restart strategy that survives the failure rate of a large cluster."
  ]
}
---

# Sections

## Why one GPU isn't enough: the 16-bytes-per-parameter bill {#memory-bill}

Here's the uncomfortable fact that starts every conversation about distributed training: a 7-billion-parameter model is only 14 GB of weights in bf16, which fits easily on an 80 GB GPU, and yet you **cannot train it** on that GPU. The weights are the smallest part of the bill.

Think of training like running a restaurant kitchen. The recipe book (the weights) is small. But while you cook you also need the notes from the last service (gradients), the head chef's running tallies of what's been working (optimizer state), and every half-finished dish on the counter waiting to be plated (activations). The counter space runs out long before the recipe book gets too big.

Let's itemise it. We train in **mixed precision**: the forward and backward passes run in a 16-bit format (bf16, 2 bytes per number) because that's fast on tensor cores, but the optimizer keeps a 32-bit "master" copy of every weight, because tiny updates like 0.0001 × gradient get rounded away in 16 bits. The optimizer is **AdamW**, which keeps two running statistics per parameter: `m` (a moving average of the gradient, i.e. momentum) and `v` (a moving average of the squared gradient, used to scale each parameter's step). Both live in fp32. (We'll go through why Adam needs these, and why master weights exist, in **How a model learns: loss, gradients, and optimizers**.)

| What | Precision | Bytes per parameter | 7B model |
|---|---|---|---|
| Working weights | bf16 | 2 | 14 GB |
| Gradients | bf16 | 2 | 14 GB |
| Master weights | fp32 | 4 | 28 GB |
| Adam `m` | fp32 | 4 | 28 GB |
| Adam `v` | fp32 | 4 | 28 GB |
| **Total model state** | | **16** | **112 GB** |

This is exactly the accounting from the ZeRO paper, which writes it as `2Ψ + 2Ψ + KΨ = 16Ψ` bytes, where Ψ is the parameter count and K = 12 is the optimizer's bytes per parameter (4 + 4 + 4).

```formula
model_state_bytes ≈ (2 + 2 + 12) · N = 16 · N
```

`N` is the parameter count. The 2 + 2 are bf16 weights and gradients, and the 12 is fp32 master weights plus Adam's `m` and `v`.

So 7B × 16 bytes = 112 GB, and the GPU has 80 GB. It doesn't fit, and we haven't stored a single activation yet. Flip it around: an 80 GB card can hold the full training state of at most 80 / 16 = **5B parameters**, and in practice less, because activations, communication buffers and memory fragmentation all need room too.

Real frameworks vary the recipe a little. Some keep gradients in fp32 (18 bytes/param), some drop the separate bf16 copy and cast on the fly, and 8-bit optimizers shrink `m` and `v`. But "about 16 bytes per parameter for mixed-precision Adam" is the number to have in your head in an interview. The key insight is that **three-quarters of it (the 12 bytes) is optimizer state that is only touched once per step**, in the update. Hold onto that, because it's exactly what ZeRO goes after first.

> 🎬 **Animation — the 16-byte bill:** start with a single 80 GB GPU drawn as a tall empty bar with an 80 GB line. Step 1: drop in a 14 GB block labelled "bf16 weights" (it fits, lots of room). Step 2: add a 14 GB "bf16 gradients" block (28 GB total). Step 3: add a 28 GB "fp32 master weights" block (56 GB). Step 4: add a 28 GB "Adam m" block; the stack crosses the 80 GB line at 84 GB and turns red. Step 5: add the final 28 GB "Adam v" block to reach 112 GB, with a caption "and no activations yet". End by bracketing the three fp32 blocks as "84 GB = optimizer state, used once per step".

## Activations: the part of the bill that grows with the batch {#activations}

The model state is fixed by the parameter count. **Activations** are different: they're the intermediate results of the forward pass (the input to every matmul, the output of every nonlinearity) that the backward pass needs in order to compute gradients. The backward pass for `y = x · W` needs `x` to compute the gradient of `W`, so `x` has to be kept until then. Activation memory grows with batch size `B`, sequence length `T` and model width `d`, times the number of layers `L`.

Megatron's activation-recomputation paper gives a handy estimate for one standard transformer layer, in bytes, assuming 16-bit activations:

```formula
activation_bytes_per_layer ≈ T · B · d · (34 + 5 · h · T / d)
```

`T` is sequence length, `B` is micro-batch size in sequences, `d` is model width and `h` is the number of attention heads. The `34` term covers the linear layers, norms, activations and dropout masks. The `5hT/d` term is the attention score matrices, the T × T softmax inputs and outputs for every head.

Worked example, for a 7B-class shape (d = 4,096, h = 32, L = 32) at T = 4,096 and B = 1 (illustrative; the exact constant depends on the MLP variant and kernels):

```text
T·B·d               = 4,096 × 1 × 4,096     = 16.8 M
5·h·T/d             = 5 × 32 × 4,096 / 4,096 = 160

naive attention:  16.8M × (34 + 160) = 3.25 GB per layer  × 32 layers ≈ 104 GB
fused attention:  16.8M × 34         = 0.57 GB per layer  × 32 layers ≈  18 GB
```

That first line is scary: the T × T score matrices dominate. Fused attention kernels like FlashAttention never materialise the full score matrix and recompute the pieces in the backward pass, which removes the `5hT/d` term. (We'll go through how in **Making attention cheaper: GQA, FlashAttention, and long context**.) So call it roughly 18 GB of activations per sequence of 4,096 tokens for our 7B model.

### Activation checkpointing: pay compute to save memory

The big lever is **activation checkpointing** (also called rematerialisation or recomputation). Instead of keeping every activation, you keep only each layer's input (a "checkpoint") and throw the rest away. During the backward pass, when you reach a layer, you rerun its forward pass from the saved input to regenerate what you need.

```text
without checkpointing          with full checkpointing
 layer 32: [all 0.57 GB]        layer 32: [input 33 MB]   ← recompute on the way back
 layer 31: [all 0.57 GB]        layer 31: [input 33 MB]
   ...                             ...
 layer 1:  [all 0.57 GB]        layer 1:  [input 33 MB]
 total ≈ 18 GB                  total ≈ 1.1 GB + one layer's working set (~0.6 GB)
```

Each layer input is `T · B · d · 2 bytes` = 16.8M × 2 = 33.5 MB, and 32 of them is about 1.07 GB. The price: a backward pass costs roughly twice a forward pass, so a normal step is about 3 forward-units of compute, and recomputing the forward makes it about 4. That's roughly **33% more compute** for about a 17× activation saving in this example.

Megatron's follow-up work showed you don't need to recompute everything. **Selective recomputation** only recomputes the attention-score pieces, which take lots of memory but few FLOPs, and the paper reports about 65–70% activation savings for only 1.6–2.7% extra compute on its GPT-3-style models. The senior-level point: activation memory is the knob you trade against compute, while model-state memory is the knob you trade against communication. The rest of this article is mostly about the second one.

> 🎬 **Animation — activation checkpointing:** draw a column of 8 layer boxes. Forward pass: a token-batch arrow climbs up; without checkpointing each layer leaves a tall orange "stored activations" bar beside it. Replay with checkpointing: each layer leaves only a thin blue "saved input" bar. Backward pass: an arrow descends; at each layer, a small flash shows the forward being rerun from the thin bar, the tall bar reappears briefly, the gradient is computed, then the bar vanishes. A memory gauge at the side shows peak ~18 GB in the first run versus ~1.7 GB in the second, and a compute counter shows 3 units versus 4 units.

## Data parallelism: many copies, one averaged gradient {#data-parallelism}

The simplest way to use more GPUs is **data parallelism (DP)**: every GPU holds a complete copy of the model, each gets a different slice of the batch, each computes gradients on its slice, and then they **average their gradients** so every copy takes the identical optimizer step. It's several identical kitchens cooking different orders from the same recipe book, then comparing notes at closing time so tomorrow's recipe book is the same everywhere.

```text
             global batch of 32 sequences
   ┌───────────┬───────────┬───────────┬───────────┐
 GPU0: seq 0-7  GPU1: 8-15   GPU2: 16-23  GPU3: 24-31
 full model     full model   full model   full model
 grad g0        grad g1      grad g2      grad g3
   └──────────── all-reduce (average) ────────────┘
           every GPU now holds ḡ = (g0+g1+g2+g3)/4
           every GPU applies the same AdamW step
```

The communication step is an **all-reduce**: a collective operation (a communication step that every GPU in a group joins) where each GPU contributes a tensor and every GPU ends up with the sum (or average). The usual implementation is a **ring all-reduce**, which is really two phases: a **reduce-scatter** (each GPU ends up owning the fully summed version of one 1/N chunk) followed by an **all-gather** (each GPU broadcasts its finished chunk so everyone has everything). Each GPU sends about `2 · (N−1)/N` times the tensor size, which is nearly independent of how many GPUs there are. That's why ring all-reduce scales.

```formula
bytes_sent_per_GPU (ring all-reduce) ≈ 2 · (N − 1) / N · S
```

`N` is the number of GPUs in the group and `S` is the size of the tensor being reduced, in bytes.

Tiny worked example: 7B model, bf16 gradients (S = 14 GB), N = 8 GPUs. Each GPU sends 2 × 7/8 × 14 = **24.5 GB per step**. Inside an H100 server, NVLink is specced at 900 GB/s per GPU total (both directions), so call it ~450 GB/s each way, which gives an ideal lower bound of about 54 ms. Across servers, a 400 Gb/s InfiniBand NIC is 50 GB/s, so the same traffic takes about 0.49 s. That's a 9× difference just from which wire you use, and it's the theme of the whole hardware section. In practice frameworks bucket gradients and start all-reducing the last layers' gradients while the backward pass is still working on earlier layers, which hides much of this time behind compute.

Two things DP gives you for free, and one it doesn't:

- **Throughput** scales with the number of replicas, as long as communication hides behind compute.
- **Gradient accumulation** composes with it: each replica can run several micro-batches (small slices of its share of the batch, processed one after another) and sum gradients before the all-reduce. Tokens per optimizer step = DP size × micro-batch size × accumulation steps × sequence length. (Batch size and accumulation are covered in **How a model learns: loss, gradients, and optimizers**.)
- **Memory does not shrink at all.** Every replica still holds all 112 GB of model state. Plain DP of a 7B model on 80 GB GPUs is a non-starter. That redundancy is the thing ZeRO removes.

## ZeRO and FSDP: stop storing the same thing eight times {#zero-fsdp}

Look at plain DP on 8 GPUs and you'll see something silly: eight identical copies of Adam's `m` and `v`, eight identical master weights, eight identical gradient buffers. The ZeRO paper ("Zero Redundancy Optimizer") asks: what if each GPU kept only **its 1/N share** of that state, and fetched the rest just in time? It's like eight chefs who each keep one-eighth of the recipe book on their station, and read the page they need off a neighbour right before cooking that dish.

ZeRO does it in three cumulative stages:

- **Stage 1 (P_os), shard the optimizer state.** Each GPU keeps `m`, `v` and master weights for only 1/N of the parameters. After the gradient all-reduce, each GPU updates just its slice, then the updated bf16 weights are all-gathered so every GPU has the full model again.
- **Stage 2 (P_os+g), also shard the gradients.** Swap the all-reduce for a **reduce-scatter**: each GPU receives only the summed gradients for the slice it owns, which is all it needs for its update. The others are freed as soon as they're reduced.
- **Stage 3 (P_os+g+p), also shard the weights.** No GPU holds the full model, even in bf16. Right before a layer runs, its weights are **all-gathered** from all GPUs; right after, the gathered copy is freed. The same happens again in the backward pass.

### The worked memory budget

Here are per-GPU numbers for our 7B model (N = 7 × 10⁹, so 16N = 112 GB), using the ZeRO paper's formulas. N_d is the number of data-parallel GPUs sharing the state.

```formula
DP:       16·N
Stage 1:   4·N + 12·N / N_d
Stage 2:   2·N + 14·N / N_d
Stage 3:  16·N / N_d
```

| Per-GPU model state | Formula | N_d = 8 | N_d = 64 |
|---|---|---|---|
| Plain DP | 16N | 112 GB | 112 GB |
| ZeRO-1 | 4N + 12N/N_d | 28 + 10.5 = **38.5 GB** | 28 + 1.3 = **29.3 GB** |
| ZeRO-2 | 2N + 14N/N_d | 14 + 12.25 = **26.3 GB** | 14 + 1.5 = **15.5 GB** |
| ZeRO-3 | 16N/N_d | **14 GB** | **1.75 GB** |

Now add activations. With full activation checkpointing, one 4,096-token sequence costs about 1.7 GB (from the last section); without it, about 18 GB with fused attention. On one 8×H100 server:

```text
                    model state   + activations (no ckpt)   = peak     fits in 80 GB?
plain DP              112 GB          18 GB                   130 GB    no
ZeRO-1                38.5 GB         18 GB                   ~57 GB    yes, some headroom
ZeRO-2                26.3 GB         18 GB                   ~44 GB    yes
ZeRO-3                14 GB + ~0.9    18 GB                   ~33 GB    yes, room for B=2+
```

The "+ ~0.9" on ZeRO-3 is the transient: one layer of a 7B model is about 7B / 32 ≈ 219M parameters, or 0.44 GB in bf16, and you typically prefetch the next layer while computing the current one, so budget about two layers' worth gathered at once. Add a few GB for the CUDA context, communication buffers and allocator fragmentation in any real plan. These are planning numbers, not measurements.

Notice the pattern: stage 1 already takes the biggest bite, because the 12 bytes of optimizer state were the bulk of the bill. And note what ZeRO **doesn't** touch: activations. They still scale with the micro-batch you run on each GPU, which is why checkpointing and sharding get combined.

### What it costs in communication

The ZeRO paper's analysis: plain DP moves about 2Ψ elements per GPU per step (the reduce-scatter plus all-gather inside the all-reduce). Stages 1 and 2 move the **same 2Ψ**, because a reduce-scatter plus an all-gather of updated weights is the all-reduce, just split at a different point. Stage 3 moves **3Ψ, 1.5× the baseline**: an all-gather of the weights in the forward pass, another in the backward pass, and a reduce-scatter of the gradients. So stages 1–2 save memory essentially for free, and stage 3 trades 50% more traffic, plus a latency-sensitive gather before every layer, for the biggest saving.

### FSDP is PyTorch's ZeRO

**FSDP (Fully Sharded Data Parallel)** is PyTorch's native implementation of the same idea, and its docs say it was inspired by ZeRO stage 3. The mapping:

| FSDP (original API `ShardingStrategy`) | ZeRO equivalent | Behaviour |
|---|---|---|
| `NO_SHARD` | plain DP (like DDP) | all-reduce gradients |
| `SHARD_GRAD_OP` | ~stage 2 | weights stay gathered between forward and backward, resharded after backward |
| `FULL_SHARD` | stage 3 | all-gather before forward, free, all-gather again before backward, free |
| `HYBRID_SHARD` | stage 3 inside a node, replicate across nodes | expensive gathers stay on NVLink |

The newer `fully_shard` API (often called FSDP2) represents each parameter as a DTensor sharded along dimension 0, all-gathers in a pre-forward hook, and reduce-scatters gradients in a post-backward hook. Its `reshard_after_forward` flag is literally the stage-2-versus-stage-3 dial: `True` frees gathered weights after forward (less memory, one more all-gather), `False` keeps them (more memory, less traffic). Hybrid sharding is expressed with a 2D device mesh: replicate across one dimension, shard across the other.

> 🎬 **Animation — ZeRO stages on 4 GPUs:** four GPU columns, each with five stacked colour blocks (bf16 weights, grads, master, m, v) at full height, totalling 16 units each. Step 1 (stage 1): the three fp32 blocks on each GPU shrink to one quarter, each GPU keeping a different coloured quarter; a counter shows 16 → 7 units per GPU (4 + 12/4). Step 2 (stage 2): the gradient block also shrinks to a quarter; 7 → 5.5 (2 + 14/4). Step 3 (stage 3): the weight block shrinks to a quarter; 5.5 → 4 (16/4). Step 4: run a forward pass through layer 3 under stage 3: arrows from all four GPUs deliver their quarter of layer 3 to every GPU (all-gather), the full layer lights up, compute happens, then the gathered copy fades. Step 5: backward: gather again, then gradients flow out as a reduce-scatter so each GPU keeps only its quarter.

## Tensor parallelism: splitting the matmuls themselves {#tensor-parallelism}

ZeRO-3 shards the weights at rest but still **gathers a full layer to compute it**, and each GPU still processes its own full sequences, so activations don't shrink. When a single layer's weights or activations are too big, or you need more GPUs working on the same batch, you split the math inside the layer. That's **tensor parallelism (TP)**: several cooks working on the same dish at the same time.

The trick is the Megatron-LM layout for the MLP block. Remember a matrix multiply `X · A` can be split by **columns** of `A`: each GPU computes some of the output features, with no communication. Or by **rows** of `A`: each GPU computes a partial sum of every output feature, which then needs summing across GPUs.

```text
MLP:  Y = GeLU(X · A) · B          (rows of X = token positions, columns = features)

             split A by columns            split B by rows
X ──┬──► X·A₁ → GeLU → Y₁ ──► Y₁·B₁ ──┐
    │                                  ├──► all-reduce (sum) ──► Z
    └──► X·A₂ → GeLU → Y₂ ──► Y₂·B₂ ──┘
   GPU0 holds A₁, B₁      GPU1 holds A₂, B₂
```

Why this order? Splitting `A` by columns means each GPU has complete features, so the nonlinearity (GeLU, which is applied element by element) can run locally without any sync. Those per-GPU feature slices are exactly the row-slices of `B` needs, so the second matmul also runs locally and produces **partial sums**. One all-reduce at the end fixes it. Attention splits even more naturally: each GPU takes a subset of heads (the Q/K/V projections are column-split, so each head's attention is entirely local), and the output projection `W_O` is row-split, ending in one all-reduce.

The result, per the Megatron-LM paper: **two all-reduces in the forward pass and two in the backward pass per transformer layer**. Weights, gradients and optimizer state for those matrices are all divided by the TP size, so TP cuts model-state memory too.

Here's the catch. Those all-reduces carry **activations**, and they sit right on the critical path: the next layer can't start until they finish. Work out the size for our 7B shape with T = 4,096, B = 1:

```text
one all-reduce payload  = T · B · d · 2 bytes = 4,096 × 4,096 × 2 = 33.5 MB
per layer (2 fwd + 2 bwd)                     = 4 × 33.5 MB       = 134 MB
per sequence, 32 layers                                           ≈ 4.3 GB
```

That's 4.3 GB of blocking traffic **for every 4,096-token sequence**, and it scales with tokens processed. Compare DP, which moves its ~24 GB once per optimizer step no matter how many sequences you run. This is why TP lives on the fastest links you have. The Megatron-LM paper kept its tensor-parallel groups inside one server (NVSwitch at 300 GB/s per GPU in that DGX-2H setup, versus 100 GB/s per server across InfiniBand), and the 2021 Megatron scaling paper's takeaway is to use tensor parallelism up to the number of GPUs in a server and pipeline parallelism across servers. TP degree is usually 2, 4 or 8, and rarely crosses a node boundary.

A refinement worth naming: **sequence parallelism** (Megatron's term) notices that LayerNorm and dropout, which sit between the TP regions, are still computed redundantly on every TP rank over the full sequence. It splits those along the sequence dimension and replaces each all-reduce with a reduce-scatter plus an all-gather, which is the same total traffic, and the paper reports it roughly halves activation memory.

> 🎬 **Animation — Megatron MLP split:** show X as a 4 × 4 grid (4 token rows, 4 feature columns). A is 4 × 8, drawn split into a left half (blue, GPU0) and right half (green, GPU1). Step 1: X is copied to both GPUs. Step 2: each GPU multiplies X by its half of A, producing a 4 × 4 block. Step 3: GeLU is applied locally, marked with a "no communication" tick. Step 4: B (8 × 4) is split into top rows (blue) and bottom rows (green); each GPU multiplies and produces a full 4 × 4 partial result. Step 5: the two partial 4 × 4 matrices are added element by element by an "all-reduce" arrow, producing Z on both GPUs. Show one worked cell: 1.5 (GPU0) + 0.7 (GPU1) = 2.2.

## Pipeline parallelism: an assembly line of layers {#pipeline-parallelism}

**Pipeline parallelism (PP)** splits the model by depth: GPU 0 owns layers 1–8, GPU 1 owns layers 9–16, and so on. Each GPU (a **stage**) only stores its own layers, and the only traffic is the activation tensor handed from one stage to the next (and its gradient handed back). It's an assembly line where each station does one part of every dish.

The naive version is terrible, though. If you push one batch through, stage 1 works while stages 2–4 wait, then stage 2 works while the others wait... Only one GPU is busy at a time. The fix from GPipe is to cut the batch into **micro-batches** so the stages overlap, then accumulate gradients across all micro-batches and apply one synchronous update.

```text
p = 4 stages, m = 8 micro-batches  (F = forward, B = backward; one cell = one micro-batch step)

stage 0: F1 F2 F3 F4 F5 F6 F7 F8 .  .  .  B8 B7 B6 B5 B4 B3 B2 B1
stage 1: .  F1 F2 F3 F4 F5 F6 F7 F8 .  .  .  B8 ...
stage 2: .  .  F1 F2 F3 F4 F5 F6 F7 F8 .  .  .  B8 ...
stage 3: .  .  .  F1 F2 F3 F4 F5 F6 F7 F8 B8 ...
         └─fill─┘                         └─drain─┘   dots = idle "bubble"
```

The idle time while the line fills and drains is the **pipeline bubble**. With `p` stages and `m` micro-batches:

```formula
bubble / ideal compute time = (p − 1) / m          (Narayanan et al., GPipe and 1F1B schedules)
bubble / total time         = (p − 1) / (m + p − 1) (the form in the GPipe paper)
```

Both say the same thing. The first compares idle time with the useful work; the second is the fraction of the wall-clock step spent idle.

Checked numbers: p = 4, m = 8 gives 3/8 = 37.5% extra time, meaning 3/11 ≈ 27% of the step is idle. Raise m to 32 and it's 3/32 ≈ 9.4% extra (3/35 ≈ 8.6% idle). GPipe's rule of thumb is that the overhead is almost negligible when m ≥ 4p. The catch is that more micro-batches means either smaller micro-batches (less efficient matmuls) or a bigger global batch (which changes the optimisation).

Two schedule improvements you should be able to name:

- **1F1B (one forward, one backward).** Instead of all forwards then all backwards, each stage starts a micro-batch's backward as soon as it can. Same bubble, but each stage only has to hold activations for about `p` in-flight micro-batches instead of all `m`, which is a big memory win.
- **Interleaved stages.** Give each GPU `v` smaller, non-contiguous chunks of layers (GPU 0 gets layers 1–2 and 9–10, and so on). The bubble drops to `(p − 1) / (v · m)`, e.g. 3/16 ≈ 19% for v = 2, m = 8, at the price of `v` times more stage-to-stage messages.

Pipeline traffic is light: one activation tensor per micro-batch per boundary (for our shape, 33.5 MB forward and the same backward), and only between neighbours. That makes PP the natural dimension to put across the slower inter-node links. Its failure modes are different from TP's: **unbalanced stages** (the first stage also carries the embedding table, the last carries the output projection and loss, so an even layer split isn't an even time split), and the fact that the slowest stage sets the pace for everyone.

> 🎬 **Animation — filling the pipeline:** a 4-row by 19-column grid, one row per stage, time running left to right. Step 1: animate naive pipelining with m = 1: a single coloured block travels diagonally down through the stages and back up, and 75% of the grid is grey. Step 2: replay with m = 8 micro-batches in 8 colours: forwards cascade diagonally, backwards cascade back; the grey triangles at the start and end are labelled "fill" and "drain", with a counter "idle = 3/11 ≈ 27%". Step 3: switch to 1F1B: the same bubble, but a side gauge of "activations held on stage 0" peaks at 4 micro-batches instead of 8. Step 4: slider m = 32 shrinks the grey to 8.6%.

## Long sequences and experts: two more ways to split {#sequence-and-experts}

The three classic dimensions (data, tensor, pipeline) are often called **3D parallelism**. Modern runs add two more.

**Context parallelism (CP)** splits each sequence along its length across GPUs, which matters when you train with very long contexts and activations for a single sequence won't fit even with TP. Most of a transformer layer is per-token (norms, MLP), so splitting tokens across GPUs is easy. Attention is the hard part: a token's query needs keys and values from every earlier token. The usual fix passes chunks of K and V around a ring of GPUs so each GPU sees every chunk once while computing its local queries, overlapping the transfer with compute. Llama 3's training used context parallelism as one of its four dimensions. The intuition to carry: CP trades activation memory for K/V traffic, and it's only worth it when the sequence is long.

**Expert parallelism (EP)** applies to mixture-of-experts models, where each layer has many feed-forward "experts" and a router sends each token to only a few of them (top-k). EP puts different experts on different GPUs and **moves the tokens to the weights** instead of moving weights to the tokens. That needs an **all-to-all**: every GPU sends some of its tokens to every other GPU, then gets the results back. The sizes add up fast. With T = 8,192 tokens in a layer, top-2 routing, d = 4,096 and 2-byte activations, the dispatch alone is 8,192 × 2 × 4,096 × 2 bytes = **128 MiB** for the group, before the return trip or the backward pass. All-to-all is also sensitive to imbalance: if the router sends too many tokens to one expert, that GPU becomes the straggler. We'll go through routers, load balancing and all-to-all in detail in **Mixture of experts: more parameters, same compute per token**.

One practical warning: every dimension needs an explicit **process group** (the set of **ranks**, meaning worker processes, usually one per GPU, that communicate for that axis). Some frameworks carve expert-parallel groups out of the data-parallel dimension, so multiplying TP × PP × DP × EP blindly can double-count GPUs. Draw the actual rank map.

## Putting it on real hardware: rank maps, links and MFU {#hardware-mapping}

A cluster isn't a flat pool of GPUs. It's islands of very fast links connected by a slower sea.

```text
      node 0 (8 GPUs)                        node 1 (8 GPUs)
 ┌─────────────────────────────┐        ┌─────────────────────────────┐
 │ G0 G1 G2 G3 G4 G5 G6 G7     │        │ G0 G1 G2 G3 G4 G5 G6 G7     │
 │  └──── NVLink/NVSwitch ───┘ │        │  └──── NVLink/NVSwitch ───┘ │
 │      ~900 GB/s per GPU      │        │      ~900 GB/s per GPU      │
 └──────┬──────────────────────┘        └──────┬──────────────────────┘
        │ 8 × 400 Gb/s NICs (≈50 GB/s each)    │
        └────────── InfiniBand / Ethernet switch fabric ──────────
```

Those figures come from NVIDIA's H100 and DGX H100 specs: 900 GB/s of NVLink per GPU, and eight ConnectX-7 cards at up to 400 Gb/s per server. Per GPU, that's roughly an order of magnitude between the inside of a node and the network outside it. So you place each parallelism dimension by **how much it sends and how often it blocks**:

| Dimension | What it splits | Collective | How often | Where it goes |
|---|---|---|---|---|
| Tensor (TP) | matrices within a layer | all-reduce of activations | 4× per layer, on the critical path | inside a node (NVLink) |
| Context (CP) | a sequence's tokens | K/V ring passes | every attention layer | inside a node if possible |
| Expert (EP) | experts across GPUs | all-to-all of tokens | 2× per MoE layer (+ backward) | fast links; watch bisection bandwidth |
| Pipeline (PP) | layers into stages | point-to-point activations | once per micro-batch per boundary | across nodes |
| Data (DP/FSDP) | the batch | all-reduce / reduce-scatter + all-gather of grads/weights | once per step (FSDP: per layer) | outermost; overlapped with compute |

This is the ordering Llama 3 describes for its 16K-GPU run, with TP innermost, then CP, then PP, then DP outermost, so the highest-bandwidth traffic stays within a server.

### A worked layout

Take 64 GPUs = 8 servers × 8 GPUs, with TP = 4, PP = 2 and DP = 8 (4 × 2 × 8 = 64). One TP group is 4 GPUs sharing NVLink. One pipeline replica spans 4 × 2 = 8 GPUs, which here is one server. The 8 data-parallel replicas span the 8 servers, so gradient sync crosses the network, but only once per step and overlapped. If each replica runs micro-batches of 2 sequences, accumulates 16 of them and uses T = 2,048, one optimizer step covers 8 × 2 × 16 × 2,048 = **524,288 tokens**. Notice that TP and PP don't multiply the token count, because those GPUs collaborate on the *same* sequences. Only the DP replicas see different data. (Padding and masked positions can reduce the tokens that actually contribute to the loss.)

### MFU: are the GPUs actually working?

**Model FLOPs Utilization (MFU)**, as defined in the PaLM paper, is observed throughput divided by what the hardware could do at peak, counting only the FLOPs the model *requires*. Recomputed activations don't count. (The version that does count them is **HFU**, hardware FLOPs utilization.) For a dense transformer, training costs about 6 FLOPs per parameter per token (2 forward, 4 backward; this is the `C ≈ 6·N·D` estimate from **Pretraining: data, compute, and scaling laws**).

```formula
MFU = (6 · N · tokens_per_second) / (num_GPUs · peak_FLOPs_per_GPU)
```

`N` is the parameter count, `tokens_per_second` is measured end-to-end training throughput, and peak is the **dense** bf16 tensor-core rate. Spec sheets often headline a with-sparsity number. For H100 SXM that's 1,979 TFLOPS, so dense is about 989.

Illustrative: a 7B model on 64 H100s processing 400,000 tokens/s gives 6 × 7×10⁹ × 4×10⁵ = 1.68×10¹⁶ FLOP/s, divided by 64 × 9.89×10¹⁴ = 6.33×10¹⁶, so MFU ≈ **26.5%**. For calibration from published runs: PaLM 540B reported 46.2% MFU, the Megatron 1T-parameter run reported 52% of peak per GPU on 3,072 GPUs, and Llama 3 405B reported 38–43% BF16 MFU on up to 16K H100s. Nobody gets near 100%: communication that isn't hidden, bubbles, stragglers (one slow GPU stalls every synchronous collective), data loading and small kernels all eat into it. When MFU is low, find out whether the bottleneck is compute, network, input pipeline or imbalance *before* adding GPUs. More GPUs only help if they turn into useful progress.

> 🎬 **Animation — rank map:** draw 2 servers × 8 GPUs as two rows of 8 squares. Step 1: colour groups of 4 adjacent GPUs as TP groups (4 colours), with thick NVLink bars inside each group, labelled "all-reduce ×4 per layer". Step 2: overlay PP: the two TP groups in each server become stage 0 and stage 1, with a thin arrow between them labelled "33 MB per micro-batch". Step 3: overlay DP: matching GPUs across the two servers connected by dashed network lines labelled "gradient reduce-scatter once per step, overlapped". Step 4: show a "bad" map where a TP group straddles both servers; its all-reduce bars turn red and a timer shows ~9× slower per collective.

## When something breaks: checkpoints, failures and restarts {#checkpointing}

At scale, failure isn't an edge case, it's the weather. Llama 3's paper reports 466 job interruptions during a 54-day snapshot of pretraining on up to 16K H100s, 419 of them unexpected, and about 78% of the unexpected ones attributed to hardware issues, while still achieving over 90% effective training time. Do the arithmetic: 54 days × 24 hours ÷ 419 ≈ **one unexpected interruption every ~3.1 hours**. The intuition: if each GPU fails independently, the cluster's mean time between failures is roughly one GPU's divided by the number of GPUs. A synchronous job is only as reliable as its least lucky rank.

### What a checkpoint must contain

A **resumable training checkpoint** is much more than the weights:

- model weights and **optimizer state** (the fp32 master weights plus Adam `m` and `v`; restarting with fresh moments is a different experiment, even if the loss looks fine for a while);
- learning-rate scheduler position, step and token counters;
- random-number-generator states (for dropout and data shuffling);
- the data loader's position in the token stream, so you don't repeat or skip data;
- the exact config, tokenizer and data manifest.

For our 7B model, the fp32 master weights plus `m` and `v` are 12 × 7B = **84 GB** (the bf16 weights can be re-derived from the master copy), versus 14 GB for a weights-only bf16 export for inference. Mixing those up is a classic mistake.

With ZeRO/FSDP, no rank holds the whole state, so each rank writes its own shard in parallel. PyTorch Distributed Checkpoint (DCP) saves from multiple ranks in parallel and supports **load-time resharding**, so a checkpoint saved on one topology can be loaded onto another (useful when you restart with fewer nodes after a failure). It also offers `async_save`, which copies the state to CPU memory and writes it in a background thread so training can continue, and a `Stateful` protocol for your own objects (data loader, counters) to join the checkpoint. Async only helps if the staging memory and I/O contention stay bounded.

### How often to checkpoint

Checkpointing costs a stall of `C` each time; failures destroy, on average, half an interval of work. Say you checkpoint every `τ` minutes and the mean time between failures is `M`. Per unit of time you lose `C/τ` to checkpointing and about `τ / (2M)` to redone work. Minimise the sum and you get:

```formula
τ_optimal ≈ √(2 · C · M)
```

`τ` is the checkpoint interval, `C` the time training stalls per checkpoint, and `M` the mean time between failures.

Worked example (illustrative): M ≈ 186 minutes (the Llama 3 rate above) and C = 1 minute of stall gives τ ≈ √(2 × 1 × 186) ≈ **19 minutes**, so each failure costs about 10 minutes of redone work plus restart time. Make checkpoints cheaper (async, sharded) and the optimal interval shrinks with the square root of `C`.

### Make checkpoints trustworthy

A checkpoint you can't trust is worse than none. The robust pattern works like a database commit. Write all shards under a new checkpoint ID, verify that every shard is present and checksums match, then atomically publish a small "committed" manifest. Loaders ignore anything without a manifest, so a job that dies mid-write never gets resumed from a half-written checkpoint. And before the expensive run starts, **test it**: kill a job mid-checkpoint, restore onto a different number of nodes, and confirm the loss curve continues rather than jumping.

> 🎬 **Animation — failures versus checkpoint interval:** a horizontal timeline of 12 hours with training progress as a rising line. Step 1: red lightning bolts strike at random times averaging every ~3 hours; with checkpoints every 2 hours, each bolt drops the line back to the last checkpoint marker, and the lost area is shaded red. Step 2: replay with checkpoints every 19 minutes (dense tick marks, each costing a small 1-minute flat step); the red areas shrink. Step 3: a side chart plots total waste = checkpoint cost + lost work against interval, a U-shaped curve with its minimum marked at ~19 minutes.

# Interview

## Question

You need to pretrain a 70B-parameter dense transformer on a cluster of 8-GPU H100 servers (80 GB per GPU). Walk me through how you'd decide the parallelism layout, and show me why you can't just use data parallelism.

## Answer

Start with the memory bill. Mixed-precision AdamW costs about 16 bytes per parameter: 2 for bf16 weights, 2 for bf16 gradients, and 12 for fp32 master weights plus Adam's two moments. For 70B that's 1,120 GB of model state before a single activation. Plain data parallelism replicates all of it on every GPU, so it's off the table by a factor of 14.

First lever: shard the model state. With ZeRO-3/FSDP full sharding over 64 GPUs, the per-GPU state is 1,120 / 64 = 17.5 GB. That fits, but every layer's weights get all-gathered twice per step, and I still need activations for whatever micro-batch each GPU runs. At 70B scale the width is large (d around 8,192), so per-layer activations and TP-able matmuls are big. I'd add **tensor parallelism of 8 inside each server** over NVLink, because Megatron-style TP costs two all-reduces forward and two backward per layer, on the critical path, and that traffic scales with tokens. That has to stay on the ~900 GB/s links, not the ~50 GB/s-per-GPU network.

If TP=8 still leaves too much per GPU, or the model has too many layers to gather comfortably, I'd add **pipeline parallelism across servers**, say 4 stages, because stage-to-stage traffic is one activation tensor per micro-batch, which the network handles fine. I'd use a 1F1B or interleaved schedule with m ≥ 4p micro-batches to keep the bubble under about 10%, and balance stages so the embedding and loss layers don't make the first and last stages the stragglers. Data parallelism (or sharded DP) goes outermost, with gradient reduce-scatter overlapped with backward. The DP size is set by the global batch I want, in tokens per step.

Then activations: use selective or full activation checkpointing depending on headroom, and sequence parallelism alongside TP to split the LayerNorm/dropout activations. If the target context is very long, add context parallelism.

Finally, I'd validate with numbers, not vibes: a rank map listing each collective's participants, bytes and frequency; MFU measured with 6·N·tokens/s against dense peak (40%-ish is respectable at scale); and a checkpoint/restart plan sized to the cluster's failure rate, with sharded async checkpoints, a commit manifest and a tested restore onto a different node count.

## Follow-ups

- What exactly is the communication cost of ZeRO-3 compared with plain DP, and when would you prefer ZeRO-2 or hybrid sharding instead?
- Why does Megatron split the first MLP matrix by columns and the second by rows? What would go wrong the other way around?
- Your MFU is 25% and the profiler shows GPUs idle in bursts every step. How do you tell a pipeline bubble from a straggler from a data-loader stall?
- How does gradient accumulation interact with pipeline micro-batches and with the global batch size?
- A node dies at hour 300. Walk through the restart, including what happens if you come back with one fewer node.

# Pitfalls

- **"7B is 14 GB, so it fits on an 80 GB GPU."** That's inference weights only. Training with mixed-precision AdamW needs about 16 bytes per parameter (112 GB) plus activations.
- **Thinking data parallelism saves memory.** Plain DP replicates the full model state on every GPU. It adds throughput, not capacity; only sharding (ZeRO/FSDP), TP or PP reduce per-GPU state.
- **Treating ZeRO stages as free.** Stages 1–2 keep DP's communication volume, but stage 3 is 1.5× and adds a gather before every layer that must be prefetched to hide latency. And no ZeRO stage reduces activation memory.
- **Running tensor parallelism across nodes.** TP's all-reduces carry activations four times per layer on the critical path. Over inter-node links they can dominate step time; keep TP within the NVLink domain.
- **Using too few micro-batches in a pipeline.** With p = 4 and m = 4, the bubble adds 75% extra time. Aim for m ≥ 4p, and remember stages must be time-balanced, not just layer-balanced.
- **Multiplying TP × PP × DP to get tokens per step.** Only DP replicas see different data; tokens per step = DP × micro-batch × accumulation × sequence length.
- **Saving weights only and calling it a checkpoint.** Without optimizer moments, RNG state and data-loader position, a "resume" is a different experiment.
- **Quoting MFU against the sparsity peak, or counting recomputation.** MFU uses dense peak and only model-required FLOPs; including recompute gives HFU.

# Checklist

- Compute the model-state memory of any model with mixed-precision AdamW (16 bytes/param) and say whether it fits.
- Estimate activation memory per layer and explain what activation checkpointing saves and costs (~33% more compute for full recompute).
- Produce the per-GPU memory for plain DP and ZeRO stages 1, 2 and 3 for a given model and GPU count.
- Explain ring all-reduce as reduce-scatter plus all-gather, and estimate its traffic per GPU.
- Draw the Megatron MLP split and state how many all-reduces per layer TP needs.
- Compute the pipeline bubble for given p, m and interleaving v.
- Place TP, CP, EP, PP and DP onto NVLink versus network links and justify the order.
- Compute MFU from tokens/s, parameter count and dense peak FLOPs.
- List what a resumable checkpoint contains and size a checkpoint interval from a failure rate.

# Sources

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) — The 16Ψ mixed-precision Adam accounting (K = 12), the per-device formulas for stages P_os, P_os+g, P_os+g+p, communication volume 2Ψ vs 3Ψ, and activation-memory examples.
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053) — Column-then-row MLP split, head-wise attention split, two all-reduces forward and two backward per layer, and in-server model-parallel groups over NVSwitch.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473) — Composing TP, PP and DP; bubble (p−1)/m and interleaved (p−1)/(v·m); 1F1B memory; TP within a server and PP across servers; 1T parameters on 3,072 GPUs at 52% of peak.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al., 2018)](https://arxiv.org/abs/1811.06965) — Micro-batch pipelining with synchronous gradient accumulation, bubble O((K−1)/(M+K−1)) negligible when M ≥ 4K, and re-materialisation.
- [Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198) — Per-layer activation memory sbh(34 + 5as/h), sequence parallelism, and selective recomputation savings and overhead.
- [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html) — ShardingStrategy options (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD) and FSDP's inspiration from ZeRO stage 3.
- [PyTorch fully_shard (FSDP2) documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) — DTensor dim-0 sharding, pre-forward all-gather, post-backward reduce-scatter, reshard_after_forward, and HSDP via a 2D device mesh.
- [PyTorch Distributed Checkpoint (DCP) documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — Parallel multi-rank save/load, load-time resharding, async_save and the Stateful protocol.
- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) — Definition of MFU versus HFU and PaLM 540B's 46.2% MFU.
- [The Llama 3 Herd of Models (Llama Team, Meta, 2024)](https://arxiv.org/abs/2407.21783) — 4D parallelism order (TP, CP, PP, DP), 38–43% BF16 MFU on up to 16K H100s, and 466 interruptions (419 unexpected) in 54 days.
- [NVIDIA H100 GPU product page](https://www.nvidia.com/en-us/data-center/h100/) — H100 SXM 80 GB, 900 GB/s NVLink, BF16 tensor-core peak quoted with sparsity.
- [NVIDIA DGX H100 User Guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) — 8 GPUs per system, NVLink/NVSwitch, and eight ConnectX-7 cards at up to 400 Gb/s for the compute fabric.

# Flashcards

## bytes-per-param-mixed-adam

**Q:** How many bytes per parameter does mixed-precision AdamW training need for model state, and what are they?

About 16: 2 for bf16 weights, 2 for bf16 gradients, and 12 for the fp32 optimizer state (4 for master weights, 4 for Adam's first moment `m`, 4 for the second moment `v`). The master copy exists because tiny updates get rounded away in 16-bit. A 7B model therefore needs about 112 GB of model state before any activations, which is why it can't train on one 80 GB GPU even though its bf16 weights are only 14 GB.

## activation-memory

**Q:** What are activations in training, and why does their memory grow with batch and sequence length?

They're the intermediate forward-pass values (inputs to each matmul, outputs of nonlinearities) that the backward pass needs to compute gradients. Every token in every sequence produces them at every layer, so memory scales roughly with B × T × d × L. Naive attention adds T × T score matrices per head, which is why fused attention kernels that never materialise them matter so much at long context.

## activation-checkpointing

**Q:** What does activation checkpointing trade, and roughly how much?

It keeps only each layer's input and reruns that layer's forward pass during backward to regenerate the rest. That cuts activation memory dramatically (in the 7B example, ~18 GB down to ~1–2 GB per sequence) for roughly one extra forward pass, i.e. ~33% more compute (3 units to 4). Selective recomputation only redoes the cheap-to-compute, memory-heavy attention parts, getting most of the saving for a few percent of compute.

## data-parallel-allreduce

**Q:** In data parallelism, what gets communicated, when, and does it save memory?

Each replica computes gradients on its slice of the batch; then an all-reduce averages gradients so every replica applies the identical update. It's once per optimizer step, and can overlap with the backward pass by bucketing gradients. It saves no memory: every GPU still holds the full weights, gradients and optimizer state. It buys throughput only.

## ring-allreduce

**Q:** Why does ring all-reduce scale well, and what is it made of?

It's a reduce-scatter (each GPU ends up owning the full sum of one 1/N chunk) followed by an all-gather (each GPU shares its finished chunk). Each GPU sends about 2·(N−1)/N times the tensor size, which is nearly constant as N grows, so adding GPUs doesn't blow up per-GPU traffic. Latency grows with N, though, which is why hierarchical or tree variants exist.

## zero-stages

**Q:** What do ZeRO stages 1, 2 and 3 shard, and what's the per-GPU memory for each?

Stage 1 shards optimizer state (4N + 12N/N_d bytes), stage 2 also shards gradients (2N + 14N/N_d), and stage 3 also shards the weights themselves (16N/N_d). For 7B on 8 GPUs: 112 GB for plain DP, then 38.5, 26.25 and 14 GB. Stage 1 takes the biggest bite because optimizer state is 12 of the 16 bytes.

## zero-communication

**Q:** How does ZeRO's communication compare with plain data parallelism?

Stages 1 and 2 move the same ~2Ψ per GPU as a DP all-reduce, because a reduce-scatter of gradients plus an all-gather of updated weights is the all-reduce, just split at a different point. Stage 3 moves ~3Ψ (1.5×): an all-gather of weights in forward, another in backward, and a reduce-scatter of gradients. The gathers also sit before every layer, so they must be prefetched to hide latency.

## fsdp-vs-zero

**Q:** How does PyTorch FSDP map onto ZeRO, and what does HYBRID_SHARD do?

FULL_SHARD corresponds to ZeRO-3 (gather before forward and backward, free after each), SHARD_GRAD_OP to roughly ZeRO-2 (weights stay gathered between forward and backward), and NO_SHARD to plain DDP. HYBRID_SHARD fully shards within a node and replicates across nodes, so the frequent all-gathers and reduce-scatters stay on NVLink and only a gradient all-reduce crosses the network. In FSDP2's fully_shard, reshard_after_forward is the same memory-versus-traffic dial.

## megatron-tensor-parallel

**Q:** How does Megatron tensor parallelism split an MLP block, and why that way?

The first matrix is split by columns, so each GPU produces complete output features and can apply GeLU locally without communication. The second matrix is split by rows, matching those feature slices, so each GPU produces a partial sum; one all-reduce adds them. Attention splits by heads, with the output projection row-split. Total: two all-reduces forward and two backward per layer.

## tp-placement

**Q:** Why is tensor parallelism kept inside a node?

Its all-reduces carry activations (T·B·d·2 bytes each, e.g. 33.5 MB for T = 4,096, d = 4,096), four per layer, on the critical path, and the traffic scales with every token processed. Inside a node NVLink gives ~900 GB/s per GPU; across nodes a GPU typically gets ~50 GB/s. Crossing that boundary can make TP collectives an order of magnitude slower and dominate step time.

## pipeline-bubble

**Q:** What is the pipeline bubble, and how big is it for p = 4 stages and m = 8 micro-batches?

It's idle time while the pipeline fills and drains. Bubble relative to useful compute is (p − 1)/m = 3/8 = 37.5%, meaning 3/11 ≈ 27% of the step is idle. More micro-batches shrink it (m ≥ 4p is GPipe's rule of thumb), and interleaving v chunks per GPU divides it by v at the cost of more messages. 1F1B scheduling doesn't shrink the bubble but caps in-flight activations at about p micro-batches.

## parallelism-placement-order

**Q:** In what order do you map parallelism dimensions onto a cluster, and why?

From most to least communication-hungry: tensor (and context) parallelism innermost on NVLink, then expert and pipeline parallelism, with data parallelism outermost across the network. TP talks several times per layer on the critical path; PP sends one activation per micro-batch to a neighbour; DP syncs once per step and overlaps with backward. Llama 3 used exactly TP, CP, PP, DP from inner to outer.

## mfu

**Q:** How do you compute MFU, and what's the common mistake?

MFU = 6 · N · tokens_per_second / (num_GPUs · dense peak FLOP/s). It counts only the FLOPs the model requires, so activation recomputation is excluded (including it gives HFU). Common mistakes: using the with-sparsity peak (H100's 1,979 TFLOPS headline versus ~989 dense), or forgetting that published large runs land around 40–50%, not near 100%.

## training-checkpoint-contents

**Q:** What must a resumable training checkpoint contain, and how do you pick its interval?

Weights plus full optimizer state (master weights, Adam m and v), scheduler and step/token counters, RNG states, data-loader position, and the config/tokenizer/data manifest. For 7B that's ~84 GB of fp32 state versus a 14 GB bf16 weights-only export. The interval balances checkpoint cost C against expected lost work: τ ≈ √(2·C·M) for mean time between failures M, e.g. ~19 minutes for C = 1 minute and M ≈ 3 hours.
