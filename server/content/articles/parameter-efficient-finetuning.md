---
{
  "slug": "parameter-efficient-finetuning",
  "title": "Fine-tuning on a budget: LoRA and QLoRA",
  "category": "alignment",
  "summary": "When to fine-tune at all, how LoRA learns a tiny low-rank patch instead of rewriting the model, how QLoRA squeezes the frozen base into 4 bits, and how to budget memory, build the data, evaluate, and ship adapters.",
  "difficulty": "Advanced",
  "minutes": 30,
  "prerequisites": ["transformer-foundations", "optimization-generalization", "post-training-alignment"],
  "learningObjectives": [
    "Decide between prompting, retrieval, and fine-tuning by classifying what is actually failing.",
    "Derive the LoRA update, count its trainable parameters, and explain rank, alpha, initialization, and target modules.",
    "Explain what QLoRA stores in 4 bits, what it computes in bf16, and why gradients still flow through the frozen base.",
    "Build a back-of-envelope training memory budget for full fine-tuning, LoRA, and QLoRA on an 8B-class model.",
    "Audit a fine-tuning dataset for template, loss-mask, truncation, and split-leakage bugs.",
    "Choose between merging an adapter and serving it separately, including multi-LoRA serving."
  ]
}
---

# Sections

## Before you train anything: is fine-tuning even the right tool? {#prompt-rag-or-finetune}

**Fine-tuning** means taking a model someone already trained and continuing to train it on your own examples, so its weights drift toward the behaviour you want. It's powerful, and it's also the most expensive and least reversible of your three options. So the first senior move is to ask what's actually broken.

Here's the analogy I like. You've hired a very smart contractor.

- **Prompting** is giving them better instructions. Cheap, instant, reversible.
- **RAG** (retrieval-augmented generation: look up relevant documents at request time and paste them into the prompt) is handing them the right binder before each job. We'll go through it properly in **RAG: giving the model the right evidence**.
- **Fine-tuning** is sending them on a training course. It changes *habits*, not the contents of the binder.

That last line is the whole decision. Fine-tuning is good at teaching **behaviour**: a house style, a strict output convention, a domain's vocabulary, a repeated decision pattern ("tickets that mention chargebacks go to the billing queue"). It's bad at teaching **facts that change**. If your refund policy changes weekly, baking it into weights means retraining weekly, and you lose the ability to say *which document* an answer came from, or to delete a record when someone asks.

A practical exercise interviewers love: take 50–100 real failures and label each one.

| Failure type | Example | First fix to try |
|---|---|---|
| Missing or stale evidence | Quotes last quarter's policy | RAG / better retrieval |
| Ambiguous instructions | Unsure whether to include tax | Rewrite the prompt, add examples |
| Output-format violation | Invalid JSON | Constrained decoding (only allow tokens that keep the output valid against a schema) |
| Consistent wrong *behaviour* | Valid JSON, wrong escalation category, even with good prompt + evidence | **Fine-tune** |

Fix the cheap rows first, re-measure, and only the leftover behavioural gap is what a fine-tune has to close. That gives you a baseline the adapter has to beat, and a way to defend the spend. RAG and fine-tuning also combine happily: an adapter can learn *how to use* retrieved evidence (cite it, abstain when it's missing) while the index keeps supplying *what* the evidence says.

> 🎬 **Animation — which lever fixes which failure:** show a pile of 20 failure cards, each coloured by type (8 blue "missing evidence", 4 yellow "ambiguous prompt", 3 grey "bad JSON", 5 red "wrong behaviour"). Step 1: a "RAG" funnel absorbs the 8 blue cards. Step 2: a "prompt rewrite" funnel absorbs the 4 yellow ones. Step 3: a "constrained decoding" funnel absorbs the 3 grey ones. Step 4: only the 5 red cards remain, and they slide into a box labelled "fine-tune target: 5/20 = 25% of original failures".

## Why full fine-tuning is so expensive {#full-finetuning-memory}

To see why LoRA exists, count what full fine-tuning keeps in GPU memory. Training uses an optimizer like **AdamW**, which keeps two running averages per trainable weight (roughly, the average gradient and the average squared gradient). We cover it properly in **How a model learns: loss, gradients, and optimizers**. With standard **mixed precision** (compute in bf16, a 16-bit float format, but keep an fp32 "master" copy so tiny updates don't round away), the usual accounting per trainable parameter is:

| Item | Bytes per parameter |
|---|---|
| bf16 weights (used in forward/backward) | 2 |
| bf16 gradients | 2 |
| fp32 master copy of weights | 4 |
| fp32 Adam first moment (m) | 4 |
| fp32 Adam second moment (v) | 4 |
| **Total** | **16** |

For an 8-billion-parameter model: 8 × 10⁹ × 16 bytes = 128 GB, **before** a single activation is stored. That's more than one 80 GB GPU, so you're into sharding weights and optimizer state across several GPUs, which we cover in **Distributed training: fitting a training run onto a cluster**.

Notice where the bytes go: only 2 of the 16 are the model itself. The other 14 exist *because the weight is trainable*. A frozen weight needs its 2 bytes (or fewer, as we'll see) and nothing else. That's the lever: **make almost nothing trainable**, and the gradient + optimizer bill nearly disappears.

The other cost is the artifact. Full fine-tuning produces a full new copy of the model per task. Ten tasks on an 8B model at bf16 means ten 16 GB checkpoints, and ten models to keep in GPU memory if you want to serve them all.

```text
 Full fine-tune, 8B params, mixed-precision AdamW (no activations yet)

 weights  ██                              16 GB
 grads    ██                              16 GB
 master   ████                            32 GB
 Adam m   ████                            32 GB
 Adam v   ████                            32 GB
          ───────────────────────────────
                                         128 GB
```

## LoRA: learn a small patch instead of rewriting the matrix {#lora-math}

Most of a transformer's parameters live in **linear projections**: weight matrices that map one vector to another. The attention projections `W_Q, W_K, W_V, W_O` and the three MLP matrices in each block are all linear layers (see **The transformer block: assembling the full model**). Fine-tuning changes a weight matrix `W₀` into `W₀ + ΔW`. The **LoRA** (Low-Rank Adaptation) bet is that `ΔW`, the *change* you need for one task, is simple, even though `W₀` isn't.

Analogy: a mixing desk with 4,096 sliders. Full fine-tuning lets you nudge every slider independently. LoRA gives you, say, 16 master knobs, each wired to move all the sliders together in its own fixed pattern. You lose some freedom, but most real adjustments ("a bit more bass everywhere") are combinations of a few patterns.

Concretely, freeze `W₀` and learn two thin matrices:

```formula
h = W₀x + (α / r) · B(Ax)
A ∈ ℝ^(r × d_in),   B ∈ ℝ^(d_out × r),   r ≪ min(d_in, d_out)
```

- `x` is the input vector (one token's features, length `d_in`), `h` the output (length `d_out`).
- `A` squashes `x` down to just `r` numbers; `B` expands those `r` numbers back to `d_out`.
- `r` is the **rank**: how many independent directions the update can push the output in.
- `α` (alpha) is a number you pick; `α / r` scales the whole side branch.

Since `BA` is a `d_out × d_in` matrix built from only `r` columns and `r` rows, its rank is at most `r`. The *update* is low-rank. `W₀` and the final `W₀ + (α/r)BA` are not. That's a favourite interview gotcha.

```text
        x (d_in)
        │
   ┌────┴──────────────┐
   │                   │
   ▼                   ▼
 ┌──────────┐      ┌──────┐   A: r × d_in   (trainable)
 │    W₀    │      │  A   │   squeeze to r numbers
 │  frozen  │      └──┬───┘
 │d_out×d_in│         ▼
 └────┬─────┘      ┌──────┐   B: d_out × r  (trainable)
      │            │  B   │   expand back to d_out
      │            └──┬───┘
      │               │ × α/r
      └──────► + ◄────┘
               │
               ▼
            h (d_out)
```

**Tiny worked example (rank 1, α/r = 1).** Let

```text
W₀ = [1 2]     B = [2]     A = [1 3]     x = [1]
     [3 4]         [1]                       [1]
```

Frozen path: `W₀x = [1+2, 3+4] = [3, 7]`. Adapter path, computed the cheap way: `Ax = 1·1 + 3·1 = 4` (a single number, because r = 1), then `B·4 = [8, 4]`. Sum: `h = [11, 11]`.

Check it the expensive way: `BA = [[2, 6], [1, 3]]`. Row 2 is half of row 1, so it really is rank 1. `W₀ + BA = [[3, 8], [4, 7]]`, and `[[3, 8], [4, 7]]·[1, 1] = [11, 11]`. Same answer. That equality is also why you can later **merge** the adapter into the weights for free inference.

**Real sizes.** A 4,096 × 4,096 projection has 16,777,216 weights. Rank-16 LoRA adds `r(d_in + d_out) = 16 × 8,192 = 131,072` trainable parameters: 0.78125% of the matrix. The LoRA paper's headline was on GPT-3 175B: roughly 10,000× fewer trainable parameters, VRAM during training down from 1.2 TB to 350 GB, and a checkpoint that shrinks from 350 GB to about 35 MB (rank 4, query and value projections only).

**Initialization.** The paper initializes `A` randomly (Gaussian) and `B` to zero, so `BA = 0` at step 0 and the model starts out *exactly* as the base model. Why not both zero? Then the gradient for `A` is proportional to `B` (zero) and the gradient for `B` is proportional to `Ax` (zero): nothing ever moves. One random factor and one zero factor gives "no change yet, but gradients can flow". (Hugging Face's PEFT library defaults to a Kaiming-uniform `A` with zero `B`, the same idea.)

> 🎬 **Animation — rank-1 LoRA on a 2×2 matrix:** use W₀ = [[1,2],[3,4]], B = [2,1]ᵀ, A = [1,3], x = [1,1]. Step 1: x splits into two paths. Step 2 (top): W₀x lights up as [3,7]. Step 3 (bottom): A squeezes x into the single number 4 (show a funnel from 2 lanes to 1), then B fans 4 out to [8,4]. Step 4: the paths sum to [11,11]. Step 5: fade to the "merged" view: the 2×2 grid BA = [[2,6],[1,3]] slides onto W₀ to give [[3,8],[4,7]], x multiplies through, and the same [11,11] appears. Caption: "same answer, no side branch".

## Rank, alpha, and where the adapters go {#rank-alpha-targets}

Three knobs cause most LoRA confusion. Take them one at a time.

**Rank `r`: capacity.** Bigger `r` means more independent directions the update can use, and more trainable parameters, linearly. The LoRA paper found surprisingly tiny ranks (even 1 or 2) worked for its tasks when adapting query and value projections of GPT-3. But "works on those benchmarks" isn't "always enough". *LoRA Learns Less and Forgets Less* (Biderman et al., 2024) found that for heavier jobs like continued pretraining on code and maths, LoRA substantially underperformed full fine-tuning, and the updates full fine-tuning learned had rank 10–100× higher than typical LoRA settings. The flip side: LoRA also **forgot less** of the base model's abilities outside the target domain. So: small `r` for style and format, and more rank (or full fine-tuning) when you're teaching a lot of new material.

**Alpha `α`: volume.** The side branch is multiplied by `α / r`. Rank 16 with α = 32 gives a multiplier of 2. The paper notes that with Adam, tuning α is roughly the same as tuning the learning rate, so they set α to the first `r` they tried and left it alone. The trap: if you double `r` but keep α fixed, you've *halved* the multiplier, so your "rank experiment" is secretly also a learning-rate experiment. PEFT also offers `use_rslora`, which scales by `α / √r` instead, to keep behaviour steadier as rank grows. Record which convention you used next to `r` and α.

**Target modules: where the adapters go.** Each linear layer you adapt is a **target module**. The original paper mostly adapted only `W_Q` and `W_V`. The QLoRA paper found that to match full 16-bit fine-tuning you needed LoRA on **all linear layers** in each block, and that once you did, the exact `r` mattered much less. PEFT supports `target_modules="all-linear"` for exactly this.

Module shapes aren't all square, so count carefully. Here's one block of an illustrative 8B-class model: `d = 4096`, MLP width 14,336, and **GQA** (grouped-query attention, where several query heads share one key/value head, so `W_K` and `W_V` are narrower; see **Making attention cheaper: GQA, FlashAttention, and long context**) with `h_kv = 8` heads of width 128, so K and V output 1,024 features.

| Module | Shape (d_in → d_out) | LoRA params at r = 16: `16 × (d_in + d_out)` |
|---|---|---|
| `W_Q` | 4096 → 4096 | 131,072 |
| `W_K` | 4096 → 1024 | 81,920 |
| `W_V` | 4096 → 1024 | 81,920 |
| `W_O` | 4096 → 4096 | 131,072 |
| MLP gate | 4096 → 14336 | 294,912 |
| MLP up | 4096 → 14336 | 294,912 |
| MLP down | 14336 → 4096 | 294,912 |
| **Per block** | | **1,310,720** |

Times `L = 32` blocks gives 41,943,040 ≈ **42 M** trainable parameters, about 0.5% of 8B. Adapting only `W_Q` and `W_V` would be (131,072 + 81,920) × 32 = 6,815,744 ≈ 6.8 M. Always print the real module names and shapes (`model.print_trainable_parameters()` in PEFT) instead of copying a config from a different architecture. A typo in a module name can silently adapt nothing at all.

> 🎬 **Animation — where adapters attach in a block:** draw one transformer block with seven labelled boxes: Q, K, V, O in the attention half, and gate, up, down in the MLP half, with widths drawn to scale (K and V visibly narrower, MLP boxes wide). Step 1: "Q+V only": small orange adapter pairs clip onto Q and V; a counter reads 213,KB → "212,992 params/block". Step 2: "all-linear": adapters clip onto all seven; the counter climbs to 1,310,720. Step 3: the block multiplies ×32 into a stack; the counter ends at 41,943,040 next to a greyed-out "8,000,000,000 frozen".

A few more practical knobs. **LoRA dropout** randomly zeroes inputs to the side branch during training; QLoRA found 0.05 helpful for 7B and 13B models but not for the larger ones. Adapters usually want a **higher learning rate** than full fine-tuning; Hugging Face's TRL docs suggest around 1e-4. And LoRA's variants (DoRA, which splits the update into magnitude and direction; alternative initializations like PiSSA or LoftQ) are all configuration flags in PEFT. Know they exist and treat them as experiments, not defaults.

## QLoRA: store the base in 4 bits, compute in bf16 {#qlora}

LoRA killed the gradient and optimizer bill. What's left is the frozen base itself: an 8B model at 2 bytes per weight is still 16 GB, and a 65B model is 130 GB. **QLoRA** (Dettmers et al., 2023) attacks that by storing the frozen base in 4 bits, while the adapters train in higher precision. Their headline: fine-tuning a 65B model on a single 48 GB GPU while matching full 16-bit fine-tuning quality on their benchmarks.

**Quantization** means storing each weight as a small integer code plus a shared scale, instead of a full 16-bit number. We go deep in **Quantization: spending fewer bits per weight**. Here's just what you need for QLoRA.

Analogy: the base model is a reference library you'll only *read*, never edit. So keep the books zipped on the shelf, unzip one chapter at a time onto your desk while you use it, and put it back. Your notebook (the adapters) stays open in full precision, because that's what you're writing in.

QLoRA combines three ideas:

1. **NF4 (4-bit NormalFloat).** Four bits give 16 possible values. Trained weights cluster around zero in a roughly bell-shaped (normal) distribution, so rather than spacing the 16 levels evenly, NF4 puts them at quantiles of a normal distribution: dense near zero, sparse in the tails. It also includes an exact zero. Weights are quantized in **blocks of 64**. Each block is divided by its largest absolute value (its **absmax**) so everything lands in [−1, 1], then each value snaps to the nearest NF4 level.
2. **Double quantization.** Every block of 64 needs its own scale. Stored as a 32-bit float that's 32/64 = 0.5 extra bits per weight. QLoRA quantizes those scales too (to 8 bits, in blocks of 256), bringing the overhead to 8/64 + 32/(64·256) ≈ 0.127 bits per weight. That saves about 0.373 bits per parameter, roughly 3 GB on a 65B model.
3. **Paged optimizers.** Memory use spikes on unusually long batches. Paged optimizers use NVIDIA unified memory so optimizer state can be paged out to CPU RAM during a spike instead of crashing with an out-of-memory error.

**Worked NF4 example.** Take a (tiny, illustrative) block of 4 weights: `[0.12, −0.40, 0.03, 0.25]`.

```text
absmax = 0.40
normalized = [0.30, −1.00, 0.075, 0.625]

nearest NF4 level (from the 16-entry table in bitsandbytes):
  0.30  → 0.3379  (index 11)     neighbours: 0.2461, 0.3379
 −1.00  → −1.0000 (index 0)
  0.075 → 0.0796  (index 8)      neighbours: 0.0, 0.0796
  0.625 → 0.5626  (index 13)     neighbours: 0.5626, 0.7230

stored:  codes [11, 0, 8, 13] (4 bits each) + scale 0.40
decoded: [0.3379, −1, 0.0796, 0.5626] × 0.40
       = [0.135, −0.400, 0.032, 0.225]
error:   [+0.015, 0, +0.002, −0.025]
```

The small weights come back almost exactly because the levels are dense near zero. The big-ish one loses the most, since the levels thin out there.

**The key distinction: storage precision vs compute precision.** Nothing is *multiplied* in 4 bits. During the forward and backward pass, each 4-bit block is dequantized to bf16 right before its matmul, used, and thrown away. The adapters `A` and `B` live in bf16 the whole time. So the precision stack is:

```text
 ┌──────────────── on the GPU, persistent ─────────────────┐
 │ W₀ : NF4 codes + scales  (≈ 0.5 + 0.127 bits/weight)    │  frozen
 │ A, B : bf16               + their grads + Adam state    │  trainable
 └──────────────────────────────────────────────────────────┘
            │ per layer, per step
            ▼
 dequantize W₀ block → bf16 → matmul → discard the bf16 copy
```

**Frozen doesn't mean "not in the backward pass".** Adapters in layer 3 only get a learning signal if the gradient can travel back from the loss through layers 32 → 4, and each of those layers contains frozen `W₀` matmuls. So backprop still flows *through* the quantized weights (dequantized on the fly). What frozen buys you is: no gradient *for* `W₀`, no optimizer state for `W₀`, no update to `W₀`. The QLoRA abstract puts it exactly: it "backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters."

> 🎬 **Animation — QLoRA storage vs compute:** show one layer's weight block as 64 tiny 4-bit tiles plus a small "scale" tag. Step 1 (forward): the tiles unzip into a bf16 grid (colour change from grey to blue), the input vector multiplies through it, and the bf16 grid dissolves. In parallel, a bf16 A→B side branch (orange) computes and adds its output. Step 2 (backward): a red gradient arrow flows back through the re-dequantized blue grid *and* the orange branch, but only the orange A and B tiles flash "updated". The grey 4-bit tiles never change. Caption: "stored in 4 bits · computed in bf16 · only adapters learn".

Two practical cautions. First, in Hugging Face land, the recipe is a `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16`, then `prepare_model_for_kbit_training`, then attach LoRA. A 4-bit *inference* loader on its own isn't a training recipe. Second, QLoRA saves memory, not time: dequantizing every block on every step adds work, so expect it to be slower per step than bf16 LoRA on hardware where both fit.

## Will it fit? A memory budget for an 8B model {#memory-budget}

This is the back-of-envelope an interviewer wants to watch you do. Same illustrative 8B-class model as above: about 6.98 B parameters in the block linear layers, plus about 1.05 B in the input embedding and output projection (vocab 128,256 × 4,096 each, untied), for ≈ 8.03 B total. Adapters: all-linear, r = 16, so ≈ 41.9 M trainable parameters.

Assumptions (state them out loud in an interview):

- Trainable parameters cost 16 bytes each (bf16 weight + bf16 grad + fp32 master + fp32 Adam m and v).
- In QLoRA, only the block linear layers are 4-bit; the embedding and output layers stay in bf16 (a common default, but check your stack).
- Activations are counted separately, below.

| Component | Full FT (bf16 + AdamW) | LoRA (bf16 base) | QLoRA (NF4 base) |
|---|---|---|---|
| Frozen base weights | none | 8.03 B × 2 B = 16.1 GB | 6.98 B × 0.5 B = 3.49 GB, + 0.11 GB scales, + 2.10 GB bf16 embed/output = 5.7 GB |
| Trainable weights + grads + optimizer | 8.03 B × 16 B = 128.5 GB | 41.9 M × 16 B = 0.67 GB | 0.67 GB |
| **Subtotal before activations** | **≈ 128.5 GB** | **≈ 16.7 GB** | **≈ 6.4 GB** |

(The scale overhead: 6.98 × 10⁹ weights × 0.127 bits ÷ 8 ≈ 0.11 GB.)

So full fine-tuning needs multiple 80 GB GPUs. LoRA fits one 24–40 GB card with room left for activations. QLoRA fits a 16–24 GB card. For comparison, the QLoRA paper reports the 4-bit LLaMA-7B base using 5,048 MB during training, which is the same ballpark as our estimate.

**Now the part people forget: activations.** **Activations** are intermediate values saved during the forward pass so the backward pass can use them. They scale with batch size `B` and sequence length `T`, not with how many parameters are trainable. LoRA does **not** shrink them. Two illustrative numbers for `B = 1`, `T = 4,096`:

- **Logits.** The output layer produces `T × V` scores. In fp32 that's 4,096 × 128,256 × 4 bytes ≈ 2.1 GB, for one sequence. This is why trainers compute the loss in chunks (TRL's default `chunked_nll` loss exists for exactly this reason).
- **Gradient checkpointing** (throw away most activations and recompute them during the backward pass) can cut stored activations down to about one residual-stream tensor per block: 4,096 × 4,096 × 2 bytes = 32 MiB, × 32 blocks = 1 GiB, plus one block's internals being recomputed at a time. You pay with roughly one extra forward pass of compute.

Put those together and QLoRA on this model at 4k context sits around 10 GB before allocator overhead and temporary workspaces. Plausible on a 16 GB card, comfortable on 24 GB. Double `T` and the activation terms roughly double too. **Gradient accumulation** (summing gradients over several micro-batches before one optimizer step) lets you get a big effective batch without holding all those activations at once.

The senior takeaway: *parameter reduction is not memory reduction is not compute reduction*. LoRA cut trainable parameters by ~190×, but the base forward and backward passes still run over every layer. Compute only drops because the weight-gradient matmuls for frozen weights are skipped (the LoRA paper measured a 25% training speedup on GPT-3 175B). Always measure peak memory on your *longest* realistic examples; a smoke test on short ones proves nothing.

> 🎬 **Animation — the memory bar race:** three horizontal stacked bars on a shared GB axis (0–140), labelled Full FT, LoRA, QLoRA. Segments: grey "frozen base", purple "trainable + grads + Adam", green "activations (B=1, T=4k, checkpointed)". Step 1: Full FT's purple bar extends to 128.5 GB, past a dashed "80 GB GPU" line. Step 2: LoRA's grey 16.1 GB + a sliver of purple 0.67 GB. Step 3: QLoRA's grey shrinks to 5.7 GB. Step 4: green activation segments (~3–4 GB) are added to LoRA and QLoRA; a slider for T moves from 4k to 16k and the green segments stretch while grey and purple stay fixed. Caption: "LoRA shrinks purple, QLoRA shrinks grey, nothing shrinks green".

## The data is the real model: templates, loss masks, quality {#data}

LoRA and QLoRA decide *which* weights change. **SFT** (supervised fine-tuning) decides *what they change toward*: maximise the probability of good responses given their prompts. These are independent choices. "Should we do SFT or LoRA?" is a category error, because you usually do SFT *with* LoRA. SFT itself (and RL-style methods you might also run on top of LoRA) is covered in **From base model to assistant: SFT, RLHF, DPO, and verifiable rewards**. Here we focus on the data bugs that quietly wreck adapter runs.

**1. The chat template.** A chat model doesn't see "messages"; it sees one token stream with role markers, like `<|im_start|>user … <|im_end|>`. That formatting is the **chat template**, and it's model-specific. Train with exactly the template you'll serve with. Two classic bugs, both flagged in the Transformers docs: adding the generation prompt (the dangling `<|im_start|>assistant` used at inference) to training examples, and double special tokens when you format to a string and then tokenize again with `add_special_tokens` on. Always decode a few tokenized training examples and eyeball the boundaries.

**2. The loss mask.** You want the model graded on the **answer**, not on reproducing the user's question. So each position `t` gets a mask value:

```formula
L = − [ Σₜ mₜ · log p_θ(yₜ | tokens before t) ] / [ Σₜ mₜ ]
mₜ ∈ {0, 1},   and you need Σₜ mₜ > 0
```

- `yₜ` is the correct next token at position `t`, and `p_θ(yₜ | …)` the probability the model (with parameters θ) gives it.
- `mₜ = 1` for assistant tokens (graded), `0` for system, user, tool output, and padding.
- The denominator averages over graded tokens only.

```text
 token:  <user> Refund  my  order  <end> <asst> Sure  , done  <end>
 mask:     0     0      0     0      0     0      1    1   1    1
 seen by attention?  yes for every token (all of it is context)
```

That picture shows the thing people mix up: a **loss mask** is not an **attention mask**. The user's tokens are still *read*, just not *graded*. It's a student reading the whole exam question but only being marked on their answer. In TRL, prompt-completion datasets are completion-only by default, and conversational datasets need `assistant_only_loss=True`, which in turn needs a chat template that marks assistant spans. Watch **truncation** too: if a long prompt pushes the whole answer past `max_length`, you get an example with `Σ mₜ = 0` that teaches nothing.

**3. Quality over quantity.** The QLoRA paper found a 9k-example curated dataset (OASST1) beat a 450k-example subsample of FLAN v2 for chatbot quality. For behaviour-shaping, a few thousand clean, consistent, representative examples usually beat a pile of noisy ones. If your labellers disagree on the escalation category, the adapter will faithfully learn the disagreement.

**4. Splits and leakage.** **Leakage** is when near-identical information lands in both training and test data, inflating test scores. Split *before* you make paraphrases or augmentations. Keep a customer, a source document, or a conversation thread entirely on one side. If you'll serve future traffic, hold out the most recent time window.

> 🎬 **Animation — loss mask vs attention mask:** show the 10-token row from the text above. Step 1: an "attention" spotlight sweeps left to right; every token lights up as visible context for the tokens after it. Step 2: a "grader's red pen" moves along the same row, skips the grey user and system tokens, and ticks only the four assistant tokens "Sure , done <end>". Step 3: a truncation cutter slides in at position 6 on a second, longer example; all four ticks vanish and a counter shows "Σ mₜ = 0: this example teaches nothing".

## Did it work? Evaluating an adapter {#evaluation}

Validation loss going down tells you the adapter is learning *something*. It doesn't tell you the product got better. Token-level likelihood can improve while the one field that matters (the escalation category) gets worse, because it's one token out of hundreds.

A decent eval plan for an adapter (full treatment in **Evaluation and observability: knowing whether it actually works**):

- **Task metrics tied to the failure you set out to fix.** For extraction: per-field precision and recall, schema validity, and "made-up values" as separate numbers.
- **Compare against the strongest baseline**, not the weakest: the same base model with your best prompt and retrieval. Same decoding settings, same template.
- **Regression checks.** Fine-tuning can erode general instruction-following or safety behaviour (the "forgetting" side of the tradeoff). Keep a small general-capability and safety suite and run it every time.
- **Slices.** Long inputs, rare categories, conflicting documents, cases where the right answer is "I don't know". Averages hide the expensive errors.
- **Process hygiene.** Pick rank, α, learning rate, and checkpoint on a dev set; touch the test set once. Report uncertainty; a 1-point gain on 200 examples may be noise.

And measure the *system* side separately: peak training memory, tokens per second, serving latency with the adapter attached, and adapter load time.

**Debugging "is it capacity or data?"** If training loss is still high, you may need more capacity: more target modules, higher rank, or full fine-tuning. If training loss is low but held-out results are poor, more rank won't help; you're overfitting or your labels are inconsistent. Go read failures.

## Shipping adapters: merge, hot-swap, or serve many at once {#serving}

An adapter is a patch, not a model. It only means something next to the **exact** base checkpoint, tokenizer, chat template, and target-module config it was trained with. Pin all of them together as one versioned bundle, or you'll get quietly wrong outputs.

You have two ways to serve it.

**Option A: merge.** Since `W₀x + (α/r)BAx = (W₀ + (α/r)BA)x`, you can fold the patch into the weights once (PEFT's `merge_and_unload()`) and ship an ordinary model. Zero extra inference latency, which was one of LoRA's original selling points. The cost: you're back to one full-size model per task, and the returned model no longer has adapter-management abilities, so keep the separate adapter and base reference for rollback.

**Option B: keep adapters separate.** One copy of the base sits in GPU memory; each request names which adapter to apply. The side branch adds only about `r(d_in + d_out)` multiply-adds per adapted matrix per token, versus `d_in × d_out` for the base matmul: for our 4,096 × 4,096 example at r = 16 that's 131,072 vs 16,777,216, under 1% extra. Serving engines like vLLM support this with settings such as `max_loras` (adapters active in one batch), `max_lora_rank`, and `max_cpu_loras` (a CPU-side cache), plus per-request adapter selection and dynamic loading. Research systems like S-LoRA push further: keep thousands of adapters in host memory, page the active ones onto the GPU in a pool shared with the KV cache, and use custom kernels to batch requests for *different* adapters together.

```text
            one base model in GPU memory (e.g. 16 GB)
   ┌────────────────────────────────────────────────────┐
   │  W₀ (shared by everyone)                           │
   └───────▲───────────────▲──────────────────▲─────────┘
           │               │                  │
      adapter "legal"  adapter "support"  adapter "sql"     each ~84 MB
       (r=16, 42M)       (r=16, 42M)       (r=8)            (42M × 2 bytes)
           ▲               ▲                  ▲
       requests         requests           requests   ← mixed in one batch
```

(42 M params × 2 bytes in bf16 ≈ 84 MB per adapter, versus 16 GB per merged model: 100 tasks is 8.4 GB of adapters, versus 1.6 TB of merged models.)

The real costs of multi-LoRA are operational: loading and evicting adapters, batching requests that use different adapters (why `max_lora_rank` shouldn't be set higher than you need), and cold-start latency. That's where batching and scheduling come in, covered in **Serving many users: batching, scheduling, and speculative decoding**.

**QLoRA deployment trap.** You trained against a 4-bit base. If you merge into a bf16 copy and then re-quantize for serving, the numbers differ from what you trained and evaluated. The algebra is identical in exact arithmetic, but rounding isn't. Some quantization backends don't support merging at all. Whatever artifact you ship, **evaluate that exact artifact**, not its ancestor.

**Merging several adapters** (say "legal tone" + "SQL skill") into one model is possible arithmetically, but nothing guarantees the behaviours compose without interfering. Treat it as a new model that needs its own eval.

> 🎬 **Animation — merge vs hot-swap:** split screen. Left, "merge": a small orange BA patch slides into a big grey W₀ grid, which turns a single blended colour and is stamped "model v2 (16 GB)"; three tasks produce three such 16 GB blocks, stacked, with a total of 48 GB. Right, "hot-swap": one grey W₀ block (16 GB) stays put; three small coloured adapter chips (84 MB each) sit in a tray. Requests arrive tagged red/blue/green; each picks up its chip, passes through W₀ + chip, and leaves. A batch of 6 mixed-colour requests goes through together. Total memory counter: 16.25 GB.

# Interview

## Question

A support team has 30,000 reviewed ticket-to-JSON examples, a refund policy that changes weekly, and one 24 GB GPU for training. An 8B instruct model already outputs valid JSON but often picks the wrong escalation category. Propose an adaptation, training, and release plan, and defend the memory budget.

## Answer

First, split the failures. Weekly policy changes are **evidence**, so they belong in retrieval: fine-tuning them in would mean retraining weekly and losing attribution. JSON validity is already fine (and constrained decoding could guarantee it anyway). The remaining gap, a consistent wrong category mapping even with a good prompt and the current policy in context, is a stable **behaviour**, which is what fine-tuning is for. I'd measure that gap against a strong prompt + RAG baseline before spending anything.

Method: QLoRA, or bf16 LoRA if it fits. Budget for this 8B model with all-linear r = 16 (≈ 42 M trainable parameters): bf16 LoRA is ~16 GB of frozen weights plus ~0.7 GB of adapter weights, gradients and Adam state, which leaves only ~7 GB for activations on a 24 GB card. QLoRA puts the base in NF4 (~5.7 GB including bf16 embeddings and scales), giving lots of headroom for long tickets. I'd turn on gradient checkpointing, use chunked loss to avoid a multi-GB logits tensor, and use gradient accumulation for batch size. Then I'd measure peak memory on the *longest* tickets before the real run.

Data: apply the model's own chat template, assistant-only loss masking, and check for truncated examples with zero supervised tokens. Split by customer and conversation, hold out the most recent weeks as the test set, and dedupe near-copies across splits. Audit label consistency on the rare, expensive categories, since 30k examples with inconsistent labels will teach the inconsistency.

Evaluate category precision and recall per class, with extra weight on costly mis-escalations, plus schema validity and a general-capability and safety regression suite. Compare to the prompt + RAG baseline with identical decoding. Tune rank, α and learning rate on dev; touch the test set once.

Ship it as a separate adapter pinned to the exact base revision, tokenizer and template, served via multi-LoRA if other teams share the base. If we merge, we evaluate the merged (and possibly re-quantized) artifact itself. Staged rollout with a full rollback bundle. The policy stays in retrieval.

## Follow-ups

- Training loss is low but held-out category accuracy barely moved. Is rank the problem? How would you tell capacity issues from label noise?
- You doubled r from 16 to 32 and results got worse. What else changed besides rank?
- Why are the LoRA parameter counts for `W_K` and `W_V` smaller than for `W_Q` in a GQA model?
- The team wants to merge the adapter and serve a 4-bit GGUF. What could differ from what you evaluated, and how do you check?
- Ten other teams want their own adapters on the same base. How does serving change, and what limits how many you can run at once?

# Pitfalls

- **"SFT or LoRA?"** They answer different questions. SFT is the objective (what you train toward); LoRA is the parameterization (which weights change). You usually do SFT *with* LoRA.
- **"LoRA makes the model low-rank."** Only the update `BA` has rank ≤ r. `W₀` and `W₀ + BA` are full-rank as usual.
- **"QLoRA does the math in 4 bits."** Weights are *stored* in NF4 and dequantized to bf16 for every matmul; adapters, activations and gradients are bf16.
- **"The base is frozen, so no backprop through it."** Gradients must flow through frozen layers to reach earlier adapters. Freezing only removes the base's gradients, optimizer state, and updates.
- **"0.5% of parameters trainable means 0.5% of the memory or compute."** The frozen base still has to be stored, activations are unchanged, and the forward and backward passes still cover every layer.
- **Changing rank without thinking about α.** With the standard α/r scaling, doubling r at fixed α halves the update's scale, which confounds the experiment.
- **Masking attention when you meant to mask loss.** The prompt should stay visible as context; it should just not be graded.
- **Deploying an adapter on a different base revision or chat template**, or assuming a merged and re-quantized export behaves exactly like the evaluated training setup.
- **Fine-tuning facts that change weekly** instead of retrieving them.

# Checklist

- Classify a sample of failures into evidence / instruction / format / behaviour, and justify fine-tuning only for the behaviour slice.
- Write `h = W₀x + (α/r)B(Ax)` from memory and explain every symbol, the initialization, and why rank(BA) ≤ r.
- Compute `r(d_in + d_out)` for square and GQA-shaped projections, and total it across layers.
- Build a memory table for full FT, LoRA, and QLoRA from bytes-per-parameter, and add the activation and logits terms.
- Explain NF4 blocks, absmax scaling, double quantization, and paged optimizers, and state what is stored vs computed in which precision.
- Inspect decoded training examples for template correctness, loss masks, and truncation.
- Split data by group and time before any augmentation.
- Evaluate against the best prompt + RAG baseline with task metrics, slices, and regression suites.
- Choose merge vs separate-adapter serving and name what must be pinned alongside an adapter.

# Sources

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — The update h = W₀x + (α/r)BAx, Gaussian-A/zero-B initialization, α tuning, W_Q/W_V targets, GPT-3 175B VRAM (1.2 TB → 350 GB) and checkpoint (350 GB → 35 MB) figures, 25% speedup, and merging for zero added latency.
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — NF4, block size 64, double quantization (0.5 → 0.127 bits/param), paged optimizers, bf16 compute, all-linear adapters needed to match full fine-tuning, LoRA dropout findings, 7B 4-bit memory, and the OASST1 vs FLAN v2 data-quality result.
- [bitsandbytes source: NF4 lookup table (functional.py)](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py) — The 16 NF4 values used in the worked quantization example.
- [LoRA Learns Less and Forgets Less (Biderman et al., 2024)](https://arxiv.org/abs/2405.09673) — LoRA underperforms full fine-tuning on code and maths, forgets less, and full fine-tuning learns updates with 10–100× higher rank.
- [Hugging Face PEFT: LoRA reference](https://huggingface.co/docs/peft/package_reference/lora) — LoraConfig options (r, lora_alpha, target_modules="all-linear", use_rslora, init_lora_weights, use_dora) and merge_and_unload.
- [Hugging Face PEFT: Quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) — The QLoRA recipe with BitsAndBytesConfig, prepare_model_for_kbit_training, all-linear targeting, LoftQ, and backend-specific merge limitations.
- [Hugging Face TRL: SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) — Completion-only and assistant-only loss, template requirements, PEFT integration, the ~1e-4 adapter learning-rate tip, and chunked NLL loss.
- [Hugging Face Transformers: Chat templates](https://huggingface.co/docs/transformers/chat_templating) — add_generation_prompt=False for training and avoiding duplicated special tokens.
- [vLLM: LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html) — Multi-adapter serving on one base, per-request adapters, max_loras / max_lora_rank / max_cpu_loras, and dynamic loading.
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters (Sheng et al., 2023)](https://arxiv.org/abs/2311.03285) — Host-memory adapter storage, unified paging with the KV cache, and batched heterogeneous-adapter kernels.

# Flashcards

## intervention

**Q:** When should you use retrieval instead of fine-tuning?

When the failure is missing or changing **evidence**: policies, prices, customer records. Retrieval supplies current facts at request time, with attribution and easy deletion. Putting facts into weights means retraining whenever they change and losing the ability to say where an answer came from. Fine-tuning is for stable **behaviour** (format, style, a decision pattern). The two combine well: an adapter can learn how to use retrieved evidence while the index supplies it.

## sft-lora

**Q:** Why can a training run be both SFT and LoRA?

They answer different questions. SFT is the **objective**: maximise the probability of good responses given prompts, with loss on response tokens. LoRA is the **parameterization**: which weights are allowed to change (only small adapter matrices). The same SFT loss can update all weights (full fine-tuning) or only LoRA factors, and preference methods can also run on LoRA.

## rank-bound

**Q:** In LoRA, what exactly is "low rank"?

Only the **update** `BA`. Since `B` is `d_out × r` and `A` is `r × d_in`, their product has rank at most `r`. The frozen `W₀` and the effective weight `W₀ + (α/r)BA` are generally full-rank. LoRA assumes the task-specific *change* is simple, not that the model is.

## parameter-count

**Q:** How many trainable parameters does LoRA add to one `d_out × d_in` matrix at rank r?

`r(d_in + d_out)`: `A` has `r × d_in` entries and `B` has `d_out × r`. It grows linearly with `r` and with the matrix's edges, not with its area. That's why it's tiny compared with `d_in × d_out` for large matrices. For non-square matrices (like GQA's K/V projections), use the real shape.

## worked-count

**Q:** How many parameters does rank-16 LoRA add to a 4,096 × 4,096 projection, and what fraction is that?

16 × (4,096 + 4,096) = **131,072**, versus 16,777,216 in the matrix: **0.78125%**. That's a parameter ratio, not a memory or compute ratio. The frozen matrix still has to be stored and multiplied every step.

## alpha

**Q:** How do rank r and alpha α differ, and what's the trap when you change r?

`r` sets capacity (how many directions the update can use). α sets the branch's scale through `α/r` in standard LoRA. So it acts like a learning-rate knob, which is why the LoRA paper just set α to the first r tried and left it. The trap: doubling `r` at fixed α halves the scale, so a "rank experiment" also changes effective step size. rsLoRA (`α/√r`) is one way to reduce this coupling.

## initialization

**Q:** Why is one LoRA factor initialized to zero and the other randomly, and not both to zero?

With `B = 0` and `A` random, `BA = 0`, so training starts exactly at the base model. If both were zero, `A`'s gradient (proportional to `B`) and `B`'s gradient (proportional to `Ax`) would both be zero, and nothing would ever update. One random factor breaks the symmetry while keeping the starting output unchanged.

## qlora-precision

**Q:** Does QLoRA do its arithmetic in 4 bits?

No. The frozen base is **stored** as NF4 codes plus per-block scales. Each block is dequantized to **bf16** right before its matmul and then discarded. Adapters, activations and gradients stay in bf16. Storage dtype and compute dtype are separate choices, and QLoRA saves memory, not compute.

## nf4

**Q:** What is NF4, and how does a block of weights get quantized with it?

A 4-bit, 16-level code whose levels sit at quantiles of a normal distribution: dense near zero, where most trained weights are, and sparse in the tails, with an exact zero. Weights are grouped in blocks of 64. Each block is divided by its absmax to land in [−1, 1], and each value snaps to the nearest level. You store 4-bit indices plus the scale. Double quantization then compresses the scales, cutting overhead from 0.5 to about 0.127 bits per weight.

## frozen-backprop

**Q:** If the base is frozen, why does backprop still go through it?

Adapters in early layers only get a learning signal if gradients flow backward from the loss through every later layer, and those layers are made of frozen `W₀` matmuls. Freezing removes the base's **weight gradients, optimizer state and updates**, not its place in the computation graph. Activation gradients still pass through it (dequantized, in QLoRA).

## memory

**Q:** Why doesn't adapter parameter count predict GPU memory?

Training memory = frozen base storage + (trainable params × ~16 bytes for weights, grads, master copy and Adam moments) + **activations** + logits + workspace. LoRA shrinks only the middle term. For an 8B model: full FT ≈ 128 GB before activations, bf16 LoRA ≈ 16.7 GB, QLoRA ≈ 6.4 GB. Long sequences can then make activations (e.g. a ~2 GB fp32 logits tensor at T = 4k) the biggest remaining term.

## mask

**Q:** How does a loss mask differ from an attention mask?

A **loss mask** decides which positions are *graded* (usually only assistant tokens). An **attention mask** decides which positions can be *seen*. Prompt tokens should stay visible as context but excluded from the loss. Also watch truncation: if the answer is cut off, the example has zero graded tokens and teaches nothing.

## split

**Q:** Why split data before creating paraphrases or augmentations?

Otherwise variants of the same underlying case land in both train and test, so the model gets rewarded for memorising that case rather than generalising. The same goes for grouping by customer, document, or conversation. For forward-looking use, hold out the most recent time window.

## evaluation

**Q:** Why doesn't lower validation loss prove the adapter is better?

Loss averages over all graded tokens, so it can improve while the one decision token that matters (a category, a yes/no) gets worse, or while rare, costly slices regress. Evaluate task metrics per class and per slice against the best prompt + RAG baseline, and run general-capability and safety regression checks.

## merge

**Q:** What should you check before deploying a merged (and possibly re-quantized) QLoRA result?

That the backend supports merging for that quantization format, that the base revision, tokenizer and template match exactly, and, above all, evaluate **the exported artifact itself**. Merging into bf16 and re-quantizing changes the rounding compared with what was trained and evaluated. The algebra is equal only in exact arithmetic. Keep the original adapter and base for rollback.

## multi-lora

**Q:** Why serve adapters separately instead of merging, and what limits it?

One base in GPU memory can serve many tasks: each adapter is ~tens of MB (42 M params × 2 bytes ≈ 84 MB) versus a full model copy per task, and the side branch adds under ~1% FLOPs per adapted matmul at typical ranks. Limits are operational: how many adapters can be active in a batch (`max_loras`), the maximum rank the kernels reserve for, load and eviction latency, and batching requests that use different adapters. S-LoRA-style systems page adapters from host memory to scale to thousands.
