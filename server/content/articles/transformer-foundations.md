---
{
  "slug": "transformer-foundations",
  "title": "The transformer block: assembling the full model",
  "category": "foundations",
  "summary": "Open the black box between embeddings and logits: add position, wire attention and an MLP around a residual stream with normalization, stack L blocks, count every parameter of a 7B-class model, and see why parallel training and one-token-at-a-time generation lead straight to the KV cache.",
  "difficulty": "Core",
  "minutes": 30,
  "prerequisites": ["tokens-and-embeddings", "attention-from-scratch"],
  "learningObjectives": [
    "Explain why attention on its own is order-blind and compare learned, sinusoidal, and rotary (RoPE) position schemes.",
    "Draw a pre-norm decoder block and describe what the residual stream, the normalization, and the MLP each contribute.",
    "Compute LayerNorm and RMSNorm by hand, and count the parameters and bf16 weight memory of a Llama-2-7B-shaped model from its config.",
    "Contrast teacher-forced parallel training with sequential generation, and size the KV cache that the asymmetry creates."
  ]
}
---

# Sections

## Opening the black box {#opening-the-black-box}

In **Text in, next token out: tokens, embeddings, and sampling** we built the whole pipeline with a hole in the middle. Text becomes token IDs, each ID looks up a row of the embedding table, and at the far end a final projection (the *unembedding*) turns each position's vector into one score per vocabulary entry. Those scores are the *logits*, and softmax turns them into next-token probabilities. In **Attention from scratch: how tokens talk to each other** we filled in the most famous piece of the hole: attention, where each token builds a query, compares it against every visible token's key, and takes a weighted average of their values, with a causal mask so nobody peeks at the future.

This article assembles the rest. By the end you'll be able to draw the entire model on a whiteboard, count its parameters to within a rounding error, and explain why training is fast and parallel while generation crawls along one token at a time.

Here's the one-screen map. Keep the shape column in your head, because it barely changes. Rows are **token positions** and columns are **features** (`T` = tokens in the sequence, `d` = hidden width, `V` = vocabulary size, `L` = number of blocks).

```text
 token IDs                                   shape  [T]
     │  embedding lookup (V × d table)
     ▼
 x₀  ──────────────────────────────────────  [T × d]   "the residual stream"
     │
 ┌───▼──────────────── block 1 ───────────────┐
 │  x = x + Attention(Norm(x))   mixes tokens │   [T × d] in, [T × d] out
 │  x = x + MLP(Norm(x))         per token    │
 └───┬────────────────────────────────────────┘
     ▼
    ...   same block design, new weights, L times
     ▼
 final Norm                                   [T × d]
     │  unembedding (d × V)
     ▼
 logits                                       [T × V]   one score per vocab entry, per position
```

That `T × d` invariant is the most useful thing on the page. Every block takes a `T × d` matrix and returns a `T × d` matrix. That's why you can stack as many as you like, and why "a 32-layer model" and "an 80-layer model" can share the same code.

> 🎬 **Animation — the full stack at a glance:** start with the sentence "the cat sat" split into 3 token chips. Step 1: each chip drops into a tall embedding table and pulls out a row of 4 coloured cells, forming a 3 × 4 grid labelled "T × d". Step 2: the grid slides up into a box labelled "Block 1", which briefly shows two inner stages, "Attention (tokens talk)" with arrows between rows and "MLP (each token thinks)" with arrows only within each row, then exits still 3 × 4. Step 3: the grid passes through Blocks 2, 3, … L (fast-forward with a counter), its colours shifting slightly each time. Step 4: the last row only is multiplied by a wide d × V matrix, producing a long bar of V logits, and a softmax turns the tallest bar ("on") into a probability of 0.41 (illustrative).

## Attention can't tell "dog bites man" from "man bites dog" {#order-blind}

Here's a strange fact that surprises most people: the attention you learned has no idea what order the tokens are in.

Think about what attention computes for one position. It scores the query against a set of keys, softmaxes, and adds up the values. Addition doesn't care about order: `a + b + c` equals `c + a + b`. If you shuffle the other tokens, each one carries its own key and value along with it, so the scores get shuffled right alongside them and the weighted sum comes out identical. Attention treats context as a **bag** of vectors, not a sequence.

A tiny example. Compare two prompts whose last token is `man`:

```text
 prompt A:  "dog  bites  man"        prompt B:  "bites  dog  man"
 last token "man" sees {dog, bites, man}     sees {bites, dog, man}

 same query (man), same keys, same values → same weights → same output
```

In the first layer, the output at `man` is bit-for-bit identical in both prompts, even though one is news and the other is nonsense. Without *mask* the effect is total: shuffle all the tokens and every output just gets shuffled the same way. (The fancy term is *permutation equivariant*.)

A careful interviewer may push back: "doesn't the causal mask leak order?" Somewhat. With a causal mask, `dog` in prompt A sees only itself, while `dog` in prompt B sees `bites` too, so earlier positions end up with different vectors, and later layers can pick up on that. But that's an accidental, indirect signal. Every mainstream LLM design adds position information explicitly, and that's what the next section is about.

> 🎬 **Animation — attention is order-blind:** show two rows of token chips, "dog bites man" and "bites dog man", each with 3 value vectors as small coloured bars. Step 1: highlight the last chip "man" in both rows and draw attention arrows from it to all three chips, labelled with the same weights 0.5 / 0.3 / 0.2 (illustrative) on matching tokens. Step 2: show the weighted sums being built bar by bar; the two result bars end up identical, and a "=" sign pops between them. Step 3: add small position badges "0, 1, 2" to each chip; now the weights differ between rows (e.g. 0.5/0.3/0.2 vs 0.2/0.6/0.2) and the two outputs visibly diverge.

## Three ways to stamp a position onto a token {#position}

There are two broad strategies. You can **add a "where am I" vector** to each token's embedding once at the bottom, or you can **twist the queries and keys** inside every attention layer so that their dot product depends on distance. The table sums up the three designs you should know.

| Scheme | Where it acts | What position means | Seen in |
|---|---|---|---|
| Learned absolute | Added to embeddings once | A learned vector per slot 0 … max_len−1 | GPT-2-style models |
| Sinusoidal | Added to embeddings once | Fixed sine/cosine waves, no parameters | Original Transformer (2017) |
| RoPE (rotary) | Applied to Q and K in every layer | Rotate each pair of features by an angle ∝ position | LLaMA family and many modern LLMs |

**Learned absolute positions** are the most obvious: keep a second table with one row per position, and add row `p` to the token at position `p`. It's simple and it works, but the table has a fixed number of rows. Position 5,000 has no row if you trained with 4,096.

**Sinusoidal positions** replace the learned table with a formula from the original Transformer paper:

```formula
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Here `pos` is the token position, `i` indexes a pair of features, and `d` is the model width. Each feature pair is a wave with its own wavelength. Think of an odometer: the fast-spinning digits (small `i`) distinguish neighbouring positions, and the slow ones (large `i`) tell far-apart regions apart. With `d = 4` and `pos = 1`, the pair `i = 0` gives `[sin 1, cos 1] = [0.8415, 0.5403]`, and the pair `i = 1` uses `1/10000^(2/4) = 1/100`, giving `[sin 0.01, cos 0.01] ≈ [0.0100, 1.0000]`. So position 1 gets the vector `[0.8415, 0.5403, 0.0100, 1.0000]` added to it. The paper reports that learned and sinusoidal positions gave nearly identical results, and says they chose sinusoids hoping they would extrapolate to longer sequences.

**RoPE (rotary position embedding)** is what you'll find in most modern open models, so it deserves the most attention. The clock-hand picture works well. Split a query or key vector into pairs of numbers, and treat each pair as a clock hand in 2-D. Before computing the dot product, rotate each hand by an angle proportional to the token's position. A dot product between two 2-D vectors depends on the **angle between them**. When both hands have been turned by their own positions, the extra angle between them is proportional to the **gap** between the positions, not to where the pair sits in the document.

Worked example with one pair and a rotation of 30° per position. Take a query `q = [1, 0]` at position 5 and a key `k = [1, 0]` at position 3.

```text
 q at pos 5 → rotated 150°      k at pos 3 → rotated 90°
 angle between them = 60°       q·k = cos 60° = 0.5

 now move both 10 tokens later:
 q at pos 15 → rotated 450°     k at pos 13 → rotated 390°
 angle between them = 60°       q·k = cos 60° = 0.5   (same!)
```

Same distance, same score, no matter where in the document the pair sits. That's the relative-position property the RoFormer paper derives. The rotation for one pair at position `m` with frequency `θ` is:

```formula
[x₁, x₂]  ↦  [x₁·cos(mθ) − x₂·sin(mθ),  x₁·sin(mθ) + x₂·cos(mθ)]
θ_i = 10000^(−2i/d)
```

`x₁, x₂` are one feature pair of a query or key, `m` is the token position, and `θ_i` is that pair's frequency. The same geometric ladder as the sinusoidal scheme gives fast hands and slow hands. Two details matter in interviews. First, RoPE touches **only queries and keys**, never values: position shapes *who you listen to*, not *what they say*. Second, LLaMA applies it **at every layer** instead of adding anything to the embeddings.

What RoPE does **not** do is guarantee that a model works at context lengths it wasn't trained on. At unseen lengths the slow hands reach angles the model never saw during training, and quality can fall apart. Stretching context usually means rescaling those frequencies and doing some further training, and even then you have to test recall at different depths. We'll go through this in detail in **Making attention cheaper: GQA, FlashAttention, and long context**.

> 🎬 **Animation — RoPE as clock hands:** draw a 2-D circle. Step 1: a query arrow q = [1, 0] and a key arrow k = [1, 0] both start pointing right. Step 2: label q "position 5" and rotate it 150°; label k "position 3" and rotate it 90°; shade the 60° wedge between them and show "q·k = cos 60° = 0.5". Step 3: slide a position counter +10 for both; q spins to 450° (one full turn plus 90°), k to 390°; the 60° wedge reappears unchanged, with "q·k = 0.5" again. Step 4: zoom out to show several circles side by side (one per feature pair), with the first hand spinning fast and the last barely moving, captioned "θ_i = 10000^(−2i/d): fast hands and slow hands".

## The residual stream: a shared bus every layer writes to {#residual-stream}

Now for the plumbing that holds the block together. Each sublayer (attention, then the MLP) doesn't *replace* its input. It computes an **update** and **adds** it:

```formula
u = x + Attention(Norm(x))
y = u + MLP(Norm(u))
```

`x` is the `T × d` input to the block, `u` is the stream after the attention update, and `y` is the block's output. The running vector that gets added to all the way up the stack is called the **residual stream**.

Here's the analogy I'd use. Picture a shared notebook passed up a line of specialists. Nobody rewrites the page. Each one reads a tidied copy, then scribbles a note in the margin. Or, for the systems-minded, it's a **bus**. The embedding puts a value on the bus, every attention layer and every MLP reads from it and writes a delta back, and the unembedding reads whatever is on the bus at the end. Anthropic's interpretability work frames it exactly this way: every component communicates by reading from and writing to the residual stream.

A tiny worked example for one token with `d = 4`:

```text
 x  (entering block)     [ 1.0,  2.0, -1.0,  0.5]
 + attention update      [ 0.2, -0.1,  0.0,  0.3]
 = u                     [ 1.2,  1.9, -1.0,  0.8]
 + MLP update            [-0.4,  0.0,  0.5,  0.1]
 = y  (leaving block)    [ 0.8,  1.9, -0.5,  0.9]
```

Notice that feature 2 (value `1.9`) made it through the MLP untouched, because the MLP wrote a zero there. A layer that has nothing useful to add can write roughly zero and get out of the way. Without the residual connection, every layer would have to rebuild the entire representation from scratch just to pass information along.

Why does this matter so much? Two reasons.

1. **Gradients get a highway.** When training pushes the error signal backwards, the derivative of `x + f(x)` with respect to `x` is `1 + f′(x)`. That `1` is a direct route from the loss back to the earliest layers, even if the individual `f′` terms are small. Stacking dozens of layers without this is a recipe for vanishing signal. (Gradients themselves are covered in **How a model learns: loss, gradients, and optimizers**.)
2. **Layers become incremental editors.** Each block refines what's there instead of re-encoding it, which is part of why very deep stacks are trainable at all.

> 🎬 **Animation — the residual bus:** draw a thick horizontal pipe running left-to-right labelled "residual stream (d = 4)", carrying the vector [1.0, 2.0, −1.0, 0.5] as four coloured cells. Step 1: an "Attention" box above the pipe taps a copy via a "Norm" valve, then drops the update [0.2, −0.1, 0.0, 0.3] back onto the pipe with a "+" junction; the cells animate to [1.2, 1.9, −1.0, 0.8]. Step 2: an "MLP" box does the same with update [−0.4, 0.0, 0.5, 0.1], giving [0.8, 1.9, −0.5, 0.9]; the second cell glows to show it passed through unchanged. Step 3: zoom out to show L such tap-and-add pairs along one long pipe, ending at an "Unembedding" reader. Step 4: reverse the flow direction in red to show gradients riding the pipe straight back to the start.

## Keeping the numbers in range: LayerNorm, RMSNorm, and pre-norm {#normalization}

Adding updates layer after layer has a side effect: the size of the vectors on the bus can drift, sometimes a lot. Normalization is the thermostat. Before a sublayer reads the stream, it rescales each token's vector to a standard size, so the attention and MLP weights always see inputs in a predictable range.

Two things normalization is **not**: it doesn't mix tokens (each position is normalized on its own, across its `d` features), and it has nothing to do with the softmax over the vocabulary.

**LayerNorm** centres and rescales each token's vector. It subtracts the mean of the features, divides by their standard deviation, and then applies a learned per-feature gain `γ` and bias `β`. **RMSNorm** skips the centring. It divides by the root-mean-square of the features and applies a learned gain `γ` (no bias).

```formula
LayerNorm(x) = γ ⊙ (x − mean(x)) / √(var(x) + ε) + β
RMSNorm(x)   = γ ⊙ x / √(mean(x²) + ε)
```

`x` is one token's `d`-dimensional vector, `⊙` is element-wise multiplication, and `ε` is a tiny constant (Llama's config uses `1e−6`) that prevents division by zero.

Worked example with `x = [3, 4]`, taking `γ = 1`, `β = 0`, and ignoring `ε`:

| Step | LayerNorm | RMSNorm |
|---|---|---|
| Centre | mean = 3.5 → `[−0.5, 0.5]` | (skipped) |
| Scale | std = √0.25 = 0.5 | RMS = √((9 + 16)/2) = √12.5 ≈ 3.5355 |
| Result | `[−1, 1]` | `[0.8485, 1.1314]` |

Same input, different outputs. So these aren't interchangeable. You can't swap one for the other in a trained checkpoint and expect it to keep working. The RMSNorm paper argues that the centring step isn't what makes LayerNorm useful, and reports comparable quality with lower running time. LLaMA uses RMSNorm.

**Where** the norm sits matters as much as which one you use. The original 2017 Transformer used **post-norm**, `LayerNorm(x + Sublayer(x))`, which normalizes *after* the add, so the bus itself gets renormalized at every step. Modern decoders use **pre-norm**, `x + Sublayer(Norm(x))`, which normalizes only the copy the sublayer reads and leaves the bus itself as a clean running sum.

```text
 post-norm (2017)                     pre-norm (modern LLMs)
 x ──┬──► Sublayer ──► (+) ──► Norm ──►     x ──┬──► Norm ──► Sublayer ──► (+) ──►
     └──────────────────┘                       └───────────────────────────┘
   the bus itself gets normalized          the bus stays an untouched sum
```

Xiong et al. showed that post-norm gives large gradients near the output at initialization, which is why it needs a careful learning-rate warmup, and that pre-norm behaves well from the start. LLaMA's paper says it normalizes the input of each sublayer "to improve the training stability." One consequence of pre-norm: since the bus is never normalized inside the stack, you need **one final norm** after the last block before the unembedding, which is why the map at the top has one.

> 🎬 **Animation — LayerNorm vs RMSNorm on [3, 4]:** plot the vector [3, 4] as an arrow on a 2-D grid. Left panel (LayerNorm): step 1 draws the mean 3.5 as a dot on the diagonal and slides the arrow to [−0.5, 0.5]; step 2 stretches it by 1/0.5 to [−1, 1]. Right panel (RMSNorm): a single step shrinks the arrow along its own direction by 1/3.5355 to [0.8485, 1.1314]; a dashed circle of radius √2 shows the new length. End with both results side by side and the caption "different functions: not swappable."

## The MLP: where each token thinks on its own {#mlp}

Attention is the only place where tokens exchange information. The second sublayer, the **MLP** (multi-layer perceptron, also called the feed-forward network), does the opposite. It processes **each position independently**, with the same weights for every position. If attention is the meeting where everyone shares notes, the MLP is everyone going back to their desk to think about what they heard.

```text
            positions →
  attention: ●───●───●───●     rows talk to each other
             ╲ ╱ ╲ ╱ ╲ ╱
  MLP:       ●   ●   ●   ●     each row processed alone, same weights
             │   │   │   │
```

The classic MLP expands, applies a nonlinearity, and compresses back. The original Transformer went from `d = 512` up to `2048` (4×), applied ReLU (`max(0, z)`), and came back down to 512. The expansion gives the model a wide scratch space, and the nonlinearity is what lets it compute things that aren't just linear blends. Without it, the whole stack would collapse into a big linear map. Many interpretability results suggest that much of a model's stored "knowledge" lives in these MLP weights. Treat that as a helpful intuition, not a hard rule.

Modern LLMs use a **gated** variant called **SwiGLU**. Instead of one up-projection, there are two. One goes through a smooth activation called **SiLU** (`SiLU(z) = z · sigmoid(z)`, which looks like ReLU with the sharp corner sanded off) and acts as a gate. The other carries the content. They're multiplied element by element and projected back down:

```formula
SwiGLU(x) = [ SiLU(x·W_gate) ⊙ (x·W_up) ] · W_down
```

`W_gate` and `W_up` are `d × m`, `W_down` is `m × d`, and `m` is the intermediate width. Think of the gate as a row of **dimmer switches**, one per hidden feature: continuous, and chosen by the input itself.

Tiny worked example with `d = 2` and `m = 2`. Take `x = [1, 2]`, `W_gate = [[2, 1], [0, −1]]`, `W_up = [[1, 0], [1, 2]]`, and `W_down = [[1, 1], [0, 1]]` (rows are input features, columns are output features):

```text
 x·W_gate = [1·2 + 2·0,  1·1 + 2·(−1)] = [ 2,     −1    ]
 SiLU     = [2·σ(2),     −1·σ(−1)]     = [ 1.7616, −0.2689]
 x·W_up   = [1·1 + 2·1,  1·0 + 2·2]    = [ 3,      4    ]
 gate ⊙ up                             = [ 5.2848, −1.0758]
 · W_down → [5.2848,  5.2848 − 1.0758] = [ 5.2848,  4.2090]
```

The second hidden feature had a big "content" value (4) but its gate was nearly shut (−0.27), so it barely contributed. That's the dimmer in action. Shazeer's GLU-variants paper found that gated feed-forward layers like this beat the plain ReLU and GELU versions, and LLaMA adopted SwiGLU.

Sizing matters. A plain MLP has two matrices (`2·d·m` parameters) and SwiGLU has three (`3·d·m`). To keep the parameter count comparable to a 4× plain MLP (`2·d·4d = 8d²`), LLaMA shrinks the width to `m = ⅔ · 4d`, so that `3·d·(8d/3) = 8d²`. For `d = 4096`, that's 10,922.67, and the released config rounds it up to **11,008** (= 43 × 256, a GPU-friendly multiple). A fair comparison between MLP variants has to match parameters like this. Otherwise you're just measuring "bigger is better."

One more pointer: the gate here is **not** a router. Mixture-of-experts models replace this single MLP with many expert MLPs and a router that sends each token to a few of them. That's a different idea, covered in **Mixture of experts: more parameters, same compute per token**.

> 🎬 **Animation — SwiGLU as dimmer switches:** show x = [1, 2] as two cells. Step 1: it fans out through two matrices, producing a "gate" pair [2, −1] (orange) and an "up" pair [3, 4] (blue). Step 2: the gate pair passes through a SiLU curve plot; points at z = 2 and z = −1 light up and map to 1.7616 and −0.2689; draw them as dimmer knobs, one nearly full and one nearly off. Step 3: multiply element-wise with the blue pair; 3 → 5.2848 glows bright, 4 → −1.0758 dims. Step 4: W_down compresses back to [5.2848, 4.2090], which drops onto the residual bus with a "+".

## Stacking L blocks and counting every parameter {#stacking-and-counting}

Put it together and a whole decoder is almost embarrassingly short:

```python
def decoder(token_ids):
    x = embed[token_ids]                     # [T, d]
    for block in blocks:                     # L times, each with its own weights
        x = x + block.attn(rms_norm(x, block.g1), causal=True, rope=True)
        x = x + block.mlp(rms_norm(x, block.g2))   # SwiGLU
    x = rms_norm(x, final_g)
    return x @ unembed                       # [T, V] logits
```

Now the classic interview exercise: **count the parameters** of a Llama-2-7B-shaped model. The Hugging Face `LlamaConfig` defaults match that checkpoint: `V = 32,000`, `d = 4,096`, `m = 11,008`, `L = 32`, `h = 32` heads (so `d_head = 128`), no biases, and untied input and output embeddings (`tie_word_embeddings = False`), so the embedding table and the unembedding are two separate matrices.

| Piece | Formula | Parameters |
|---|---|---|
| Token embedding | V · d = 32,000 · 4,096 | 131,072,000 |
| Attention per block (W_Q, W_K, W_V, W_O) | 4 · d² = 4 · 4,096² | 67,108,864 |
| SwiGLU MLP per block (gate, up, down) | 3 · d · m = 3 · 4,096 · 11,008 | 135,266,304 |
| Two RMSNorm gains per block | 2 · d | 8,192 |
| **One block** | sum of the three rows above | **202,383,360** |
| All 32 blocks | 32 · 202,383,360 | 6,476,267,520 |
| Final RMSNorm | d | 4,096 |
| Unembedding (untied) | d · V | 131,072,000 |
| **Total** | | **6,738,415,616 ≈ 6.74 B** |

That's why it's called a "7B" model. A few senior-level takeaways fall out of this table:

- **The blocks are ~96%** of the weights (6.476 B of 6.738 B). At this vocabulary size, embeddings are a rounding error. That flips for small models with huge vocabularies, where embeddings can dominate, and that's when tying the input and output matrices really pays.
- **The MLP is two-thirds of each block** (135.3 M of 202.4 M). When people talk about where parameters live, "mostly the MLPs" is the right first answer.
- **The 12·L·d² shortcut.** Attention is `4d²`, and an MLP sized to `8d²` gives `12d²` per block. So `12 · 32 · 4,096² = 6,442,450,944`, within 0.5% of the exact block total. That's great for napkin math on unfamiliar models.
- **Norms are free.** 8,192 parameters per block are noise in the count, but they still matter a lot for stability.

**Memory for the weights alone** at bf16 (2 bytes per parameter): `6,738,415,616 × 2 = 13,476,831,232 bytes ≈ 13.5 GB ≈ 12.55 GiB`. That's before the KV cache, activations, or anything training needs (gradients and optimizer state multiply this several times over, covered in **Distributed training: fitting a training run onto a cluster**). Squeezing the weights into fewer bits is the job of **Quantization: spending fewer bits per weight**.

**Compute per token**, as a rule of thumb: a forward pass costs about `2 · N` FLOPs per token (one multiply and one add per weight), so about 13.5 GFLOPs per token here. This ignores the attention-score term, which grows with context length. Training costs roughly three times that, which is where the famous `C ≈ 6·N·D` comes from, covered in **Pretraining: data, compute, and scaling laws**.

> 🎬 **Animation — where the 6.74 B parameters live:** start with a single tall bar representing 6,738,415,616 parameters. Step 1: split it into three stacked segments: embedding (131 M, thin), 32 blocks (6,476 M, huge), unembedding (131 M, thin); label percentages 1.9% / 96.1% / 1.9%. Step 2: zoom into one block's 202,383,360 and split it into attention 67.1 M (33.2%), MLP 135.3 M (66.8%) and a hairline for norms (8,192). Step 3: overlay the napkin estimate 12·L·d² = 6.44 B as a dashed line just below the 6.48 B block segment. Step 4: convert the full bar to memory: × 2 bytes = 13.48 GB (bf16).

## Training: every position predicts at once {#training-parallel}

Here's the asymmetry that shapes the entire serving stack. Let's look at training first.

During training we already have the full text. So we feed the sequence in and ask **every position** to predict the token that comes next. The targets are just the inputs **shifted left by one**:

```text
 position:   0      1      2      3
 input:     The    cat    sat    on
 target:    cat    sat    on     the
            ▲ each position predicts the next token, all in one forward pass
```

Feeding the true previous tokens, rather than the model's own guesses, is called **teacher forcing**. It works in one pass because of the causal mask from **Attention from scratch: how tokens talk to each other**: position 1 can see `The cat` but not `sat`, so predicting `sat` there isn't cheating. One forward pass over a 4,096-token sequence produces 4,096 separate predictions, with 4,096 tiny exams graded at once. That's the reason transformer training maps so well onto GPUs.

Each prediction is graded with **cross-entropy**: the negative log of the probability the model gave to the correct token, averaged over positions. Lower is better. Here's a worked example on three positions with made-up probabilities:

| Position | P(correct token) | −ln P |
|---|---|---|
| 0 | 0.5 | 0.6931 |
| 1 | 0.25 | 1.3863 |
| 2 | 0.125 | 2.0794 |
| **Mean** | | **1.3863 nats** |

A *nat* is just the unit you get from using the natural log. **Perplexity** is `exp(loss) = exp(1.3863) = 4`. Read it as "on average the model was as unsure as if it were choosing among 4 equally likely tokens." The mechanics of turning this loss into weight updates belong to **How a model learns: loss, gradients, and optimizers**.

A debugging reflex worth having: **a loss that's suspiciously good is usually a bug.** If you forget to shift the labels, the model is asked to predict the token it's looking at. If the causal mask is broken, position 1 can peek at the answer. Both give amazing-looking training loss and a model that's useless at generation.

## Generation: one token at a time, and why that creates the KV cache {#generation-kv-cache}

Now flip to inference. At generation time, the next token **doesn't exist yet**. You have to run the model, pick a token (with the decoding rules from **Text in, next token out: tokens, embeddings, and sampling**), append it, and only then can you ask about the token after that. It's inherently sequential: 200 new tokens means 200 dependent steps.

```text
 training  (one pass, all positions)      generation (one step per new token)
 ┌────────────────────────────┐           step 1: [The cat sat]      → "on"
 │ The cat sat on → cat sat on the │     step 2: [The cat sat on]   → "the"
 └────────────────────────────┘           step 3: [The cat sat on the] → "mat"
    parallel across T                       each step waits for the previous one
```

The naive loop is wasteful. At step 3, we'd push `The cat sat on` through all 32 layers *again*, even though nothing about those tokens has changed. And with a causal mask, **nothing can change**: earlier positions never look at later ones, so a token's keys and values at every layer are frozen the moment it's processed. So we compute them once and keep them. That stored state is the **KV cache**: for every layer, the key and value vectors of every past token. Each new step then does full work only for the **one** new token. It computes that token's query, key, and value, appends the new K and V to the cache, and attends over the cached history.

It's like keeping notes on the chapters you've already read, instead of rereading the whole book every time you turn a page. Generation splits into two phases: **prefill**, which processes the whole prompt in one parallel pass and fills the cache, and **decode**, which is the one-token-at-a-time loop that reads the cache.

The cache isn't free. Its size is:

```formula
KV bytes = 2 · L · B · T · h_kv · d_head · bytes_per_element
```

The `2` counts keys and values, `L` is layers, `B` is sequences in the batch, `T` is cached tokens, `h_kv` is the number of key/value heads (equal to `h` in ordinary multi-head attention, as in Llama-2-7B), and `d_head` is the per-head width. For our model with one 4,096-token sequence in bf16:

```text
 2 × 32 × 1 × 4,096 × 32 × 128 × 2 bytes
 = 2,147,483,648 bytes = 2 GiB           (≈ 512 KiB per token)
```

So one long conversation costs 2 GiB of cache on top of ~12.55 GiB of weights, and 16 of them cost 32 GiB. That's why cache memory, not FLOPs, usually sets how many users a GPU can serve at once. It's also why reducing `h_kv` (grouped-query attention) is such a big deal, while turning down the sampling temperature changes nothing about memory. The cache is also **execution state tied to an exact prefix**. Edit an earlier message and everything after the edit point is invalid. It isn't a store of past answers.

Caching also doesn't make each step constant-time. The new token still reads and attends over every cached position, so decode steps get slower as the context grows. We'll go through prefill vs decode, cache sizing and paging properly in **What happens at inference: prefill, decode, and the KV cache**, and the head-sharing tricks that shrink the cache in **Making attention cheaper: GQA, FlashAttention, and long context**.

> 🎬 **Animation — training in parallel vs decoding with a growing cache:** left half: the sentence "The cat sat on" enters as 4 chips; one forward pass lights all 4 output slots simultaneously with targets "cat sat on the" and green ticks. Right half: the prompt "The cat sat" enters once (label "prefill"), and a stack of 32 thin layer trays each receives 3 K/V rows. Then a "decode" counter ticks: step 1 produces "on" and each tray grows by one row (now 4); step 2 produces "the" (5 rows); step 3 produces "mat" (6 rows). A memory gauge beside the trays fills at 512 KiB per token, reaching "2 GiB" when the counter jumps to 4,096 tokens.

# Interview

## Question

Walk me through one decoder block of a Llama-style model. Then, given d = 4096, L = 32, 32 heads, an SwiGLU intermediate width of 11,008, and a 32,000-token vocabulary with untied embeddings, estimate the parameter count and bf16 weight memory. Finally, explain why this model trains on a whole sequence in one pass but generates text one token at a time, and what that costs at inference.

## Answer

Each block reads and writes a residual stream of shape T × d. It has two pre-norm sublayers: `u = x + Attention(RMSNorm(x))` and `y = u + SwiGLU(RMSNorm(u))`. Attention is the only place tokens exchange information. RoPE rotates the queries and keys by position in every layer, because attention on its own is order-blind. The MLP processes each position independently with shared weights. Residual additions give each layer an incremental "edit" role and give gradients a direct path. Pre-norm keeps the stream an unnormalized running sum, which trains more stably than post-norm, and a final RMSNorm sits before the unembedding.

For the count, attention is 4d² = 67.1 M per block, SwiGLU is 3·d·m = 135.3 M, and the norms add 8,192, for 202.4 M per block. Times 32 that's 6.476 B. Add the 131.1 M embedding table, the 131.1 M unembedding and a 4,096-parameter final norm to get about 6.74 B. As a sanity check, 12·L·d² gives 6.44 B. At 2 bytes per parameter, that's about 13.5 GB (12.55 GiB) for the weights alone, before activations or KV cache.

Training uses teacher forcing: the targets are the inputs shifted by one, and the causal mask means every position can be scored in a single forward pass. Generation can't do this, because each new token has to be chosen before the next one can be predicted. Since causal attention never lets earlier tokens see later ones, their per-layer keys and values never change, so we cache them. That KV cache costs 2·L·T·h_kv·d_head·bytes, which is 512 KiB per token here, or 2 GiB for a 4,096-token sequence. So cache memory, more than FLOPs, limits concurrency. Decode steps still attend over the whole cache, so latency per token grows with context. The levers are fewer KV heads (GQA), cache quantization, and paging, and I'd size capacity from the checkpoint's actual `h_kv`.

## Follow-ups

- How does the parameter count and KV cache change if the model uses grouped-query attention with 8 KV heads?
- Why does pre-norm need a final normalization layer before the unembedding, and what goes wrong in post-norm without learning-rate warmup?
- If you swapped RMSNorm for LayerNorm in a trained checkpoint, what would happen, and why?
- A training run shows near-zero loss on step 200. What two bugs do you check first?
- Why does RoPE apply to queries and keys but not values, and what breaks when you run it past the trained context length?

# Pitfalls

- Saying attention "knows" word order. Without position information, attention is a weighted sum over a set, and shuffling the context leaves each token's output unchanged.
- Treating LayerNorm and RMSNorm as interchangeable. On `[3, 4]` one gives `[−1, 1]` and the other `[0.8485, 1.1314]`. They're different functions.
- Thinking the MLP mixes information across tokens. It's strictly per-position; only attention moves information between positions.
- Calling the SwiGLU gate a router. It's a continuous per-feature multiplier, while mixture-of-experts routing chooses which expert MLPs a token visits.
- Forgetting that untied models have **two** V × d matrices, or counting RoPE as parameters (it has none).
- Believing the KV cache makes decoding constant-time or that it caches answers. It removes recomputation of past K/V, but each step still attends over the whole prefix, and the cache is only valid for that exact prefix and model.
- Assuming RoPE means "works at any context length." Relative encoding doesn't guarantee quality at lengths the model never trained on.
- Celebrating a suspiciously low training loss. Unshifted labels or a broken causal mask leak the answer.

# Checklist

- Draw a pre-norm decoder block with shapes (T × d in, T × d out) and label where tokens mix and where they don't.
- Explain with an example why attention alone is order-blind, and compare learned, sinusoidal and rotary position schemes.
- Show with the clock-hand example why RoPE scores depend on relative distance.
- Compute LayerNorm and RMSNorm on a two-element vector by hand.
- Write the SwiGLU formula and explain why LLaMA uses an intermediate width of ⅔ · 4d.
- Count the parameters of a model from its config (embeddings, 4d² attention, 3dm MLP, norms, unembedding, tied or not), and sanity-check with 12·L·d².
- Convert parameters to bf16 weight memory, and size the KV cache per token and per sequence separately.
- Explain teacher forcing, shifted targets, cross-entropy and perplexity with a tiny example.
- Explain why generation is sequential and exactly what the KV cache stores and saves.

# Sources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Original Transformer: sinusoidal position encodings (and the learned-vs-sinusoidal comparison), post-norm residual sublayers, the 512 → 2048 position-wise feed-forward network.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — Rotary position embedding, the θᵢ = 10000^(−2i/d) frequencies, relative-position property, applied to queries and keys only.
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — Pre-normalization with RMSNorm, SwiGLU with ⅔·4d width, RoPE at every layer; 7B has d = 4096, 32 heads, 32 layers.
- [Hugging Face Transformers: Llama model docs and LlamaConfig](https://huggingface.co/docs/transformers/main/en/model_doc/llama) — Config defaults matching Llama-2-7B (vocab 32,000, hidden 4,096, intermediate 11,008, 32 layers, 32 heads, untied embeddings, RMSNorm ε = 1e−6) used for the parameter count.
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) — RMSNorm drops LayerNorm's re-centring; comparable quality with reduced running time.
- [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745) — Why post-norm needs warmup (large gradients near the output at initialization) and pre-norm trains stably.
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) — Gated feed-forward layers, including SwiGLU, outperforming ReLU/GELU MLPs.
- [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)](https://transformer-circuits.pub/2021/framework/index.html) — The residual stream as a communication channel that all components read from and write to.
- [Hugging Face Transformers: How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation) — Why past keys and values can be reused under causal masking, per-layer caches, and cache tensor shapes.

# Flashcards

## order-blind

**Q:** Why can't attention by itself tell "dog bites man" from "bites dog man" at the last token?

Attention computes a softmax-weighted **sum** of value vectors, and each token's key and value travel with it. Reordering the context reorders the scores and the values together, so the sum is unchanged. Attention treats context as a set. The causal mask leaks a little indirect order information in deeper layers, but real models add explicit position information (learned, sinusoidal or RoPE) so order is represented directly.

## position

**Q:** What does RoPE rotate, and what does it not guarantee?

It rotates each pair of **query and key** features by an angle proportional to the token's position (frequencies θᵢ = 10000^(−2i/d)), in every layer. Values are left alone. Because both vectors are rotated, their dot product depends on the *difference* in positions, which gives relative-position behaviour: in the 30°-per-position example, positions 5 and 3 score cos 60° = 0.5, and so do 15 and 13.

It does **not** guarantee good quality at context lengths the model wasn't trained on. Unseen lengths produce angle combinations the model never learned, so extending context needs frequency rescaling, further training, and evaluation.

## absolute-vs-rotary

**Q:** Contrast learned absolute, sinusoidal, and rotary position schemes in where and how they inject position.

Learned absolute positions add a trained vector per slot to the embeddings once, at the bottom, so there's a hard maximum length. Sinusoidal positions add fixed sine/cosine waves at the bottom, with no parameters (the 2017 paper found them roughly equal to learned ones). RoPE adds nothing to the embeddings. It rotates Q and K inside every attention layer, so position affects *who attends to whom* directly and appears as relative distance.

## residual

**Q:** What distinguishes residual connections from normalization?

A residual connection computes `x + f(x)`: the sublayer writes an **update** onto a shared running stream instead of replacing it. That preserves information that a layer has nothing to add to, and gives gradients a direct path (the derivative includes a `1`). Normalization rescales the vector a sublayer *reads* so its inputs stay in a predictable range. One provides a route, the other controls scale, so they solve different problems and you need both.

## norm

**Q:** How does RMSNorm differ from LayerNorm? Show it on [3, 4].

LayerNorm subtracts the mean and divides by the standard deviation, then applies a gain and bias. RMSNorm skips the mean subtraction and divides by the root-mean-square, with a gain only. On `[3, 4]` (unit gain, no ε), LayerNorm gives mean 3.5 and std 0.5, so `[−1, 1]`. RMSNorm gives RMS = √12.5 ≈ 3.5355, so `[0.8485, 1.1314]`. They're different functions, and a trained checkpoint can't swap one for the other. RMSNorm is cheaper and, per its paper, gives comparable quality.

## pre-norm

**Q:** What's the difference between post-norm and pre-norm, and why do modern LLMs use pre-norm?

Post-norm (2017 Transformer) computes `Norm(x + Sublayer(x))`, which renormalizes the residual stream after every sublayer. Pre-norm computes `x + Sublayer(Norm(x))`, which normalizes only the copy the sublayer reads and leaves the stream as a clean running sum. Xiong et al. showed that post-norm has large gradients near the output at initialization, so it needs careful warmup, while pre-norm trains stably. Because the stream is never normalized inside a pre-norm stack, a final norm is needed before the unembedding.

## mlp

**Q:** Does a standard transformer MLP mix token positions?

No. It applies the same weights to each position **independently**: expand, apply a nonlinearity, compress back to d. Attention is the only sublayer that moves information between positions. The MLP is where each token processes what it gathered, and it holds about two-thirds of each block's parameters in a Llama-style model.

## swiglu-width

**Q:** Why does Llama-2-7B use an MLP intermediate width of 11,008 when d = 4,096?

SwiGLU has three matrices (gate, up, down), so it costs `3·d·m` parameters instead of a plain MLP's `2·d·m`. To match a classic 4× MLP (`2·d·4d = 8d²`), LLaMA sets `m = ⅔·4d`, so `3·d·(8d/3) = 8d²`. For d = 4,096 that's 10,922.67, and the released config rounds up to 11,008 (= 43 × 256). Matching parameters is also how you compare MLP variants fairly.

## shapes

**Q:** What shapes flow through a decoder, and why does the T × d invariant matter?

Token IDs `[T]` → embeddings `[T × d]` → every block keeps `[T × d]` (rows are positions, columns are features) → final norm → logits `[T × V]`. With a batch, add a leading `B`. Because each block maps T × d to T × d, blocks stack freely, so depth `L` is just a loop count. During generation only the last position's logits are needed to pick the next token.

## param-count

**Q:** Count the parameters of a Llama-2-7B-shaped model (V = 32,000, d = 4,096, m = 11,008, L = 32, untied).

Per block: attention `4d² = 67,108,864`, SwiGLU `3dm = 135,266,304`, two norm gains `2d = 8,192`, for a total of 202,383,360. Times 32 gives 6,476,267,520. Add the embedding `Vd = 131,072,000`, the unembedding (another 131,072,000, since the matrices are untied) and the final norm (4,096) to get **6,738,415,616 ≈ 6.74 B**. At bf16 that's ≈ 13.48 GB (12.55 GiB) of weights.

## twelve-d-squared

**Q:** What is the 12·L·d² rule of thumb and how accurate is it for a 7B model?

Each block has attention `4d²` (W_Q, W_K, W_V, W_O) plus an MLP sized to about `8d²`, so about `12d²` per block, or `12·L·d²` for the stack, ignoring embeddings and norms. For L = 32 and d = 4,096 it gives 6,442,450,944, within about 0.5% of the exact 6,476,267,520 block total. It's handy for sizing unfamiliar models from two numbers.

## parallel

**Q:** Why can training predict every position at once while generation can't?

Teacher forcing: during training the whole true sequence is known, so the targets are just the inputs shifted by one, and the causal mask stops each position from seeing its own answer. One forward pass scores all T predictions. At generation time the next token doesn't exist until you've sampled it, and the following prediction depends on it, so tokens must be produced one dependent step at a time.

## cross-entropy-perplexity

**Q:** If the correct-token probabilities at three positions are 0.5, 0.25 and 0.125, what are the loss and perplexity?

Cross-entropy is the mean of −ln P: (0.6931 + 1.3863 + 2.0794) / 3 = 1.3863 nats. Perplexity is exp(1.3863) = 4, so the model is as uncertain as a uniform choice among 4 tokens. A loss far better than expected early in training usually means leaked targets (unshifted labels or a broken causal mask), not a great model.

## kv

**Q:** What does a KV cache save, and what work remains?

It stores each past token's projected **keys and values at every layer**. Under causal attention those never change once computed, so the model doesn't recompute them for every new token. Each decode step still computes the new token's own Q, K and V through all layers and attends over **all** cached positions, so per-step cost and memory reads grow with context length. The cache is only valid for that exact prefix and model, and it isn't a store of answers.

## kv-size

**Q:** How big is the KV cache for Llama-2-7B at 4,096 tokens in bf16?

`2 · L · B · T · h_kv · d_head · bytes = 2 × 32 × 1 × 4,096 × 32 × 128 × 2 = 2,147,483,648 bytes = 2 GiB`, or 512 KiB per token. That's on top of ~12.55 GiB of weights, so a handful of long conversations can rival the model itself. Fewer KV heads (GQA) shrink it proportionally, while sampling settings like temperature don't change it at all.
