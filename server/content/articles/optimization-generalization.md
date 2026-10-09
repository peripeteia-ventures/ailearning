---
{
  "slug": "optimization-generalization",
  "title": "How a model learns: loss, gradients, and optimizers",
  "category": "training",
  "summary": "Training is a loop: score the model's next-token guesses with cross-entropy, use backprop to find which way to nudge every weight, and let an optimizer like AdamW take the step. We do the chain rule and one update by hand, then cover learning-rate schedules, batch size, mixed precision, clipping, loss spikes, and how to tell learning from memorizing.",
  "difficulty": "Core",
  "minutes": 30,
  "prerequisites": ["transformer-foundations"],
  "learningObjectives": [
    "Compute cross-entropy and perplexity for a next-token prediction and explain why the loss is averaged over valid target tokens.",
    "Run backpropagation by hand on a small computation graph and take one gradient-descent step with real numbers.",
    "Explain learning-rate stability, warmup and cosine decay, and how momentum, Adam and AdamW turn gradients into updates.",
    "Calculate effective batch size, token-weighted gradient accumulation, and the per-parameter memory cost of AdamW in mixed precision.",
    "Diagnose loss spikes and overfitting, and defend a validation and test-set policy."
  ]
}
---

# Sections

## Training is a loop that turns knobs {#the-training-loop}

Here's the whole idea before any math. A model is a big function with billions of adjustable numbers inside it. Those numbers are called **parameters** (or **weights**). Think of them as knobs on a mixing desk. Right after initialization the knobs are set randomly, so the model outputs noise. Training is the process of turning every knob a tiny bit, over and over, so that the model's predictions get less wrong.

Every training step runs the same four stages:

```text
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
  1. FORWARD            2. LOSS              3. BACKWARD        4. UPDATE
  run the batch  ──►  one number that  ──►  for every knob:  ──►  optimizer nudges
  through model       says "how wrong"      "which way makes     every knob a
  (get predictions)   (cross-entropy)        the loss go up?"     little bit
                                             (gradients)         (SGD / AdamW)
```

1. **Forward pass.** Feed a batch of text through the model and get its predictions.
2. **Loss.** Compress "how wrong were those predictions" into a single number. For language models that number is **cross-entropy**.
3. **Backward pass.** For every parameter, work out how the loss would change if you nudged that parameter up a little. That sensitivity is the parameter's **gradient**. The algorithm that computes all of them efficiently is **backpropagation**.
4. **Update.** An **optimizer** uses the gradients to decide how far to actually move each knob, then moves them.

People blur stages 3 and 4 constantly, so let's be precise now: **backprop is bookkeeping, and the optimizer is policy.** Backprop tells you the slope under your feet. The optimizer decides how big a step to take, whether to trust the slope you felt a moment ago, and whether to pull the weights back toward zero. You can pair the same backward pass with plain SGD, Adam, or anything else.

Recall from **The transformer block: assembling the full model** that during training the model predicts the next token at every position of a sequence in parallel, with the targets being the same text shifted left by one. So one sequence of 2,048 tokens produces up to 2,048 separate predictions, and each one gets graded. This article is about what happens to those grades.

> 🎬 **Animation — the training loop:** a circular diagram with four stations (Forward → Loss → Backward → Update). Step 1: a batch of three token sequences flows into a box labelled "model (knobs)". Step 2: predictions come out and collapse into a single loss number, 2.31. Step 3: arrows flow backwards through the model box, and a small arrow appears next to each of six drawn knobs (some pointing up, some down), labelled "gradient". Step 4: each knob rotates slightly opposite its arrow. Step 5: the loop repeats and the loss counter ticks down 2.31 → 2.27 → 2.24 while the knobs keep moving.

## Scoring a guess: cross-entropy and perplexity {#cross-entropy}

At each position the model outputs one raw score per vocabulary entry. These raw scores are **logits**: any real number, positive or negative, with no requirement to sum to anything. **Softmax** turns them into probabilities by exponentiating each one (making it positive) and dividing by the total (making them sum to 1).

Now the grading rule. We look up the probability the model gave to the token that *actually* came next, and charge it **the negative log of that probability**. That's cross-entropy for one position.

```formula
pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
ℓ = −log p_y
L = (1/M) · Σₜ ℓₜ
perplexity = exp(L)
```

Here `z` is the vector of logits (one per vocabulary entry, length `V`), `pᵢ` is the probability of entry `i`, `y` is the index of the correct next token, and `ℓ` is the loss at one position. `L` is the mean over the `M` **valid target tokens** in the batch (more on "valid" in a moment). We use the natural log, so the unit is **nats per token**.

**A tiny worked example.** Vocabulary of three tokens: `sat`, `ran`, `the`. After the context "the cat" the model outputs logits `[2, 1, 0]`, and the true next token is `sat` (index 0).

| token | logit | exp(logit) | probability |
|---|---|---|---|
| sat ✓ | 2 | 7.389 | 0.6652 |
| ran | 1 | 2.718 | 0.2447 |
| the | 0 | 1.000 | 0.0900 |
| **sum** | | **11.107** | **1.0000** |

Loss = −ln(0.6652) = **0.4076 nats**.

Why a log and not just "1 − probability"? Because the log punishes confident mistakes brutally, and that's exactly what we want:

| probability on the right token | loss (nats) |
|---|---|
| 1.0 | 0 |
| 0.8 | 0.223 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |
| 0.0001 | 9.210 |

Going from 0.8 to 0.01 multiplies the loss by about 20. Accuracy would call both of these "right" or "wrong" and hide the difference. Cross-entropy also has a nice pedigree: minimizing it is the same as maximizing the likelihood the model assigns to the training text.

**Perplexity** is just `exp(mean loss)`. A loss of 2 nats/token gives perplexity e² ≈ 7.39. Read it as "the model is, on average, as uncertain as if it were choosing uniformly among about 7.4 tokens." A useful sanity anchor: a model that knows nothing and spreads probability uniformly over a vocabulary of `V = 50,000` has loss ln(50,000) ≈ 10.8 nats. If your freshly initialized model's first logged loss is near ln(V), your plumbing is probably right. One catch interviewers love: perplexities are only comparable **with the same tokenizer and the same evaluation text**. A tokenizer that chops text into fewer, bigger tokens changes what "per token" means.

**What "valid target tokens" means.** Batches are rectangular, so short sequences get padded with filler tokens. Padding must contribute nothing: not to the sum, and not to the count `M`. The same goes for tokens you deliberately don't train on, like the prompt part of a chat example during fine-tuning (see **From base model to assistant: SFT, RLHF, DPO, and verifiable rewards**). PyTorch's `CrossEntropyLoss` handles this with `ignore_index` (default −100): those positions are skipped and the mean is taken over the rest.

**Two implementation notes that come up in code review.** First, library cross-entropy functions take **logits**, not probabilities. Internally they compute `log p_y = z_y − log Σ exp(zⱼ)` using the **log-sum-exp trick**: subtract the largest logit before exponentiating so `exp` never overflows. If you apply softmax yourself and pass the result in, the library will softmax it again and you'll silently train on the wrong objective. Second, the mean is over tokens, not sequences. If you average per-sequence means, a 10-token sequence counts as much as a 2,000-token one. That's a different objective, and it matters when we get to gradient accumulation.

> 🎬 **Animation — from logits to loss:** three vertical bars labelled sat/ran/the with heights 2, 1, 0 (logits). Step 1: each bar is replaced by exp(logit): 7.39, 2.72, 1.00. Step 2: bars shrink proportionally so they sum to 1: 0.665, 0.245, 0.090. Step 3: the "sat" bar is highlighted with a ✓ and a readout shows −ln(0.665) = 0.408. Step 4: a slider drags the "sat" probability from 0.9 down to 0.01 while a curve of −ln(p) is traced on the right, with the loss readout climbing steeply toward 4.6 as p approaches 0.

## Backprop by hand on a tiny graph {#backprop-by-hand}

We have one loss number. We want, for every parameter, "if I nudge this knob up by a hair, how much does the loss change?" That's a partial derivative, `∂L/∂w`. Stacking all of them gives the **gradient**, a vector with one entry per parameter, pointing in the direction where the loss rises fastest.

The naive way would be to wiggle each parameter one at a time and re-run the model. With 7 billion parameters that's 7 billion forward passes per step. Not happening. **Backpropagation** gets all of them in roughly the cost of one extra pass, using the **chain rule**: if `a` affects `b`, and `b` affects `c`, then `∂c/∂a = (∂c/∂b) · (∂b/∂a)`. You multiply local slopes along the path.

The analogy I like is a relay of blame. The final station knows how bad the result was. It tells the station before it "here's how much I'd improve per unit change in what you handed me." That station multiplies by its own local sensitivity and passes the blame further back. Nobody needs to see the whole factory, just their own inputs and outputs.

Let's do it with real numbers. Here's the smallest model that still looks like a neuron: one input `x`, a weight `w`, a bias `b`, a **sigmoid** (the two-class cousin of softmax, `σ(z) = 1 / (1 + e⁻ᶻ)`, which squashes any number into a probability between 0 and 1), and a cross-entropy loss for label `y = 1` ("this is the correct class").

```text
  x = 2 ──┐
          ├─► [ × ] ── u = w·x ──┐
  w = 0.5 ┘                      ├─► [ + ] ── z ──► [ σ ] ── p ──► [ −log ] ── ℓ
                        b = −0.5 ┘
```

**Forward pass (left to right).** Compute and *save* each intermediate value, because the backward pass will need them.

| node | computation | value |
|---|---|---|
| u | w · x = 0.5 × 2 | 1.0 |
| z | u + b = 1.0 − 0.5 | 0.5 |
| p | σ(0.5) | 0.6225 |
| ℓ | −ln(0.6225) | 0.4741 |

**Backward pass (right to left).** Start with `∂ℓ/∂ℓ = 1` and multiply by each node's local derivative as you walk back.

| step | local derivative | value | running product = ∂ℓ/∂(this node) |
|---|---|---|---|
| ℓ → p | ∂ℓ/∂p = −1/p | −1.6065 | ∂ℓ/∂p = −1.6065 |
| p → z | ∂p/∂z = p(1 − p) | 0.2350 | ∂ℓ/∂z = −1.6065 × 0.2350 = **−0.3775** |
| z → b | ∂z/∂b = 1 | 1 | ∂ℓ/∂b = **−0.3775** |
| z → u | ∂z/∂u = 1 | 1 | ∂ℓ/∂u = −0.3775 |
| u → w | ∂u/∂w = x | 2 | ∂ℓ/∂w = −0.3775 × 2 = **−0.7551** |
| u → x | ∂u/∂x = w | 0.5 | ∂ℓ/∂x = −0.1888 |

Look at `∂ℓ/∂z = −0.3775`. That's exactly `p − y = 0.6225 − 1`. The messy `−1/p` and `p(1 − p)` cancel into something beautiful, and it's no coincidence. Softmax plus cross-entropy has the same shape over a whole vocabulary:

```formula
∂ℓ/∂zᵢ = pᵢ − 1[i = y]
```

Here `1[i = y]` is 1 for the correct token and 0 for everything else (a **one-hot** vector). In words: *the gradient on each logit is "what you predicted minus what was true."* For our three-token example, `p = [0.6652, 0.2447, 0.0900]` and the target is `[1, 0, 0]`, so the logit gradient is `[−0.3348, +0.2447, +0.0900]`. The correct token gets a negative gradient, so stepping *against* the gradient raises its logit, and the wrong tokens get pushed down in proportion to how much probability they stole. The entries sum to zero, which makes sense: adding the same constant to every logit doesn't change softmax at all.

**Two rules that make backprop work on any graph.**

- **Multiply along a path.** That's the chain rule, as above.
- **Add where paths merge.** If a value is used in two places, its gradient is the sum of the blame from both uses. Example: `L = 3w + w²` at `w = 2`. The `3w` path contributes 3, the `w²` path contributes `2w = 4`, so `∂L/∂w = 7`. In a transformer this happens all the time. The residual stream feeds every block, and the embedding matrix is often reused as the output projection. PyTorch does exactly this: gradients are *accumulated* (summed) into each parameter's `.grad` field, which is also why you must zero them between steps.

**Why this is cheap, and where the cost moves to.** This method is **reverse-mode differentiation**. It's efficient whenever one scalar output depends on many inputs, which is exactly the "one loss, billions of weights" situation. A common rule of thumb is that the backward pass costs about twice the forward pass in FLOPs, so a training step is about three forward passes' worth of compute. That's where the `6·N·D` training-compute estimate comes from, and we'll go through it in **Pretraining: data, compute, and scaling laws**. The price is **memory**. Every saved intermediate (like `u`, `z`, `p` above) is an **activation** that must live until the backward pass consumes it. For a big model on long sequences, activations can outweigh the weights. **Activation checkpointing** throws some away and recomputes them during the backward pass, trading compute for memory. We'll cover that in **Distributed training: fitting a training run onto a cluster**.

> 🎬 **Animation — backprop on the tiny graph:** draw the graph x, w, b → [×] → u → [+] → z → [σ] → p → [−log] → ℓ. Step 1 (forward): values light up left to right in blue: u = 1.0, z = 0.5, p = 0.6225, ℓ = 0.4741, each stamped "saved". Step 2 (backward): an orange "1" appears at ℓ, then flows left; at each node show the local derivative as a small label (−1/p = −1.607, p(1−p) = 0.235, ×1, ×x = 2) and the running product in orange (−1.607, −0.378, −0.378, −0.755). Step 3: highlight the identity ∂ℓ/∂z = p − y = −0.378. Step 4: a side panel with L = 3w + w² shows two orange arrows (3 and 4) merging into a "+" to give 7.

## One step of gradient descent, with real numbers {#one-step}

Now we have slopes. **Gradient descent** says: move every parameter a small distance *against* its gradient.

```formula
θ ← θ − η · g
```

`θ` (theta) is the vector of all parameters, `g` is the gradient `∂L/∂θ`, and `η` (eta) is the **learning rate**, a positive number that sets the step size. The minus sign is the whole trick: the gradient points uphill, so we step downhill.

Take the graph from the last section with `η = 0.5`:

| parameter | old value | gradient | − η·g | new value |
|---|---|---|---|---|
| w | 0.5 | −0.7551 | +0.3775 | 0.8775 |
| b | −0.5 | −0.3775 | +0.1888 | −0.3112 |

Re-run the forward pass: `z = 0.8775 × 2 − 0.3112 = 1.4439`, `p = σ(1.4439) = 0.8091`, loss = −ln(0.8091) = **0.2119**. The loss went from 0.4741 to 0.2119, and the model is now 81% confident instead of 62%. That's learning. There's no more to it than that, just repeated a few hundred thousand times over billions of knobs.

```text
 loss
 0.47 ●  before  (p = 0.62)
      │ ╲
      │   ╲   one step, η = 0.5
      │     ╲
 0.21 │       ●  after (p = 0.81)
      └──────────────────► w
        0.5            0.88
```

Here's a subtle point that separates people who've trained models from people who've read about it. The gradient is a **local, linear** prediction. The first-order estimate of the loss drop is `η · ‖g‖² = 0.5 × (0.7551² + 0.3775²) = 0.356`. The actual drop was 0.262. The surface curves, so the straight-line prediction overshoots. With a small step that error is small. With a big step, the curvature can dominate and the loss can go *up*. That's the whole story of the next section.

The same update applied directly to our three-token logits (pretending the logits themselves are parameters, with `η = 1`) gives new logits `[2.335, 0.755, −0.090]`, new probability on `sat` = 0.772, and loss 0.408 → 0.258. The correct logit went up, and the wrong ones went down in proportion to their probability.

**Where "stochastic" comes in.** The true loss is an average over the entire training set. Computing its exact gradient would take a pass over trillions of tokens per step. Instead we estimate the gradient from a random **mini-batch** of sequences. That estimate is noisy but unbiased, and it's cheap. Gradient descent on mini-batch gradients is **stochastic gradient descent (SGD)**. Every optimizer in this article is a variation on "SGD, but smarter about the step."

> 🎬 **Animation — one gradient-descent step:** a 2D contour plot of loss over (w, b) with a dot at (0.5, −0.5), loss 0.474. Step 1: draw the gradient arrow (−0.755, −0.378) pointing uphill, drawn in red. Step 2: flip it and scale by η = 0.5 to get the step (+0.378, +0.189), drawn in green. Step 3: the dot moves to (0.878, −0.311) and the loss readout changes to 0.212. Step 4: a dashed tangent-plane line predicts 0.474 − 0.356 = 0.118, and a small bracket shows the gap to the actual 0.212, labelled "curvature".

## How big a step: learning rate, warmup, and cosine decay {#learning-rate}

The learning rate is the most important hyperparameter you'll touch. Too small and training crawls. Too big and it blows up. Here's the precise reason it blows up.

Picture a valley. Small strides take you down toward the floor. A giant stride launches you past the floor and up the opposite wall, higher than where you started. The steeper the walls (the higher the **curvature**), the smaller the stride has to be.

Make that exact with a 1-D bowl, `L(w) = a·w²/2`, whose gradient is `a·w`. One gradient step gives:

```formula
w_next = w − η·a·w = (1 − η·a) · w
```

Each step multiplies `w` by `(1 − η·a)`. If that factor's magnitude is below 1, `w` shrinks toward the minimum at 0. If it's above 1, `w` grows without bound. So gradient descent converges only when **0 < η·a < 2**. At exactly `η·a = 2` it bounces between `+w` and `−w` forever.

Two runs on `a = 1`, starting at `w = 1` (loss 0.5):

| step | η = 0.5: w | loss | η = 2.2: w | loss |
|---|---|---|---|---|
| 0 | 1 | 0.5 | 1 | 0.5 |
| 1 | 0.5 | 0.125 | −1.2 | 0.72 |
| 2 | 0.25 | 0.03125 | 1.44 | 1.037 |
| 3 | 0.125 | 0.0078 | −1.728 | 1.493 |

Both runs "follow the gradient" every step. One converges, the other diverges. A real network isn't a bowl, but locally it looks like one with millions of different curvatures in different directions. The **sharpest** direction sets the stability limit, while the flattest directions are the ones making painfully slow progress. That tension is why adaptive optimizers exist.

> 🎬 **Animation — stable vs unstable step size:** two side-by-side parabolas L(w) = w²/2. Left panel (η = 0.5): a ball starts at w = 1 and hops to 0.5, 0.25, 0.125, with a loss readout 0.5 → 0.125 → 0.031 → 0.008. Right panel (η = 2.2): the ball hops to −1.2, 1.44, −1.728, climbing higher on alternating walls, with a loss readout 0.5 → 0.72 → 1.04 → 1.49 flashing red. A caption shows the multiplier (1 − ηa): 0.5 on the left, −1.2 on the right.

**Schedules: the learning rate changes over time.** Nobody trains an LLM at one fixed learning rate. The standard shape has two phases.

- **Warmup.** Start near zero and ramp linearly to the peak over the first few hundred or few thousand steps. Early on the weights are random, gradients are large and erratic, and Adam's running statistics (next section) haven't settled yet. A full-size step at step 1 is a common way to blow up a run in the first minute.
- **Decay.** After the peak, lower the rate gradually so that late training makes fine adjustments instead of bouncing around the valley floor. **Cosine decay**, from the SGDR paper, follows half a cosine wave from the peak down to a minimum:

```formula
warmup (t ≤ W):   η(t) = η_max · t / W
decay  (t > W):   η(t) = η_min + ½ (η_max − η_min) · (1 + cos(π · (t − W) / (T − W)))
```

`t` is the current step, `W` the number of warmup steps, `T` the total number of steps, and `η_max`, `η_min` the peak and floor. Toy example: `η_max = 0.001`, `η_min = 0`, `W = 100`, `T = 1000`.

| step | phase | learning rate |
|---|---|---|
| 50 | warmup | 0.000500 |
| 100 | peak | 0.001000 |
| 325 | decay, ¼ through | 0.000854 |
| 550 | decay, ½ through | 0.000500 |
| 775 | decay, ¾ through | 0.000146 |
| 1000 | end | 0 |

```text
 η
 0.001 │      ●●●●
       │     ●     ●●
       │    ●         ●●
 0.0005│   ●            ●●
       │  ●               ●●
       │ ●                  ●●●
     0 ●─────────────────────────●──► step
       0  100     550          1000
        warmup    cosine decay
```

Cosine is a sensible default, not a law of nature, and the floor is often a fraction of the peak rather than zero. The senior-level detail is **what unit "t" is measured in**. If the schedule counts optimizer steps and you double the batch size, each step now sees twice the tokens, so your warmup covers twice as much data and the decay finishes on a different token budget. Always say whether a schedule is defined in steps or tokens, and re-plan it deliberately if you stop early or extend training.

## Momentum, Adam, and AdamW: smarter steps {#adam-and-adamw}

Plain SGD has two annoyances. Mini-batch gradients are noisy, so the path zig-zags. And one learning rate for every parameter is a bad fit when some parameters see gradients 1,000× larger than others. Three ideas fix this, each building on the last.

**Momentum: a heavy ball.** Instead of stepping along today's gradient, keep a running average of recent gradients and step along that. Like a heavy ball rolling downhill, it builds speed in directions that stay consistent and averages out side-to-side jitter. The running average is an **exponential moving average (EMA)**: `m ← β·m + (1 − β)·g`, where `β` (say 0.9) controls how much memory it has. With `β = 0.9` it roughly averages the last 10 gradients.

**Adam: a per-parameter volume knob.** Adam keeps two EMAs for *every* parameter: `m`, the average gradient (which way it's been pushed), and `v`, the average *squared* gradient (how loud the pushes have been). These are called the **first and second moments**. Then it divides one by the square root of the other. Parameters with consistently loud gradients get their steps turned down, and quiet ones get turned up.

```formula
m_t = β₁·m_{t−1} + (1 − β₁)·g_t
v_t = β₂·v_{t−1} + (1 − β₂)·g_t²
m̂_t = m_t / (1 − β₁ᵗ)          v̂_t = v_t / (1 − β₂ᵗ)
θ_t = θ_{t−1} − η · m̂_t / (√v̂_t + ε)
```

`β₁` and `β₂` (paper defaults 0.9 and 0.999) are the forgetting rates, `ε` (default 10⁻⁸) keeps you from dividing by zero, and `t` is the step count. The hats are **bias correction**. Both EMAs start at zero, so after one step `m` is only 10% of the true gradient. Dividing by `1 − β₁ᵗ` (0.1 at `t = 1`) scales it back up. As `t` grows, the correction fades to 1.

To be clear about what this is not: dividing by `√v̂` is a cheap, **diagonal** rescaling, one number per parameter. It is not the full second-order Newton method, which would need the **Hessian** (the matrix of all second derivatives, `N × N` entries, which is hopeless at `N` = billions).

**AdamW: weight decay done properly.** **Weight decay** is regularization: every step, shrink each weight slightly toward zero so the model doesn't lean on huge weights. The old way to get it was **L2 regularization**, adding `λ·θ²/2` to the loss, which adds `λ·θ` to the gradient. For plain SGD that's the same thing as shrinking. But in Adam, that extra `λθ` term goes *through* the `√v̂` normalization, so parameters with big gradient history barely get regularized. Loshchilov and Hutter showed the two aren't equivalent for adaptive optimizers and proposed **decoupling**: shrink the weights directly, outside Adam's machinery. That's AdamW, the default for LLM training. PyTorch's version applies `θ ← θ − η·λ·θ`, then the Adam step.

**One AdamW step by hand.** Two parameters, `η = 0.01`, `λ = 0.1`, `β₁ = 0.9`, `β₂ = 0.999`, first step (`t = 1`), zero initial moments, and `ε` negligible.

| | parameter A | parameter B |
|---|---|---|
| θ, g | 2, 0.5 | −1, 0.001 |
| decay: θ − ηλθ | 2 − 0.002 = 1.998 | −1 + 0.001 = −0.999 |
| m = 0.1·g | 0.05 | 0.0001 |
| v = 0.001·g² | 0.00025 | 1×10⁻⁹ |
| m̂ = m/0.1, v̂ = v/0.001 | 0.5, 0.25 | 0.001, 1×10⁻⁶ |
| step = η·m̂/√v̂ | 0.01 × 0.5/0.5 = 0.01 | 0.01 × 0.001/0.001 = 0.01 |
| new θ | **1.988** | **−1.009** |

Stare at the "step" row. Parameter A's gradient is 500× bigger than B's, yet both move by exactly 0.01. On the first step Adam behaves like "move by η in the sign of the gradient." That's the volume knob at its most extreme, and it's why η for Adam roughly means "the typical per-step change in a weight," which is a much easier thing to reason about than SGD's η.

**The price: optimizer state.** Adam stores `m` and `v` for every parameter, usually in fp32, so 8 extra bytes per parameter. In the usual mixed-precision setup (next sections) the tally is roughly:

| item | bytes per parameter |
|---|---|
| bf16 weights (used in forward/backward) | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam m (fp32) | 4 |
| Adam v (fp32) | 4 |
| **total, before activations** | **≈ 16** |

For a 7B-parameter model that's 7×10⁹ × 16 bytes ≈ **112 GB**, before a single activation, which is more than one 80 GB GPU. That number is the starting point of **Distributed training: fitting a training run onto a cluster**.

> 🎬 **Animation — SGD vs momentum vs Adam on a ravine:** a long, narrow elliptical contour plot (steep in y, shallow in x). Three dots start at the same corner. Step 1: SGD (grey) zig-zags across the narrow axis and creeps along the long one. Step 2: momentum (blue) zig-zags less and accelerates along the valley floor. Step 3: Adam (orange) takes near-equal-sized steps in x and y, heading almost straight for the minimum. Step 4: an inset table repeats the AdamW example, showing parameters A (g = 0.5) and B (g = 0.001) both moving by 0.01, with A's final value 1.988 and B's −1.009.

## Batch size, gradient noise, and accumulation {#batch-size}

The mini-batch gradient is an average of per-example gradients, and each of those is a noisy vote about which way is downhill. Averaging more votes reduces the noise, but slower than you'd hope. If individual gradients are independent with standard deviation σ, the mean of `B` of them has standard deviation **σ/√B**. It's the polling rule: survey 4× more people and your margin of error only halves.

| batch size B | relative noise (σ/√B) | cost per step |
|---|---|---|
| 1 | 1.00 | 1× |
| 4 | 0.50 | 4× |
| 16 | 0.25 | 16× |
| 64 | 0.125 | 64× |

So bigger batches give cleaner gradients and keep GPUs busy, but for a fixed token budget they also mean **fewer optimizer steps**. Past some point, doubling the batch buys almost no extra progress per token. One caveat: tokens within one document are correlated, so a batch of a million tokens is not a million independent votes. The batch-size and learning-rate interaction is real but not a clean exchange rate. Smith et al. showed that growing the batch can stand in for decaying the learning rate in their experiments, but that's not a guarantee your AdamW run will behave identically.

**Gradient accumulation: a big batch in small pieces.** Suppose you want 256 sequences per update but only 4 fit in GPU memory. Run 4 at a time as **micro-batches**, add their gradients into `.grad` *without* stepping, and after the last one take a single optimizer step. It's like collecting receipts from several shopping trips and balancing the budget once. Training usually also runs on several **data-parallel** replicas: full copies of the model, each on different data, whose gradients get averaged across the copies each step.

```formula
effective batch = data-parallel replicas × sequences per micro-batch × accumulation steps
```

Worked example: 8 replicas × 4 sequences × 8 accumulation steps = **256 sequences per update**. At 2,048 valid target tokens each, that's 524,288 tokens per update. Count *replicas*, not GPUs. If each replica is itself split across 4 GPUs by tensor parallelism, 32 GPUs still means 8 replicas.

```text
 replica 0:  [mb1][mb2]...[mb8] ─┐
 replica 1:  [mb1][mb2]...[mb8] ─┤   sum grads locally,
    ...                          ├─► average across replicas ─► ONE optimizer step
 replica 7:  [mb1][mb2]...[mb8] ─┘   (all-reduce)               ONE scheduler tick
      each mb = 4 sequences
```

**The weighting trap.** Micro-batches rarely have equal numbers of valid tokens. Suppose micro-batch 1 has 100 valid tokens with mean loss 2, and micro-batch 2 has 300 with mean loss 4. The correct token-mean is `(100·2 + 300·4) / 400 = 3.5`. Naively averaging the two means gives 3, which over-weights the short micro-batch's tokens 3× relative to the long one's. The gradients inherit the same bias. The fix is to sum per-token losses and divide by the total valid-token count for the *whole* effective batch (across micro-batches and replicas).

```python
# Sketch: token-weighted accumulation (single replica)
total_tokens = sum(mb.num_valid_tokens for mb in microbatches)
optimizer.zero_grad()
for mb in microbatches:
    logits = model(mb.inputs)
    loss_sum = F.cross_entropy(logits.flatten(0, 1), mb.targets.flatten(),
                               ignore_index=-100, reduction="sum")
    (loss_sum / total_tokens).backward()   # grads add up in .grad
optimizer.step()                           # once per effective batch
scheduler.step()                           # once per effective batch
```

Two more things to keep straight. Accumulation does **not** reduce the memory of a micro-batch. What saves memory is making the micro-batch smaller, and accumulation is how you keep the effective batch the same while you do. Also, the optimizer, weight decay, and scheduler must tick **once per effective batch**, not once per micro-batch. Otherwise decay is applied 8× too often and your schedule finishes 8× early.

> 🎬 **Animation — gradient noise vs batch size:** a true gradient arrow (black) points down-right. Step 1: B = 1, a dozen sampled grey arrows scatter widely around it. Step 2: B = 4, the scatter shrinks to half the radius. Step 3: B = 16, half again. A small chart plots noise radius against B, labelled 1/√B. Step 4: two micro-batch bars (100 tokens at loss 2, 300 tokens at loss 4) merge into one bar; a wrong readout "3.0" is crossed out and "3.5" is shown instead.

## Keeping the numbers healthy: mixed precision, clipping, and loss spikes {#numerics-and-stability}

**Mixed precision.** Doing the matrix multiplies in 16-bit floats halves memory and bandwidth and unlocks the fast tensor-core paths. But 16 bits force a trade between **range** (how big or small a number can be, set by the exponent bits) and **precision** (how finely numbers are spaced, set by the mantissa bits).

| format | sign / exponent / mantissa bits | largest value | spacing just above 1.0 |
|---|---|---|---|
| fp32 | 1 / 8 / 23 | ≈ 3.4×10³⁸ | ≈ 1.2×10⁻⁷ |
| fp16 | 1 / 5 / 10 | 65,504 | ≈ 0.00098 |
| bf16 | 1 / 8 / 7 | ≈ 3.4×10³⁸ | 0.0078 |

**fp16** has decent precision but tiny range. Its smallest positive value is about 6×10⁻⁸, so a gradient of 10⁻⁸ simply becomes 0 (it **underflows**). The fix is **loss scaling**: multiply the loss by a big factor like 65,536 before backward, so every gradient is 65,536× larger (10⁻⁸ becomes about 6.6×10⁻⁴, comfortably representable), then divide the gradients by the same factor in fp32 before the optimizer uses them. Dynamic loss scalers lower the factor when they see infinities and skip that step.

**bf16** keeps fp32's 8 exponent bits, so it has the same range, and loss scaling is typically unnecessary. That's the main reason large-model training moved to bf16. The cost is coarse precision, and that's why we keep **fp32 master weights**. Near 1.0, bf16 numbers are spaced 0.0078 apart. If a weight is 1.0 and the optimizer wants to add 0.001, then in bf16 `1.0 + 0.001` rounds right back to `1.0`, and the update is lost, every step, forever. So the optimizer updates an fp32 copy, and a bf16 copy is made from it for the next forward pass. That's the "mixed" in mixed precision, and it's the recipe from Micikevicius et al. (fp32 master copy, loss scaling, fp32 accumulation).

**Gradient clipping.** Occasionally one batch produces a gigantic gradient, and one giant step can wreck weeks of progress. **Clipping by global norm** caps the total size: compute `‖g‖`, the length of the full gradient vector across all parameters, and if it exceeds a threshold `c`, rescale the whole vector by `c / ‖g‖`. The direction is kept, and only the length is capped. Example: `g = [3, 4]` has norm 5. With `c = 1` it becomes `[0.6, 0.8]`. Order matters. Clip **after** accumulation (clipping each micro-batch separately is nonlinear and changes the direction of the sum), and **after** unscaling (otherwise your threshold is being compared with a gradient inflated 65,536×). PyTorch's AMP docs spell this out: call `unscale_` once, after all micro-batches, then clip, then step.

**Loss spikes.** Sometimes the loss curve, happily declining, suddenly jumps up. Small runs usually recover, and big runs sometimes don't. The PaLM paper reported roughly 20 spikes in its largest model's run *despite gradient clipping*. Their pragmatic fix was to restart from a checkpoint about 100 steps before the spike and skip roughly 200–500 data batches. They also found that replaying the same batches from a different checkpoint didn't spike, so the cause was the combination of specific data with a specific parameter state, not simply "bad data."

A senior debugging checklist for a spike:

| look at | what it tells you |
|---|---|
| global gradient norm per step | did the gradient explode before the loss did? |
| per-layer update-to-weight ratio (‖Δθ‖ / ‖θ‖) | which layer is taking outsized steps |
| clip frequency, inf/NaN count, skipped steps | numeric trouble vs genuine optimization trouble |
| learning rate at the time | spikes near the peak of warmup suggest the rate is too high |
| the batches involved | duplicated or garbage data, extreme sequence lengths |

The usual levers are lowering the peak learning rate, lengthening warmup, tightening the clip threshold, fixing the data, or rolling back and skipping ahead like PaLM did. Also make sure a skipped step (non-finite gradients) doesn't silently advance a scheduler that's supposed to count real updates.

> 🎬 **Animation — why bf16 needs fp32 master weights:** a number line from 0.99 to 1.02 with bf16 tick marks every 0.0078. Step 1: a weight sits at 1.0; an update arrow of +0.001 lands between ticks and snaps back to 1.0 (shown in red: "update lost"). Step 2: switch to an fp32 number line with dense ticks; the same +0.001 lands at 1.001 and ten such updates accumulate to 1.010, which finally rounds to the bf16 tick at 1.0078. Step 3: a side panel shows an fp16 gradient of 1e-8 flushing to 0, then ×65,536 becoming 6.6e-4 and surviving, then ÷65,536 in fp32.

## Training loss going down isn't the goal: generalization {#generalization}

Everything so far has been about pushing the **training loss** down. But nobody ships a model to re-predict its training data. What we care about is **generalization**: how well the model does on inputs it has never seen, drawn from the situations it'll actually face once deployed.

The failure mode is **overfitting**: the model starts memorizing quirks of the training examples instead of patterns that transfer. It's the student who memorizes last year's exam answers. Their practice score is perfect, and their real exam score isn't.

To measure generalization you hold data out:

- **Validation set.** Held-out data you look at *during* development to pick checkpoints, learning rates, and data mixes.
- **Test set.** Held-out data you look at *once*, at the end, to report how the final choice performs. If you keep choosing models based on test results, the test set quietly becomes a second validation set, and your number stops meaning anything.

The classic picture (illustrative numbers, not a measurement):

```text
 loss
  3.0 │●
      │ ●○
  2.5 │   ●○
      │     ●  ○
  2.0 │       ●   ○  ○  ○ ○ ○ ○   ← validation (○) bottoms out, then rises
      │          ●  ●               
  1.5 │               ●  ●  ●  ●   ← training (●) keeps falling
      └────────────────────────────► update
                  300
             best checkpoint
```

When validation bottoms out while training keeps falling, **early stopping** means keeping the checkpoint from the validation minimum. Pick that rule *before* you look at the curves.

**How this plays out for LLMs specifically.** Large pretraining runs often see most data only about once, so each batch is fresh text and training and validation loss track closely. Classic overfitting shows up when data gets **repeated** many times (see **Pretraining: data, compute, and scaling laws**) and, very commonly, in **fine-tuning** on a few thousand examples, where the model can memorize the set in a few epochs. The more insidious failure is **contamination**: benchmark or test text leaks into the training corpus, so a "held-out" score is really a memory test. Duplicate and near-duplicate detection across splits is the defense, and we'll go through it properly in **Evaluation and observability: knowing whether it actually works**.

**Split by the thing that must generalize.** If the same customer's tickets, the same template, or the same source document can land on both sides of a random row split, your validation set is partly a training set. For anything time-dependent, validate on data from *after* the training period. Fit any learned preprocessing on training data only. And compare like with like. Training loss is often logged with dropout on and from a mixture that's still shifting, so compare against an evaluation-mode pass over a fixed validation sample, not against the noisy live training curve.

Finally, the regularization toolbox (more data, better filtering, weight decay, dropout, early stopping) addresses different causes of poor generalization. None of them proves your held-out data is actually clean. That takes auditing the splits.

> 🎬 **Animation — overfitting and the held-out boundary:** top panel: training and validation loss curves draw themselves in over 1,000 updates, with validation reaching its minimum at update 300 and rising after; a vertical marker drops at 300 labelled "early stopping checkpoint". Bottom panel: a dataset drawn as a grid of cards coloured by customer; step 1, a random row split scatters one customer's cards into both train and validation (flagged red, "leak"); step 2, a grouped split moves each customer's cards entirely to one side (green).

# Interview

## Question

An eight-replica language-model training run doubles gradient accumulation to relieve memory pressure. Afterwards the training curve looks smoother, but validation loss is worse at the same reported step. How do you investigate?

## Answer

First I'd pin down what actually changed, because "doubled accumulation to relieve memory" is suspicious on its face. Accumulation by itself doesn't reduce the memory of a micro-batch. Only a smaller micro-batch does that. So either the micro-batch was halved and accumulation doubled to keep the effective batch constant, or the micro-batch stayed the same and the effective batch doubled. I'd compute valid tokens per optimizer update before and after: replicas × sequences per micro-batch × accumulation steps × valid tokens per sequence.

If the effective batch doubled, "the same step" is no longer the same amount of data, so the comparison is unfair. Each step now sees twice the tokens. I'd re-plot both runs against **tokens consumed** (and compute), not steps. A smoother curve is expected from a bigger batch, since the gradient noise drops by about 1/√2. The schedule is also a suspect. If warmup and decay are defined in steps, the new run warms up over twice the tokens and sits on a different part of the cosine at any given token count. Cumulative AdamW decay per token also changed, because decay is applied per step.

If the effective batch didn't change, the runs should be nearly identical, so I'd look for bugs in the accumulation path. Is the loss token-weighted across micro-batches (sum over valid tokens, divided by the total), rather than an average of micro-batch means? Is `zero_grad` called once per effective batch? Do the optimizer, weight decay, and scheduler step once per effective batch? Is clipping applied to the full accumulated, unscaled gradient? Is the data-parallel all-reduce averaging correctly? I'd also check gradient norms, clip frequency, and non-finite or skipped steps.

On the validation side, I'd confirm both runs are evaluated in eval mode on the same fixed validation sample, that data ordering or the mixture didn't shift, and that the difference is larger than evaluation noise. Then I'd run a controlled comparison, matching token budgets and a token-based schedule and varying only the batch configuration, before blaming "less gradient noise." The final configuration gets picked on validation, and the test set is only touched at the end.

## Follow-ups

- When is gradient accumulation exactly equivalent to a physical large batch, and when does it fail (batch-dependent layers, dropout RNG, reduction order)?
- How would you normalize the loss correctly when sequences have different valid-token counts across replicas?
- What must a checkpoint contain for an exact resume (optimizer moments, step count, scheduler, RNG, data-loader position, loss-scaler state)?
- Why does Adam's first update have magnitude about η regardless of gradient size, and what does that imply for warmup?
- What evidence would distinguish overfitting from a shifted validation distribution?

# Pitfalls

- Treating backpropagation and the optimizer as the same thing. Backprop computes derivatives. The optimizer (SGD, AdamW) decides how to move the parameters using them.
- Passing softmax probabilities into a cross-entropy function that expects logits. It applies softmax again internally, so you silently train on the wrong objective.
- Averaging micro-batch mean losses when their valid-token counts differ. This over-weights short micro-batches. Sum over tokens and divide by the total instead.
- Believing gradient accumulation lowers activation memory. Only a smaller micro-batch does. Accumulation just restores the effective batch.
- Stepping the optimizer, weight decay, or scheduler once per micro-batch instead of once per effective batch.
- Using L2 loss regularization with Adam and calling it weight decay. With adaptive optimizers the two aren't equivalent, and AdamW decouples them.
- Clipping gradients before unscaling (fp16 loss scaling) or per micro-batch. The threshold becomes meaningless or the direction changes.
- Comparing perplexities across different tokenizers, or picking models repeatedly on the test set and still calling it held-out.

# Checklist

- Compute cross-entropy and perplexity from a set of logits by hand, and state the unit (nats per token).
- Run a forward and backward pass on a small graph, apply "multiply along paths, add where paths merge", and verify that ∂ℓ/∂z = p − y.
- Take one gradient-descent step and check that the loss decreased.
- Explain the 0 < ηa < 2 stability condition and sketch warmup plus cosine decay, saying whether the schedule counts steps or tokens.
- Write the Adam/AdamW update, explain bias correction and decoupled weight decay, and tally about 16 bytes per parameter of training state.
- Compute an effective batch from replicas, micro-batch size, and accumulation steps, and weight the loss by valid tokens.
- Explain why bf16 needs fp32 master weights, why fp16 needs loss scaling, and where clipping goes in the step.
- Design a validation/test split that matches how the model must generalize, and pick checkpoints by a pre-declared validation rule.

# Sources

- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html) — Inputs are unnormalized logits, ignore_index (default −100) excludes targets, and the mean reduction averages over non-ignored targets.
- [PyTorch: Autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html) — The graph recorded during the forward pass, chain-rule evaluation from roots to leaves, saved intermediates, and gradients accumulated into .grad.
- [Kingma and Ba: Adam, A Method for Stochastic Optimization (2014)](https://arxiv.org/abs/1412.6980) — First and second moment estimates, bias correction, and default β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸.
- [Loshchilov and Hutter: Decoupled Weight Decay Regularization (2017)](https://arxiv.org/abs/1711.05101) — L2 and weight decay are equivalent for SGD but not for Adam, and decoupling them gives AdamW.
- [PyTorch: AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html) — The update order used here (θ ← θ − γλθ, then the Adam step with ε outside the square root) and default hyperparameters.
- [Loshchilov and Hutter: SGDR, Stochastic Gradient Descent with Warm Restarts (2016)](https://arxiv.org/abs/1608.03983) — The cosine annealing formula. This article uses one decay segment without restarts.
- [Smith, Kindermans, Ying and Le: Don't Decay the Learning Rate, Increase the Batch Size (2017)](https://arxiv.org/abs/1711.00489) — The empirical link between increasing batch size and decaying the learning rate, and its effect on the number of updates.
- [Micikevicius et al.: Mixed Precision Training (2017)](https://arxiv.org/abs/1710.03740) — FP32 master weights, loss scaling, and FP32 accumulation for half-precision training.
- [PyTorch: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html) — Keeping the scale constant during accumulation, and unscaling once before clipping.
- [Google Cloud: The bfloat16 numerical format](https://docs.cloud.google.com/tpu/docs/bfloat16) — bfloat16 has the same dynamic range as float32 in half the memory.
- [Pascanu, Mikolov and Bengio: On the difficulty of training Recurrent Neural Networks (2012)](https://arxiv.org/abs/1211.5063) — Gradient norm clipping as a remedy for exploding gradients.
- [Chowdhery et al.: PaLM, Scaling Language Modeling with Pathways (2022)](https://arxiv.org/abs/2204.02311) — Section 5.1: about 20 loss spikes despite clipping, mitigated by restarting about 100 steps earlier and skipping 200–500 batches.

# Flashcards

## loss-unit

**Q:** What does a language-model cross-entropy of 2 nats per token mean, and what's its perplexity?

It's the average of −ln(probability the model gave the correct next token), taken over valid target tokens (padding and masked tokens excluded). Perplexity is exp(2) ≈ 7.39: the model is on average as uncertain as a uniform choice among about 7.4 tokens. Comparisons only make sense with the same tokenizer, masking policy, and evaluation text, because "per token" means something different under a different tokenizer.

## logit-gradient

**Q:** What is the gradient of softmax cross-entropy with respect to the logits?

∂ℓ/∂zᵢ = pᵢ − 1[i = y], meaning "predicted minus true." For logits [2, 1, 0] with the correct token at index 0, p = [0.665, 0.245, 0.090], so the gradient is [−0.335, +0.245, +0.090]. The correct logit gets a negative gradient, so a descent step raises it, and wrong tokens are pushed down in proportion to the probability they took. The entries sum to zero because shifting all logits equally doesn't change softmax.

## backprop-role

**Q:** How does backpropagation differ from gradient descent?

Backpropagation computes the derivative of the loss with respect to every parameter by applying the chain rule backwards through the recorded computation graph: multiply local derivatives along paths, and add where paths merge. Gradient descent (or AdamW, etc.) is the optimizer that *uses* those derivatives to change the parameters. The same backward pass can feed any optimizer.

## chain-rule-by-hand

**Q:** For z = w·x + b, p = σ(z), ℓ = −ln p with x = 2, w = 0.5, b = −0.5, y = 1, what are ∂ℓ/∂w and ∂ℓ/∂b?

Forward: z = 0.5, p = 0.6225, ℓ = 0.474. Backward: ∂ℓ/∂z = (−1/p)·p(1 − p) = p − 1 = −0.3775. Then ∂ℓ/∂b = −0.3775 × 1 and ∂ℓ/∂w = −0.3775 × x = −0.755. One SGD step with η = 0.5 moves w to 0.878 and b to −0.311, and the loss drops to 0.212.

## curvature-bound

**Q:** When does gradient descent converge on L(w) = a·w²/2, and what happens otherwise?

Each step multiplies w by (1 − ηa). It converges when |1 − ηa| < 1, i.e. 0 < ηa < 2. At ηa = 2 it oscillates forever, and above that it diverges even though every step follows the negative gradient. With a = 1 and w₀ = 1, η = 0.5 gives loss 0.0078 after three steps, while η = 2.2 gives 1.49. In a network, the sharpest-curvature direction sets the limit.

## schedule-time

**Q:** Why use warmup, and why must you say whether a schedule counts steps or tokens?

Warmup ramps the learning rate up from near zero because early weights are random, gradients are erratic, and Adam's moment estimates haven't settled. A full-size step at the start can blow up the run. If a schedule counts optimizer steps and you change the batch size, each step covers a different number of tokens, so warmup and decay happen over a different amount of data. State the unit, and re-plan when batch size or run length changes.

## adam-bias

**Q:** Why does Adam apply bias correction to its moment estimates?

m and v are exponential moving averages initialized at zero, so early on they're biased toward zero (after one step with β₁ = 0.9, m is only 10% of the gradient). Dividing by (1 − β₁ᵗ) and (1 − β₂ᵗ) undoes that initialization bias. The correction fades to 1 as t grows.

## first-step

**Q:** With θ = 2, g = 0.5, η = 0.01, λ = 0.1, β₁ = 0.9, β₂ = 0.999, what is the first AdamW update from zero moments?

m = 0.05 and v = 0.00025, so bias-corrected m̂ = 0.5 and v̂ = 0.25. Ignoring ε, the adaptive step is 0.01 × 0.5/0.5 = 0.01, and decoupled decay removes ηλθ = 0.002, giving θ ≈ 1.988. Note that the first Adam step has magnitude about η whatever the gradient's size: a gradient of 0.001 would also move by 0.01.

## adamw-decoupling

**Q:** Why is AdamW not the same as Adam plus an L2 penalty in the loss?

An L2 penalty adds λθ to the gradient, which then passes through Adam's moment estimates and its per-parameter division by √v̂. Parameters with large gradient history get barely regularized. AdamW shrinks the weights directly (θ ← θ − ηλθ), outside the adaptive machinery, so every parameter decays at the same relative rate. For plain SGD the two are equivalent, but for adaptive optimizers they aren't.

## optimizer-memory

**Q:** Roughly how many bytes per parameter does AdamW mixed-precision training need before activations, and what does that mean for a 7B model?

About 16 bytes: bf16 weights (2) + bf16 gradients (2) + fp32 master weights (4) + Adam m (4) + Adam v (4). For 7 × 10⁹ parameters that's about 112 GB, which is more than a single 80 GB GPU before any activations. That's why training state gets sharded across GPUs.

## noise-scaling

**Q:** How does gradient noise change when batch size increases fourfold?

For independent per-example gradients, the variance of the mean falls 4× and the standard deviation 2× (σ/√B). Bigger batches give diminishing returns per token and fewer optimizer steps per token budget. Tokens within a document are correlated, so token count overstates the number of independent samples.

## effective-batch

**Q:** Compute the effective batch for 8 data-parallel replicas, 4 sequences per micro-batch, and 8 accumulation steps.

8 × 4 × 8 = 256 sequences per optimizer update, or 524,288 tokens at 2,048 valid tokens each. Count data-parallel replicas, not total GPUs: if tensor or pipeline parallelism splits each replica across several GPUs, those GPUs still form one replica.

## token-weighting

**Q:** Two micro-batches have 100 and 300 valid tokens with mean losses 2 and 4. What's the correct combined loss?

(100 × 2 + 300 × 4) / 400 = 3.5. Averaging the two means gives 3, which over-weights the short micro-batch. Sum per-token losses and divide by the total valid tokens across the whole effective batch, and the gradients come out correctly weighted as well.

## loss-scaling

**Q:** Why does fp16 training need loss scaling while bf16 typically doesn't, and why do both keep fp32 master weights?

fp16 has only 5 exponent bits, so its smallest positive value is about 6 × 10⁻⁸ and small gradients underflow to zero. Multiplying the loss by e.g. 65,536 lifts them into range, and they're divided back in fp32. bf16 has fp32's 8 exponent bits (same range), so underflow is rarely an issue. But bf16's 7 mantissa bits space numbers 0.0078 apart near 1.0, so a 0.001 update to a weight of 1.0 rounds away. The fp32 master copy is what accumulates small updates.

## clip-order

**Q:** Why clip the gradient after accumulation and after unscaling?

Global-norm clipping rescales the whole gradient when ‖g‖ exceeds a threshold. That's nonlinear, so clipping each micro-batch separately changes the direction of their sum. Loss scaling inflates gradient magnitudes, so clipping before unscaling compares the threshold against the wrong numbers. The order is: accumulate all micro-batches, unscale once, clip, then step.

## test-boundary

**Q:** What happens when you repeatedly select models using the test set?

The test set becomes a selection signal, effectively another validation set, and its score becomes optimistically biased. Use validation for checkpoint and hyperparameter choices, and touch a genuinely untouched test set once at the end. Also split by the unit that must generalize (customer, document, time) and check for duplicates or contamination across splits.
