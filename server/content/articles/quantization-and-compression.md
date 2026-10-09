---
{
  "slug": "quantization-and-compression",
  "title": "Quantization: spending fewer bits per weight",
  "category": "architecture",
  "summary": "Store and move a model's numbers in fewer bits without breaking what it can do: number formats, scales and zero points, group sizes, GPTQ and AWQ, KV-cache precision, and why a smaller file isn't automatically a faster one.",
  "difficulty": "Advanced",
  "minutes": 30,
  "prerequisites": ["transformer-foundations", "serving-kv-cache"],
  "learningObjectives": [
    "Calculate the memory and bandwidth effect of a quantization scheme, including the hidden cost of scale metadata.",
    "Quantize and dequantize values by hand with a scale and zero point, and explain how granularity and outliers change the error.",
    "Explain PTQ vs QAT, and what GPTQ, AWQ, SmoothQuant-style W8A8 and KV-cache quantization each change.",
    "Defend a quantization choice with a memory budget, a kernel/runtime check and a quality gate, and know when pruning or distillation is the better lever."
  ]
}
---

# Sections

## First figure out what you're actually short of {#whats-scarce}

**Quantization** means storing a model's numbers with fewer bits each: instead of writing every weight as a 16-bit floating-point number, you write it as, say, an 8-bit or 4-bit code plus a little bookkeeping that lets you turn the code back into an approximate real number. Think of saving a photo with a 16-colour palette instead of millions of colours. The file shrinks, and if you pick the palette cleverly, most people can't tell the difference. Pick it badly and the sky turns into stripes.

Before touching any bits, ask the question a senior engineer always asks first: *what's the scarce resource?* A running LLM can be short of three quite different things:

| Scarce resource | What it looks like | Does fewer bits help? |
|---|---|---|
| **Memory capacity** | The model plus its working state doesn't fit on the GPU at all | Yes, directly: fewer bytes to hold |
| **Memory bandwidth** | It fits, but generating each token is slow because the GPU is waiting on memory reads | Often yes, *if* a kernel reads the compact form directly |
| **Compute (FLOPs)** | The math units are saturated, e.g. processing long prompts for many users | Only if the hardware has faster low-precision math *and* you quantize the activations too |

The middle row surprises most newcomers. Here's the napkin version. When a model generates text it produces one token at a time (the **decode** phase), and for each new token it has to multiply the current token's vector by *every* weight matrix in the model. At batch size 1, each weight is fetched from GPU memory, used for one multiply and one add, and then not needed again until the next token. So the GPU spends most of its time waiting for weights to arrive, not doing arithmetic. That's what "memory-bandwidth-bound" means. We go through the prefill/decode split and this arithmetic-intensity argument in detail in **What happens at inference: prefill, decode, and the KV cache**.

A worked example with illustrative numbers. Take an 8-billion-parameter model and a GPU that can read memory at 1 TB/s (a round number, not a specific product):

```text
            bytes of weights       time to read them once     ceiling on tokens/s (batch 1)
bf16 (16b)  8e9 × 2   = 16 GB      16 GB ÷ 1 TB/s = 16 ms     ≈ 62
int8  (8b)  8e9 × 1   =  8 GB       8 ms                      ≈ 125
4-bit       8e9 × 0.5 =  4 GB       4 ms                      ≈ 250
```

These are *ceilings* that ignore the KV cache reads, kernel overheads and everything else, but the shape is real: during small-batch decode, halving the bytes you read roughly halves the time per token. That's why quantization is so popular for local and low-traffic serving.

It also shows what quantizing weights *doesn't* touch. A serving GPU holds three buckets of memory:

```text
┌──────────────────────────── GPU memory ─────────────────────────────┐
│ WEIGHTS           │ KV CACHE                     │ ACTIVATIONS +     │
│ fixed size,       │ grows with tokens × users    │ WORKSPACE         │
│ set by params     │ (keys/values of past tokens) │ temporary buffers │
└─────────────────────────────────────────────────────────────────────┘
  quantize weights ─► only this box shrinks
```

The **KV cache** is where attention keeps the key and value vectors of every token it has already seen, so it doesn't recompute them; it grows with context length and with the number of simultaneous conversations. **Activations** are the intermediate results flowing between layers. Going from 16-bit to 4-bit weights shrinks the first box by about 4×, not the whole GPU footprint. If your problem is that long conversations eat all the memory, weight quantization is the wrong lever, and we'll come back to KV-cache quantization later.

> 🎬 **Animation — three memory buckets:** a GPU memory bar split into Weights (16 GB, blue), KV cache (4 GB, orange), Workspace (2 GB, grey) for an 8B model at bf16. Step 1: weights compress to 4.1 GB with the label "4-bit + scales"; the other two bars don't move. Step 2: a slider raises concurrent users from 1 to 8; the KV bar grows from 4 GB to 32 GB and overflows the 24 GB line while the weights bar stays constant. Step 3: caption "quantizing weights fixes capacity for the model, not for the conversations".

## Number formats: what a bit budget buys you {#number-formats}

Every number format splits its bits between **range** (how big or small a value it can hold) and **precision** (how finely it can distinguish nearby values). Floating-point formats spend bits on a **sign**, an **exponent** (which sets the range, like the "× 10ⁿ" in scientific notation) and a **mantissa** (the significant digits). Integer formats have no exponent: they're evenly spaced codes, and you need an external scale to say what a code means.

| Format | Bits (sign/exp/mantissa) | Largest normal value | Gap after 1.0 | Typical LLM use |
|---|---|---|---|---|
| fp32 | 1 / 8 / 23 | ≈ 3.4 × 10³⁸ | ≈ 1.2 × 10⁻⁷ | optimizer master weights |
| fp16 | 1 / 5 / 10 | 65,504 | ≈ 0.00098 | older inference default |
| bf16 | 1 / 8 / 7 | ≈ 3.4 × 10³⁸ | ≈ 0.0078 | today's usual training/inference dtype |
| fp8 E4M3 | 1 / 4 / 3 | 448 | 0.125 | weights and activations on newer GPUs |
| fp8 E5M2 | 1 / 5 / 2 | 57,344 | 0.25 | gradients (more range, less precision) |
| int8 | 8-bit integer, −128…127 | needs a scale | uniform | weights, sometimes activations |
| int4 | 4-bit integer, −8…7 (or 0…15) | needs a scale | uniform | weight-only quantization |
| NF4 | 16 fixed levels | needs a scale | non-uniform | QLoRA storage |

A few things to notice.

**bf16 vs fp16.** Both are 16 bits. bf16 ("brain float") keeps fp32's 8 exponent bits, so it has the same enormous range but only 7 mantissa bits; fp16 has more precision but overflows above 65,504. Networks mostly care about range (a single overflow produces infinity and poisons everything downstream), which is why bf16 won. When people say "the unquantized model," they usually mean bf16.

**fp8 comes in two flavours.** The FP8 paper by Micikevicius et al. defines E4M3 (max 448) and E5M2 (max 57,344) and recommends E4M3 for weights and activations and E5M2 for gradients. Even so, 448 is a small ceiling, so fp8 tensors still carry a scale factor that stretches the real values into the representable range.

**int4 has only 16 codes.** That's the whole palette. Everything clever in the rest of this article is about choosing *which* 16 real values those codes stand for.

**NF4** ("4-bit NormalFloat," from the QLoRA paper) is a 4-bit code whose 16 levels aren't evenly spaced. They're placed where normally distributed weights are densest, near zero, which the paper argues is information-theoretically optimal for normally distributed weights. It's used to *store* the frozen base model during QLoRA fine-tuning, and it's dequantized to bf16 for the actual math. We cover that setup in **Fine-tuning on a budget: LoRA and QLoRA**.

The shorthand you'll hear in interviews is **WxAy**: W4A16 means 4-bit weights and 16-bit activations; W8A8 means both are 8-bit. It says nothing about the KV cache, the accumulator precision, or which layers were left alone, so always ask.

## Scale and zero point: squeezing real numbers into a few integers {#scale-and-zero-point}

Here's the core arithmetic. The most common scheme is **uniform affine quantization**. "Affine" just means multiply-then-shift, like Celsius to Fahrenheit (× 1.8, + 32). You pick two numbers:

- the **scale** `s`: how much real value one step between neighbouring codes is worth;
- the **zero point** `z`: which integer code stands for real 0.0.

```formula
q  = clamp( round(x / s) + z,  q_min,  q_max )
x̂ = s · (q − z)
```

`x` is the original real value, `q` is the stored integer code, `q_min`/`q_max` are the ends of the code range (0 and 15 for unsigned 4-bit), `clamp` pins anything outside the range to the nearest end, and `x̂` ("x-hat") is the reconstructed value you actually compute with. Going to `q` is **quantizing**; coming back to `x̂` is **dequantizing**. You can never get the rounding loss back.

**Worked example.** Suppose a block of weights ranges from −1 to 2, and we use unsigned 4-bit codes 0…15.

```text
s = (2 − (−1)) / 15 = 3/15 = 0.2      (15 steps span the range)
z = round(0 − (−1)/0.2) = 5           (code 5 means real 0.0)

real:  −1.0  −0.8  ...  0.0  0.2  0.4  0.6  0.8  ...  2.0
code:    0     1   ...   5    6    7    8    9   ...   15
```

| x | x / s | round, + z | clamp | x̂ = 0.2·(q − 5) | error |
|---|---|---|---|---|---|
| 0.74 | 3.7 | 4 + 5 = 9 | 9 | 0.8 | 0.06 |
| −0.91 | −4.55 | −5 + 5 = 0 | 0 | −1.0 | 0.09 |
| 2.4 | 12.0 | 12 + 5 = 17 | **15** | 2.0 | **0.4** |

Two error sources show up. **Rounding error** is at most half a step, `s/2 = 0.1`, for anything inside the range. **Clipping error** is unbounded: 2.4 was outside the range, got pinned to 15, and lost 0.4. Choosing the range is therefore a real tradeoff. A wider range avoids clipping but makes every step coarser for all the ordinary values; a narrower range clips a few rare values to buy finer steps for everyone else. Choosing that range by looking at data is called **calibration**. (Also note that tie-breaking, whether 2.5 rounds to 2 or 3, differs between libraries; state it if you're reproducing numbers.)

> 🎬 **Animation — quantizing 0.74:** a horizontal number line from −1 to 2 with 16 tick marks labelled with codes 0–15 (code 5 sits under 0.0). Step 1: a dot appears at 0.74. Step 2: it snaps right to the nearest tick, 0.8 (code 9), with a small red bracket labelled "error 0.06". Step 3: a second dot at 2.4 appears past the right end and slides back to code 15 (2.0) with a large red bracket "clipped, error 0.4". Step 4: a toggle widens the range to −1…3; ticks spread out (s = 0.267), 2.4 now fits, but 0.74's error grows (it lands on 0.8 again by luck, so show −0.91 instead moving to −0.73… −1.0 with a larger bracket).

**Symmetric quantization** is the special case `z = 0`: signed codes centred on zero, with `s = max|x| / 7` for 4-bit (magnitudes up to 7, leaving code −8 unused in the common convention; some implementations use it). It's simpler and faster (no offset to subtract inside the kernel), and weights, which are roughly centred on zero, suit it well. Asymmetric (non-zero `z`) earns its keep when a range is lopsided, like our −1…2 example, where symmetric would waste the codes below −1.

Quick symmetric example: weights `[0.62, −0.33, 0.05, −1.4]`, so `s = 1.4/7 = 0.2`:

```text
x       x/s     code   x̂      error
0.62    3.1      3     0.6    0.02
−0.33  −1.65    −2    −0.4    0.07
0.05    0.25     0     0.0    0.05    ← a small weight becomes exactly zero
−1.4   −7.0     −7    −1.4    0.00
```

**Why a small per-weight error can still matter.** For a linear layer `y = W·x`, the output error is `Δy = (Ŵ − W)·x`: each weight's error gets multiplied by the input it meets. A weight error of 0.05 against an input of 0.1 contributes 0.005; the same error against an input of 100 contributes 5. So "which weights are sensitive" depends on the *activations* flowing through, not just on the weights. Hold on to that idea, because GPTQ, AWQ and SmoothQuant are all built on it.

## Who shares a scale: per-tensor, per-channel, per-group {#granularity}

**Granularity** is the question of how many values share one scale (and zero point). The tension is simple: one shared scale is cheap to store, but it's set by the biggest value in the crowd, and LLM weights and activations have **outliers**, rare values far larger than their neighbours.

A tiny example with symmetric int4 (s = max|x| / 7). The weight matrix has rows = output channels, columns = input features:

```text
W = │ 0.12  −0.21 │      row 0: small, well-behaved
    │ 8.00   0.30 │      row 1: contains an outlier (8.0)

PER-TENSOR: s = 8/7 ≈ 1.143 for everything
  0.12/1.143 = 0.105 → 0     −0.21 → −0.18 → 0     0.30 → 0.26 → 0
  ⇒ │ 0.00  0.00 │   row 0 is wiped out entirely
    │ 8.00  0.00 │

PER-ROW (per-channel): row 0 s = 0.21/7 = 0.03,  row 1 s = 8/7
  row 0: 0.12/0.03 = 4 → 0.12 ✓   −0.21/0.03 = −7 → −0.21 ✓
  row 1: 8.0 → 8.0 ✓              0.30 → 0 ✗  (still loses to its own outlier)
```

Per-channel rescued row 0, but row 1's small value still dies because it shares a scale with the 8.0. The next step down is **per-group**: split each row into blocks of, say, 32, 64 or 128 consecutive values, each with its own scale. Smaller groups track local ranges better. A group size is meaningless without an **axis** (groups of 128 along which dimension?), so always state both.

> 🎬 **Animation — who shares a scale:** show the 2×2 matrix above as coloured cells. Step 1 (per-tensor): one scale badge "s = 1.143" over the whole grid; three cells fade to grey and read 0.00. Step 2 (per-row): two badges, "s = 0.03" on row 0 and "s = 1.143" on row 1; row 0's cells turn back to full colour with their exact values; 0.30 stays grey. Step 3 (per-group of 1, reductio): each cell gets its own badge and everything is exact, then a counter shows "metadata: 4 scales × 16 bits = 64 bits to store 16 bits of codes", making the cost visible.

**The metadata isn't free.** Each group stores a 16-bit scale (and maybe a zero point), which adds bits per weight:

```formula
effective bits per weight = b + (metadata bits per group) / G
```

`b` is the code width and `G` is the group size. With 4-bit codes and one fp16 scale per group:

| Group size G | Extra bits per weight | Effective bits | 8B-param model |
|---|---|---|---|
| 128 | 16/128 = 0.125 | 4.125 | 8e9 × 4.125 / 8 ≈ 4.13 GB |
| 64 | 0.25 | 4.25 | 4.25 GB |
| 32 | 0.5 | 4.5 | 4.5 GB |
| 128, scale + fp16 zero point | 32/128 = 0.25 | 4.25 | 4.25 GB |

You can see this in the wild. llama.cpp's quantize documentation lists Llama-3.1-8B at 16.0005 bits/weight (14.96 GiB) for F16, **8.5008** bits/weight (7.95 GiB) for Q8_0, and **4.8944** bits/weight (4.58 GiB) for Q4_K_M. That 8.5 is exactly what you'd get from blocks of 32 eight-bit codes sharing one 16-bit scale (8 + 16/32). And the "4-bit" Q4_K_M is nearly 4.9 bits because it keeps some sensitive tensors at higher precision and carries block metadata. "4-bit model" is a nickname, not a size.

There's also a speed cost to small groups: more scales to load and more multiplications to apply them inside the kernel. And there's another knob, **clipping the range on purpose**: setting the scale to cover the 99.9th percentile instead of the absolute max, so the common values get finer steps. Good calibration picks the range that minimizes a meaningful error (like layer output error), not the one that preserves every extreme.

## When to quantize: after training (PTQ) or during it (QAT) {#ptq-vs-qat}

**Post-training quantization (PTQ)** takes a finished checkpoint and converts it, with no training run. The simplest form, round-to-nearest (RTN), looks only at the weights. Smarter PTQ feeds a few hundred **calibration** samples (unlabelled, representative text) through the model to see what activations actually look like, then uses that to pick ranges or to decide which errors matter. Calibration data should look like production traffic: the right languages, formats, lengths and domains. If you calibrate on English prose and serve Python and Japanese, you've calibrated for the wrong model. And keep it separate from your evaluation set.

**Quantization-aware training (QAT)** puts the quantization into the training loop so the weights learn to live with it. In the forward pass you **fake-quantize**: round each weight to its code and immediately dequantize it, so the network sees exactly the values it'll have in deployment, while the optimizer keeps full-precision copies to update. The snag is that rounding is a staircase function: its derivative is zero almost everywhere, so no gradient flows. The standard hack is the **straight-through estimator (STE)**: in the backward pass, pretend rounding was the identity and let the gradient through unchanged. It's an approximation that works well enough in practice, not a proof that rounding is differentiable. (Gradients and backprop are covered in **How a model learns: loss, gradients, and optimizers**.) The Jacob et al. paper from Google is the classic reference for integer-only inference and this training-with-simulated-quantization approach.

```text
QAT forward:   W (fp32 master) ──► fake-quant: round to code, dequant ──► Ŵ ──► layer ──► loss
QAT backward:  ∂loss/∂Ŵ ──────────── STE: pass straight through ─────────► update W
```

An analogy: PTQ is re-recording a finished album on a cheaper tape deck; QAT is rehearsing with the cheap mic so the band adjusts its playing to it.

| | PTQ | QAT |
|---|---|---|
| Cost | minutes to hours on one GPU | a training run: data, GPUs, a pipeline |
| Needs | checkpoint + small calibration set | training access, data, hyperparameters |
| Quality at 8-bit | usually fine | rarely needed |
| Quality at 4-bit and below | good with GPTQ/AWQ-class methods | can recover more, especially at very low bits |
| When to choose | first, almost always | when PTQ misses your quality bar and it's worth the spend |

The operational rule: try PTQ first, measure, and escalate to QAT (or to keeping more layers at higher precision) only when the numbers say so. Whatever you do, validate the **exported** artifact in the real runtime. A fake-quantized model in PyTorch isn't the thing you ship.

## Smarter rounding: how GPTQ and AWQ decide which errors matter {#gptq-and-awq}

Round-to-nearest treats every weight independently and equally. Both of the big 4-bit PTQ methods improve on that by using calibration activations, but in different ways.

### GPTQ: round one, then let the others compensate

Think of rounding every price on a long receipt to the nearest dollar. If you round each item independently, the errors pile up. Better: round the first item, note the error, and nudge the not-yet-rounded items to cancel it, so the *total* stays right. GPTQ does that for a layer's output.

A toy example. A neuron computes `y = w₁x₁ + w₂x₂`, the quantization grid has steps of 0.1, and the calibration data shows that `x₁` and `x₂` are always equal (perfectly correlated, to keep the arithmetic simple). Take `x₁ = x₂ = 1`:

```text
true:        w₁ = 0.13, w₂ = 0.24            y = 0.37
RTN:         0.13 → 0.1, 0.24 → 0.2          y = 0.30   error 0.07
GPTQ-style:  0.13 → 0.1   (error −0.03)
             push −0.03 onto w₂: 0.24 + 0.03 = 0.27 → 0.3
                                             y = 0.40   error 0.03
```

Real GPTQ does this column by column over the whole weight matrix, and it uses approximate **second-order information** to decide how to spread each error: a matrix built from the calibration inputs (the inverse of `XXᵀ`, roughly) that says which weights' errors correlate and how sensitive the output is to each. It's a layer-by-layer reconstruction, not gradient training. The paper reports quantizing 175B-parameter GPT models to 3–4 bits in about four GPU hours, running such a model on a single GPU, and end-to-end speedups over FP16 of about 3.25× on an A100 and 4.5× on an A6000 with their kernels. Those are reported numbers for their setup, not a promise for yours.

> 🎬 **Animation — GPTQ error compensation:** a row of 6 weight bars, each with a faint grid line at multiples of 0.1. Step 1: bar 1 snaps to its nearest grid line; the gap it jumped is shown as a red sliver. Step 2: the red sliver splits and flows into bars 2–6 as small adjustments, sized by thin arrows labelled "from calibration Hessian". Step 3: bar 2 snaps, its new residual flows right, and so on to the end. Step 4: side-by-side output meters: "round-to-nearest: output error 0.07" vs "GPTQ: 0.03" using the two-weight toy numbers.

### AWQ: protect the weights that meet big activations

Remember `Δy = (Ŵ − W)·x`. If some input channels carry consistently large activations, errors in the weights that multiply them get amplified. AWQ (Activation-aware Weight Quantization) finds those channels from calibration data. The paper reports that protecting just 1% of **salient** weights greatly reduces the error. Rather than keeping them in 16-bit (which makes a messy mixed-precision format), it uses an exact algebraic trick:

```formula
w · x = (w · α) · (x / α)
```

Multiply the salient weight channel by a factor `α > 1` and divide the matching activation by `α` (the division can be folded into the previous layer, so it costs nothing at runtime). The math is unchanged before quantization, but now the weight is bigger relative to the grid, so its *relative* rounding error shrinks. Toy numbers, grid step 0.1, α = 2:

```text
plain:  w = 0.13 → 0.1                         error 0.03
AWQ:    w·2 = 0.26 → 0.3, then ÷2 → 0.15       error 0.02
        (worst-case error halves: 0.05 → 0.025)
```

The catch: if scaling up a channel raises the group's maximum, the group's scale grows and the *other* weights get coarser. So AWQ searches for the scaling factors that minimize the layer's output error rather than cranking α up.

Despite the name, **AWQ is weight-only**. Activations are *used* to decide, but they aren't stored in low bits. This is a favourite interview trap.

| | GPTQ | AWQ |
|---|---|---|
| Uses calibration activations to… | weight a layer-wise reconstruction objective | find salient channels and scale them |
| Mechanism | quantize sequentially, compensate remaining weights | equivalent per-channel scaling, then plain rounding |
| Output | int3/int4 weights + scales | int4 weights + scales |
| Risk | can overfit a narrow calibration set | relies on activation statistics being representative |

Comparing them properly means the same base checkpoint, group size, excluded layers, runtime and eval. A method name on a model card doesn't tell you the recipe.

## Weights, activations, and the KV cache are three separate decisions {#what-to-quantize}

**Weight-only (W4A16, W8A16).** Weights are stored low-bit, dequantized on the fly inside the kernel, and multiplied against 16-bit activations with higher-precision accumulators (the running sums a matrix multiply builds up). This attacks memory capacity and decode bandwidth. It doesn't make the math itself cheaper, so it helps little when you're compute-bound.

**Weights and activations (W8A8, FP8).** To use the GPU's faster int8 or fp8 matrix units, *both* inputs to the multiply must be low-precision. Activations are harder than weights because they change with every input, and LLMs develop **outlier features**: a few feature dimensions with values far larger than the rest. The LLM.int8() paper handles them by splitting them out: the outlier dimensions go through a 16-bit multiply while more than 99.9% of values are multiplied in 8-bit. SmoothQuant takes the AWQ-style route instead: an equivalent per-channel scaling that "migrates" difficulty from activations (divide the big channels down) to weights (multiply them up), so both fit in int8. The paper reports up to 1.56× speedup and 2× memory reduction with negligible accuracy loss. Activation scales can be **static** (fixed from calibration) or **dynamic** (computed per token at runtime: more accurate, with some extra overhead).

**KV cache.** A separate growth term with its own knob. Its size is:

```formula
KV bytes = 2 · L · h_kv · d_head · T · B · bytes_per_element
```

The `2` is for keys and values, `L` is layers, `h_kv` the number of key/value heads, `d_head` the per-head width, `T` tokens cached per sequence, and `B` concurrent sequences. An illustrative 8B-class shape with L = 32, h_kv = 8, d_head = 128:

```text
per token, bf16:  2 × 32 × 8 × 128 × 2 bytes = 131,072 B = 128 KiB
one 32k-token conversation: 32,768 × 128 KiB = 4 GiB
8 such conversations:                            32 GiB   (bigger than the 4-bit weights, by 8×)
same at 8-bit KV:                                16 GiB
```

At long contexts and high concurrency, the cache, not the weights, is what caps how many users fit. Reducing `h_kv` via grouped-query attention is the architectural fix, which we cover in **Making attention cheaper: GQA, FlashAttention, and long context**. Quantizing the cache is the numerical one. KIVI is a good concrete example: after studying the distributions, it quantizes **keys per-channel** (keys have outlier channels, like activations) and **values per-token**, going down to 2 bits, and reports 2.6× less peak memory (including weights) and 2.35–3.47× throughput from larger batches. Treat that as one researched design, not a law for every architecture.

KV errors behave differently from weight errors: they sit in memory and get re-read at *every* later decoding step, nudging attention scores and the values being mixed. So test long-context retrieval and long generations through the **actual quantized-cache decoding path**. A single full-sequence forward pass that never reads back from the cache doesn't test it at all.

> 🎬 **Animation — where the bits go:** a single transformer layer drawn as boxes: input activations → matmul with W → output, plus a side store labelled "KV cache" growing one row per generated token. Three toggles light up what's low-bit: "W4A16" colours only W; "W8A8" colours W and the activation arrows and shows the matmul box switching to an "int8 tensor core" icon; "KV8" colours the cache rows. Step 4: generate 5 tokens; the KV store grows by one row per token and each new row is read by every subsequent step (arrows fan out from old rows), showing why cache errors persist.

## A file format is not a speedup {#kernels-and-formats}

Once you have quantized weights, something has to run them, and this is where a lot of "it's smaller but slower" stories come from.

**Containers vs algorithms.** GGUF, the format used by llama.cpp and the ggml ecosystem, is a single-file **container**: key-value metadata (architecture, tokenizer, quantization version, alignment) plus tensors, each tagged with its own type. Those types range from F32, F16 and BF16 through block-quantized types like Q8_0, Q4_K and IQ4_XS. A `.gguf` file can be fully 16-bit. The extension tells you nothing about bit width, calibration or quality; the per-tensor types and metadata do. Similarly, llama.cpp's quantize tool takes a high-precision GGUF (e.g. F32 or BF16) as input and writes a quantized one, optionally guided by an **importance matrix** (`--imatrix`) computed from calibration text. Converting to a container and quantizing its contents are separate steps.

**Kernels decide the speed.** A **kernel** is the low-level GPU or CPU routine that performs an operation. Low-bit weights must be unpacked and scaled before (or while) they're multiplied. There are two ways to do it:

```text
FUSED (good):
  HBM ──4-bit tile + scales──► on-chip SRAM/registers ──unpack, scale──► multiply ──► accumulate
        (move 4 bits/weight)       (dequant happens where it's cheap)

UNFUSED (bad):
  HBM ──4-bit──► dequant kernel ──16-bit full matrix──► HBM ──16-bit──► matmul kernel
        read 4       write 16                            read 16 again
  ⇒ you move MORE bytes than the unquantized model did
```

Like unpacking only the ingredients you need at the stove, versus emptying the whole grocery order onto the counter first. The fused path keeps the bandwidth win; the unfused path throws it away and adds kernel launches.

**Compatibility has layers.** The runtime must know the architecture, read the container, understand the exact packing and group layout, and have a fast kernel for that combination on your device. Many GPU kernels require particular group sizes, alignments or GPU generations (fp8 matrix units exist only on newer hardware). Loading successfully can hide fallbacks to a slow path, or even CPU offload. Pin the runtime version and check what actually executes.

**Which phase speeds up?** Decode at small batch is bandwidth-bound, so weight-only quantization with a fused kernel helps a lot. Prefill (processing the whole prompt in one pass) and large-batch decode reuse each loaded weight across many tokens, so they're closer to compute-bound; weight-only quantization helps less there and the dequant work can even cost time. The scheduling side of this is in **Serving many users: batching, scheduling, and speculative decoding**. Benchmark both phases at realistic concurrency.

> 🎬 **Animation — fused vs unfused dequant:** two lanes, each showing HBM (big box, left), SRAM (small box, middle), compute units (right). Top lane (fused): small 4-bit tiles travel HBM → SRAM, expand to 16-bit inside SRAM, go straight to compute; a byte counter reads "4.1 GB moved per token". Bottom lane (unfused): tiles travel HBM → compute → a full 16-bit matrix is written back to HBM, then read again; counter reads "4.1 + 16 + 16 GB". Final frame: token latency bars, fused ≈ 4 ms, unfused ≈ 36 ms, using the 1 TB/s illustrative bandwidth, labelled "illustrative".

## Proving it's good enough {#measuring-quality}

A quantized model that loads isn't a quantized model that works. Treat acceptance as three separate gates, and set the pass bars *before* you start trying candidates, so you aren't tempted to move the goalposts.

**Gate 1: numerical fidelity.** **Perplexity** measures how surprised the model is by real text: the exponential of the average negative log-probability it assigned to each actual next token (lower is better; cross-entropy is explained in **How a model learns: loss, gradients, and optimizers**). Compare the quantized model against its own bf16 reference on identical text, tokenization and context windows. The evaluation method matters: Hugging Face's guide shows GPT-2 large scoring 19.44 with non-overlapping 1024-token chunks but 16.44 with a stride-512 sliding window, because each prediction gets more context. Same model, different number. llama.cpp also reports **KL divergence**, which compares the quantized model's full next-token probability distribution against the reference token by token. That's more sensitive than perplexity alone.

**Gate 2: task behaviour.** A small perplexity bump can hide real regressions: broken JSON, wrong tool arguments, worse arithmetic, forgotten rare languages, degraded long-context retrieval. Run paired comparisons on representative prompts and hard slices with fixed generation settings. Expect outputs to diverge textually (a tiny logit change can flip an early token and send the continuation somewhere else), so grade *semantic* success, not byte equality. Use enough examples to see the uncertainty, and read the failures. How to build these evals properly is in **Evaluation and observability: knowing whether it actually works**.

**Gate 3: serving behaviour.** Measure peak memory, time to first token, time per output token, throughput and tail latency (p95/p99) at the concurrency you'll actually run, after warm-up, on the target hardware and runtime.

```text
reference (bf16) ──► calibrate ──► convert ──► gate 1 ──► gate 2 ──► gate 3 ──► ship
      ▲                                  │ fail       │ fail       │ fail
      └──── raise precision of sensitive layers / change group size / try QAT ◄┘
```

After shipping, keep the previous artifact ready for rollback, and re-check when traffic shifts. New languages or longer contexts can break assumptions the calibration set baked in.

> 🎬 **Animation — three gates:** a candidate model card walks through three turnstiles labelled "Fidelity (PPL, KL)", "Tasks (JSON, tools, long-context)", "Serving (p99, memory)". Candidate A (Q4, group 128) passes gate 1 with "PPL +2%" but gets stopped at gate 2 by a red "tool-call JSON valid: 97% → 88%" (label: illustrative). The loop arrow sends it back; it returns as "Q4 + attention/output layers at 8-bit" and passes all three.

## When pruning or distillation is the better lever {#pruning-distillation}

Quantization changes how each number is *written*. Two other compression families change *how many numbers there are*.

**Pruning** removes weights. **Unstructured** pruning zeros individual weights anywhere. SparseGPT (from the same group as GPTQ, using a similar reconstruction idea) reports pruning 175B-class models to 50–60% sparsity in one shot with little perplexity increase. But here's the catch interviewers love: a dense matrix full of zeros runs at exactly the same speed on dense kernels. You only save time with sparse kernels, and irregular zeros are hard to exploit. **Semi-structured** patterns like **2:4** (in every block of four weights, two are zero) exist because some GPUs have hardware support for exactly that pattern; SparseGPT supports 2:4 and 4:8. **Structured** pruning removes whole heads, channels or layers, which makes the dense matrices genuinely smaller and faster everywhere, at the cost of more capability loss and usually some recovery training.

**Distillation** trains a smaller **student** model to imitate a larger **teacher**. Hinton, Vinyals and Dean's paper introduced training on the teacher's **soft targets**: its full probability distribution over outputs, softened with a temperature, rather than only the single correct answer. The student learns "the teacher thought *cat* was likely, *dog* plausible, *car* absurd," which carries far more information per example. It's like learning from a tutor's worked reasoning instead of just the answer key. A distilled 3B model is smaller in both memory *and* compute on any ordinary dense kernel, but you pay for teacher inference and a training run, and anything the training data didn't cover won't transfer.

| Lever | What shrinks | Needs special kernels? | Cost to produce | Best when |
|---|---|---|---|---|
| Quantization (PTQ) | bytes per weight | yes, for speed | low | memory or decode bandwidth is the bottleneck |
| Unstructured pruning | nonzero count | yes, and often no speedup | low–medium | mostly research, or combined with sparse hardware |
| Structured pruning | matrix dimensions | no | medium + recovery training | you need real compute savings |
| Distillation | the whole model | no | high (data + training) | you need a much smaller model for high volume |

They combine (a distilled model can also be quantized), but errors interact, so evaluate the final combined artifact. Mixture-of-experts is a different trade again, more parameters for the same compute per token, covered in **Mixture of experts: more parameters, same compute per token**.

The senior summary: name the bottleneck, then pick the cheapest lever that clears the bar. Weights barely don't fit? Supported 8-bit or 4-bit PTQ. Long contexts eating memory? KV-cache precision, GQA, admission limits. Still compute-bound at huge volume after that? A smaller distilled or structurally pruned model. Defend the choice with a memory budget and an acceptance test, not a compression ratio.

# Interview

## Question

You need to serve an 8-billion-parameter assistant on a single 12 GB GPU, and users paste long documents for extraction. Someone produced a 4-bit checkpoint: it loads, but tokens come out slower than expected and extraction accuracy dropped. Walk me through how you'd investigate and what you'd do.

## Answer

I'd start by writing down a **memory budget**, because "4-bit" doesn't mean 4 GB. An 8B model at a Q4_K_M-style encoding is about 4.9 bits per weight, roughly 4.6–4.9 GB, since scales and some higher-precision tensors come along. Then the KV cache: with an 8B-class shape (32 layers, 8 KV heads, head dim 128) that's 128 KiB per token in bf16, so a single 32k-token document is 4 GiB. Add runtime workspace and we're close to 12 GB with *one* long request. If the runtime is quietly offloading layers or cache to the CPU to make it fit, that alone explains the slowdown.

Next, **is the fast path actually running?** I'd check that the runtime has a fused kernel for this exact format, group size and GPU, and profile to see whether weights are being dequantized into full 16-bit matrices in memory, or falling back to a slow path. I'd benchmark prefill and decode separately. Long documents make prefill big, and prefill is closer to compute-bound, where weight-only 4-bit doesn't help and dequant overhead can hurt. So "slower" might be real for prefill even when decode got faster.

For **quality**, I'd establish a bf16 reference with the same tokenizer, chat template and generation settings, and confirm the quantized artifact came from the same base checkpoint. Then I'd isolate variables: weights quantized with KV in bf16, then KV quantized too. Extraction over long documents stresses long-context attention, which is exactly what KV-cache quantization can damage, and it has to be tested through the real cached decoding path. I'd also check the calibration data: if it was short English prose, it didn't represent long structured documents.

Fixes, in order of cost: keep sensitive tensors (embeddings, output head, maybe attention projections) at 8-bit; try a smaller group size or an AWQ/GPTQ variant calibrated on in-domain data; keep KV at 8-bit rather than lower; and cap context length or concurrency with an admission policy so the cache fits on-GPU. If none of that clears the pre-agreed extraction accuracy and p95 latency bars, the honest answer is a smaller or distilled model, or a bigger GPU. I'd only ship a candidate that passes all three gates (fidelity, task, serving) and keep the previous artifact for rollback.

## Follow-ups

- At high concurrency the KV cache dominates memory. How does your plan change, and what are the risks of quantizing the cache to 4 or 2 bits?
- Why might a smaller group size improve quality but reduce throughput?
- When would you choose W8A8 (e.g. SmoothQuant or FP8) over W4A16, and what hardware would you need?
- When would you distill a smaller model instead of quantizing this one?

# Pitfalls

- Saying a "4-bit" 8B model takes 4 GB. Scale metadata, higher-precision layers, the KV cache and runtime workspace all come on top. Real 4-bit GGUF files for 8B models are closer to 4.6 GB before any cache.
- Assuming AWQ quantizes activations. It uses activation statistics to pick and protect salient weight channels, but it's a weight-only method.
- Treating GGUF (or any container) as a quantization method, or assuming a `.gguf` file is low-bit and will run fast. Read the tensor types and check the kernel path.
- Expecting weight-only quantization to speed up compute-bound work like long-prompt prefill or big batches. It mostly helps small-batch decode, where memory bandwidth is the limit.
- Validating KV-cache quantization with a full-sequence forward pass that never reads back from the quantized cache.
- Signing off on a small perplexity change without task-level tests: structured output, tool calls, rare domains and long-context retrieval can regress while perplexity barely moves.
- Expecting unstructured pruning to make dense kernels faster. Zeros stored in a dense matrix still get multiplied.

# Checklist

- Name the scarce resource (capacity, bandwidth or compute) before choosing a format.
- Compute weight memory including metadata: P × (b + metadata bits / G) / 8.
- Quantize and dequantize a value by hand with a given scale and zero point, and separate rounding error from clipping error.
- Explain per-tensor, per-channel and per-group scales, and why outliers push you toward finer granularity.
- Explain PTQ vs QAT, fake quantization and the straight-through estimator.
- Describe what GPTQ and AWQ each do with calibration activations, and why AWQ is still weight-only.
- Size the KV cache with 2 · L · h_kv · d_head · T · B · bytes, and say when KV quantization beats weight quantization.
- Check that a fused kernel for your exact format and hardware actually runs, and benchmark prefill and decode separately.
- Gate a release on fidelity, task and serving tests fixed in advance, and keep a rollback artifact.

# Sources

- [Micikevicius et al.: FP8 Formats for Deep Learning (2022)](https://arxiv.org/abs/2209.05433) — E4M3 and E5M2 definitions, max values 448 and 57,344, and the recommendation of E4M3 for weights/activations and E5M2 for gradients.
- [Jacob et al.: Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2017)](https://arxiv.org/abs/1712.05877) — Affine integer quantization and training with simulated quantization.
- [Frantar et al.: GPTQ (2022)](https://arxiv.org/abs/2210.17323) — Second-order, one-shot weight quantization; 175B models to 3–4 bits in about four GPU hours; reported 3.25× (A100) and 4.5× (A6000) speedups.
- [Lin et al.: AWQ (2023)](https://arxiv.org/abs/2306.00978) — Protecting ~1% salient weights via equivalent channel scaling; weight-only low-bit quantization.
- [Dettmers et al.: LLM.int8() (2022)](https://arxiv.org/abs/2208.07339) — Outlier features, vector-wise quantization and 16-bit decomposition of outlier dimensions while >99.9% of values use 8-bit.
- [Xiao et al.: SmoothQuant (2022)](https://arxiv.org/abs/2211.10438) — W8A8 by migrating activation difficulty into weights; reported up to 1.56× speedup and 2× memory reduction.
- [Dettmers et al.: QLoRA (2023)](https://arxiv.org/abs/2305.14314) — NF4 data type, double quantization and paged optimizers.
- [Liu et al.: KIVI (2024)](https://arxiv.org/abs/2402.02750) — 2-bit KV cache with keys per-channel and values per-token; reported 2.6× less peak memory and 2.35–3.47× throughput.
- [GGML: GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — GGUF as a single-file container of metadata and typed tensors, from F32/BF16 through quantized block types.
- [llama.cpp: quantize tool README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) — High-precision GGUF input, quantized output types, `--imatrix`, and the Llama-3.1-8B bits-per-weight and size table.
- [Hugging Face Transformers: Perplexity of fixed-length models](https://huggingface.co/docs/transformers/main/perplexity) — Perplexity definition and the 19.44 vs 16.44 sliding-window example.
- [Frantar and Alistarh: SparseGPT (2023)](https://arxiv.org/abs/2301.00774) — One-shot pruning to 50–60% sparsity and 2:4 / 4:8 semi-structured patterns.
- [Hinton, Vinyals and Dean: Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531) — Knowledge distillation from a large model into a smaller deployable one.

# Flashcards

## bandwidth

**Q:** Why does 4-bit weight quantization speed up single-user token generation even if the math is still done in 16-bit?

At small batch sizes, decode is memory-bandwidth-bound: every generated token requires reading every weight once, and each weight is used for only a multiply and an add. The GPU spends its time waiting on memory. Reading 4-bit weights moves about a quarter of the bytes, so time per token drops, as long as a fused kernel dequantizes tiles on-chip instead of writing a full 16-bit copy back to memory.

## formats

**Q:** Why did bf16 largely replace fp16 for LLMs, and what are fp8's two variants for?

bf16 keeps fp32's 8 exponent bits, so it has the same huge range and doesn't overflow at 65,504 like fp16. It gives up precision (7 mantissa bits) that networks tolerate well. fp8 comes as E4M3 (more precision, max 448, recommended for weights and activations) and E5M2 (more range, max 57,344, recommended for gradients), and both usually carry a scale factor.

## affine

**Q:** What do the scale and zero point mean in affine quantization?

The scale `s` is the real-valued size of one step between neighbouring integer codes; the zero point `z` is the code that represents real 0.0. Quantize with `q = clamp(round(x/s) + z, q_min, q_max)` and dequantize with `x̂ = s·(q − z)`. Symmetric quantization is the special case z = 0.

## example

**Q:** With s = 0.2, z = 5 and unsigned 4-bit codes, what happens to 0.74?

0.74 / 0.2 = 3.7, which rounds to 4; add z to get code 9. Dequantizing gives 0.2 × (9 − 5) = 0.8, an absolute error of 0.06. The rounding error is lost for good.

## clipping

**Q:** Why isn't the s/2 rounding-error bound enough to guarantee a quantized network behaves well?

It only covers one scalar that falls inside the range. Values outside the range get clipped with arbitrarily large error, and in a network each weight error is multiplied by the activation it meets (Δy = (Ŵ − W)·x), then passed through many layers and a discrete token choice. So small errors on weights that meet large activations can matter a lot.

## symmetric

**Q:** When is symmetric quantization a good choice, and when is asymmetric better?

Symmetric (z = 0, s = max|x|/7 for int4) suits roughly zero-centred data like most weights and is cheaper in kernels because there's no offset to subtract. Asymmetric uses a non-zero zero point to spend all codes on a lopsided range (like −1 to 2), at the cost of storing and applying the offset.

## groups

**Q:** Why do smaller quantization groups usually improve quality, and what do they cost?

Each group gets its own scale, so an outlier only coarsens the few values in its group instead of the whole row or tensor. The cost is metadata (a 16-bit scale per 128 weights adds 0.125 bits per weight; per 32 adds 0.5) plus more scaling work and packing constraints in the kernel, which can reduce throughput.

## metadata-bits

**Q:** Why is llama.cpp's Q8_0 8.5 bits per weight, and why is a "4-bit" model bigger than P/2 bytes?

Q8_0's 8.5 bits matches blocks of 32 eight-bit codes sharing one 16-bit scale: 8 + 16/32 = 8.5. The same logic applies to 4-bit formats: scales and zero points add bits, and mixed recipes keep some sensitive tensors at higher precision. That's why an 8B Q4_K_M file is about 4.9 bits/weight and 4.58 GiB, not 4 GB.

## ptq

**Q:** Does post-training quantization need labelled data?

No. Round-to-nearest needs only the weights. Calibrated methods (GPTQ, AWQ, static activation scales) need a few hundred representative, unlabelled inputs to see real activation distributions. The calibration set should match production traffic and stay separate from the evaluation set.

## qat

**Q:** How does quantization-aware training get gradients through a rounding step?

The forward pass fake-quantizes (round to a code, dequantize immediately) so the network sees deployment values, while full-precision master weights are what get updated. Rounding has zero derivative almost everywhere, so the backward pass uses a straight-through estimator: it treats rounding as the identity and passes the gradient through unchanged. It's a useful approximation, not true differentiability.

## gptq

**Q:** What does GPTQ do differently from rounding each weight to the nearest code?

It quantizes a layer's weights sequentially and, after each one, adjusts the not-yet-quantized weights to compensate for its error, using approximate second-order information built from calibration inputs. The goal is to preserve the layer's output rather than each weight individually, like rounding receipt items while nudging the rest so the total stays right.

## awq

**Q:** Does AWQ store activations in low precision?

No. AWQ is weight-only. It uses activation statistics to find the ~1% of salient weight channels (those meeting large activations) and protects them with an exact rescaling, w·x = (w·α)·(x/α), which shrinks their relative rounding error. Activations stay in 16-bit.

## kv

**Q:** Why should KV-cache quantization be evaluated separately from weight quantization?

The KV cache grows with tokens × concurrent users, often exceeding the weights at long context, so it's a different memory lever. Its errors persist: every later decoding step re-reads the cached keys and values, altering attention. It has to be tested on long-context and long-generation tasks through the actual quantized-cache decoding path.

## container

**Q:** What does a .gguf extension tell you about a model's precision?

Almost nothing. GGUF is a container of metadata plus tensors, each tagged with its own type, which can be F32, F16, BF16 or a quantized block type. To know the precision, calibration and runtime compatibility, read the per-tensor types and metadata.

## dequant

**Q:** How can a smaller quantized model run slower than the 16-bit original?

If there's no fused kernel for the format and hardware, the runtime may dequantize whole weight matrices into 16-bit copies in memory and then read them again, moving more bytes than before and adding kernel launches. Fallback paths or CPU offload have the same effect. Weight-only quantization also helps little in compute-bound prefill.

## evaluation

**Q:** Why isn't a small perplexity change enough to approve a quantized model?

Perplexity averages token-prediction quality over text and can hide regressions in structured output, tool-call arguments, reasoning, rare domains and long-context retrieval. You also need paired task tests on representative and hard slices, plus serving measurements (memory, p99 latency) at real concurrency, all with pass bars set in advance.
