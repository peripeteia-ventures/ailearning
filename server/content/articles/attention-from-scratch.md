---
{
  "slug": "attention-from-scratch",
  "title": "Attention from scratch: how tokens talk to each other",
  "category": "foundations",
  "summary": "Build self-attention by hand: queries, keys and values, dot-product scores, the √d_k scaling, the causal mask, softmax, and multi-head mixing, with a full 3-token numeric example and the O(T²) cost that shapes every serving decision.",
  "difficulty": "Core",
  "minutes": 30,
  "prerequisites": ["tokens-and-embeddings"],
  "learningObjectives": [
    "Explain why a token's embedding alone is not enough, and how attention mixes information between positions",
    "Describe the different jobs of queries, keys and values and write the scaled dot-product attention formula from memory",
    "Calculate a masked causal attention output by hand for a tiny sequence, including scaling and a numerically stable softmax",
    "Track tensor shapes through multi-head attention and the output projection",
    "Estimate the O(T²) compute and memory cost of attention and explain why K and V are cached at inference"
  ]
}
---

# Sections

## Why a word needs its neighbours {#why-context}

In **Text in, next token out: tokens, embeddings, and sampling** we treated the middle of the model as a black box. Text became token IDs, each ID looked up a row in the embedding table, and at the far end a vector turned into a probability for every possible next token. This article opens the first half of that box.

Here's the problem the box has to solve. After the embedding lookup, every token is a vector (a list of numbers, `d` of them, where `d` is the model's hidden width). But that vector is **context-free**: the token "bank" gets exactly the same row whether the sentence is about rivers or about money. The embedding table is a dictionary, and a dictionary doesn't know what sentence you're reading.

```text
 "I sat on the river bank"          "I paid cash at the bank"
                      │                                  │
                      ▼                                  ▼
           embedding["bank"]  ==  embedding["bank"]     (identical rows!)
```

To predict what comes next, "bank" has to find out who its neighbours are. It needs to pull in some of "river" in the first sentence and some of "cash" in the second. **Attention** is the operation that does this. For every position, it builds a new vector that is a weighted blend of information from other positions, and the weights are computed on the fly from the content itself.

A good picture is a meeting where everyone takes notes. Each person (token) looks around the table, decides whose contributions are relevant to them, and writes a summary that is mostly those people's points plus a bit of everyone else's. Everyone does this at once, and everyone ends up with a different summary because everyone was listening for different things.

> 🎬 **Animation — context mixing:** show two sentences side by side, "I sat on the river bank" and "I paid cash at the bank". Step 1: each word is a coloured box, and both "bank" boxes are the same grey colour with a label "same embedding row". Step 2: arrows fan out from each "bank" to the earlier words, with arrow thickness showing weight; "river" gets a thick arrow in sentence one and "cash" a thick arrow in sentence two. Step 3: each "bank" box recolours as a blend of the colours it pulled from (greenish for river, goldish for cash), labelled "after attention: different vectors".

Two facts to hold onto before we build it. First, attention produces **one output vector per input position**, with the same width, so a `T × d` matrix in gives a `T × d` matrix out (T is the number of tokens, and rows are token positions while columns are features). Second, attention is the **only** place in a transformer where positions exchange information. Everything else in the block works on each position independently, which we'll see in **The transformer block: assembling the full model**.

## Queries, keys, and values: a soft search {#qkv-intuition}

How does a token decide whom to listen to? The trick is to give each token three different roles, each with its own vector.

Think of a library search. You walk in with a **query** ("books about rivers"). Every book has a **key**: its catalogue card, which advertises what it's about. And every book has a **value**: the actual content you take home. You compare your query to each catalogue card, and the better the match, the more of that book's content you use.

| Role | Question it answers | Library analogy |
|---|---|---|
| Query `q` | "What am I looking for?" | The search you type in |
| Key `k` | "What do I offer, for matching purposes?" | The catalogue card |
| Value `v` | "If you pick me, what do I hand over?" | The book's contents |

Attention differs from a real library in one crucial way: the lookup is **soft**. A hash map returns exactly one entry or nothing. Attention returns a blend of *every* allowed entry, weighted by how well each key matches. The best match might get 70% of the weight, the next 20%, and so on. That softness is what makes it trainable: small changes in the vectors cause small changes in the output, so gradients (the signal that tells each weight which way to move during training, covered in **How a model learns: loss, gradients, and optimizers**) can flow through it.

**Why three vectors and not one?** This is where people get tripped up. You could imagine each token just comparing its embedding to everyone else's and blending embeddings. But the thing that makes a token *findable* is often different from the thing it should *deliver*. A verb looking for its subject wants to match "I am a noun, probably a subject", but once found, it wants the subject's meaning ("cat", singular, animate). Separating keys (for routing) from values (the payload) lets the model learn those independently. Separating queries from keys lets the relationship be asymmetric: "sat" looking for "cat" doesn't have to score the same as "cat" looking for "sat".

Where do the three vectors come from? Each is a **learned linear projection** of the token's current vector. Stack the `T` token vectors as rows of a matrix `X` (shape `T × d`) and multiply by three weight matrices:

```formula
Q = X · W_Q      K = X · W_K      V = X · W_V
```

- `X` is the input, `T × d`: one row per token position, one column per feature.
- `W_Q`, `W_K` are `d × d_k` and `W_V` is `d × d_v`: learned weights, the same for every position.
- `Q`, `K` are `T × d_k`, and `V` is `T × d_v`. Row `i` of `Q` is token `i`'s query, and so on.
- `d_k` is the query/key width and `d_v` the value width. In practice both equal the per-head width `d_head`, which we'll meet in the multi-head section.

A note on letters: elsewhere in this course `V` means vocabulary size. In this article `V` is always the value matrix, because that's what every paper and codebase calls it, and the vocabulary doesn't appear here.

```text
            X  (T × d)                W_Q (d × d_k)          Q  (T × d_k)
   pos 0 [ x x x x ]                [ w w ]              [ q q ]  query of token 0
   pos 1 [ x x x x ]      ·         [ w w ]      =       [ q q ]  query of token 1
   pos 2 [ x x x x ]                [ w w ]              [ q q ]  query of token 2
                                    [ w w ]
   (same X, different matrices → K and V)
```

> 🎬 **Animation — Q, K and V projections:** show a 3-row × 4-column matrix X with rows labelled "the", "cat", "sat" and columns "f1–f4", values [1,0,1,0], [0,2,0,1], [1,1,0,1]. Step 1: slide W_Q (4×4) in from the right and highlight row "sat" times each column of W_Q, producing Q's "sat" row [1,2,1,2]. Step 2: repeat for all rows to fill Q. Step 3: the same X is copied twice more and multiplied by W_K and W_V to produce K and V, with the three outputs colour-coded (Q blue, K orange, V green). Caption: "one input, three learned views".

The weights are learned during training. Nobody tells the model "make keys encode grammatical role". It discovers whatever query/key/value features make next-token prediction work.

## Scoring matches with dot products {#dot-product-scores}

We need a number that says "how well does query `i` match key `j`?" Attention uses the **dot product**: multiply the vectors element by element and add up.

```formula
score(i, j) = q_i · k_j = q_i[0]·k_j[0] + q_i[1]·k_j[1] + … + q_i[d_k−1]·k_j[d_k−1]
```

Intuition: the dot product is large and positive when the two vectors point the same way, near zero when they're unrelated (perpendicular), and negative when they point opposite ways. Each feature acts like a question. If both the query and the key are high on feature 3 ("is a noun?"), that feature adds a lot to the score.

Tiny example with 2-dimensional vectors:

| Query | Key | Dot product | Reading |
|---|---|---|---|
| [1, 2] | [2, 1] | 1·2 + 2·1 = 4 | fairly aligned |
| [1, 2] | [2, −1] | 1·2 + 2·(−1) = 0 | unrelated |
| [1, 2] | [−1, −2] | −1 − 4 = −5 | opposed |

We want every query scored against every key. That's exactly a matrix multiply: `Q` is `T × d_k`, `Kᵀ` (K transposed, so rows become columns) is `d_k × T`, and the product is a `T × T` **score matrix**. Row `i` holds query `i`'s scores against all keys, and column `j` corresponds to key `j`.

```text
                 keys →   k_0    k_1    k_2
             ┌──────────────────────────────┐
  queries  q_0│  q0·k0   q0·k1   q0·k2      │
     ↓     q_1│  q1·k0   q1·k1   q1·k2      │   S = Q · Kᵀ   (T × T)
           q_2│  q2·k0   q2·k1   q2·k2      │
             └──────────────────────────────┘
```

This `T × T` grid is the heart of attention and the source of its famous cost: double the sequence and you quadruple the grid. Hold that thought for the last section.

Why a dot product rather than something fancier? The original transformer paper compared it with "additive" attention (a small neural network scoring each pair) and chose dot products because they're one big matrix multiply, which GPUs are extremely good at. It's a pragmatic choice as much as a principled one.

## Why we divide by √d_k, then softmax {#scale-and-softmax}

Raw scores can be any real number. We need **weights**: non-negative numbers that sum to 1 across each row, so the output is a proper blend. That's what **softmax** does. For one row of scores `s_0 … s_{T−1}`:

```formula
weight_j = exp(s_j) / Σ_m exp(s_m)
```

- `exp` makes everything positive and exaggerates differences: a score 1 higher gets e ≈ 2.718 times more weight.
- Dividing by the sum makes the row add up to 1.

Real implementations subtract the row's maximum before exponentiating. It doesn't change the answer (the common factor cancels in the ratio), but it stops `exp` from overflowing. `exp(1000)` is infinity in floating point, and `exp(0)` is a comfortable 1. We'll do the subtraction in the worked example.

**Now the scaling.** Before softmax, scores are divided by `√d_k`:

```formula
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

Here's the argument from the original paper. Suppose each component of a query and a key is independent, with mean 0 and variance 1. A dot product sums `d_k` such products, and each product has variance 1, so the sum has **variance `d_k`**, meaning a standard deviation of `√d_k`. With `d_k = 128` (a common head width), raw scores typically spread around ±11.3. Two scores that are two standard deviations apart differ by about 22.6, and `exp(22.6)` ≈ 6.5 billion. Softmax would put essentially all the weight on one key.

That's called **saturation**, and it's bad for two reasons. The blend collapses into a hard pick, and the gradients through a saturated softmax are nearly zero, so the query and key weights barely learn. Dividing by `√d_k` brings the variance back to about 1 regardless of head width, so softmax starts in its responsive range.

You can see the effect in the example we're about to build (`d_k = 4`, so we divide by 2). For the token "sat", the three raw scores are [2, 6, 4]:

| | weight on "the" | weight on "cat" | weight on "sat" |
|---|---|---|---|
| Unscaled scores [2, 6, 4] | 0.016 | 0.867 | 0.117 |
| Scaled scores [1, 3, 2] | 0.090 | 0.665 | 0.245 |

Same ranking, but the scaled version keeps a real mixture. At `d_k = 128` the unscaled gap would be far more extreme.

Be honest about the caveat in an interview: trained activations are *not* independent unit-variance variables. The derivation explains why `√d_k` is the right *order* of correction. It doesn't prove every model's scores are perfectly normalized. Some architectures add extra normalization on queries and keys for exactly that reason.

> 🎬 **Animation — softmax saturation:** a bar chart of three softmax weights for "sat" over keys "the", "cat", "sat". Step 1: show raw scores [2, 6, 4] and their softmax bars [0.016, 0.867, 0.117]. Step 2: a "÷ √4 = ÷ 2" operation squashes the scores to [1, 3, 2], and the bars smoothly animate to [0.090, 0.665, 0.245]. Step 3: a slider labelled d_k grows from 4 to 128 with unscaled random scores; the bars snap into a single spike near 1.0, with a caption "variance of q·k grows with d_k → softmax saturates".

## No peeking: the causal mask {#causal-mask}

A language model is trained to predict token `t+1` from tokens `0…t`. During training we feed a whole sequence at once and ask for a prediction at *every* position in parallel, which is what makes training efficient. But if position 2 could attend to position 3, it could just read the answer. The model would learn to cheat and would be useless at generation time, when the future doesn't exist yet.

The **causal mask** forbids this: query `i` may only attend to keys `j ≤ i` (itself and earlier). We implement it by adding a mask matrix `M` to the scores *before* softmax:

```formula
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k + M) · V,   M[i, j] = 0 if j ≤ i, −∞ if j > i
```

Why `−∞` and not just zeroing the weights afterwards? Because `exp(−∞) = 0`, so the forbidden keys get exactly zero weight *and* the remaining weights still sum to 1. If you zeroed weights after softmax, the row would sum to less than 1 and you'd have to renormalize. Adding `−∞` first does it in one clean step. (In code, "−∞" is often the most negative finite number of the dtype, to avoid `NaN` from `−∞ − (−∞)` in edge cases.)

```text
          keys:   the    cat    sat
 query "the"   [  s     −∞     −∞  ]     row 0 sees only itself
 query "cat"   [  s      s     −∞  ]     row 1 sees the, cat
 query "sat"   [  s      s      s  ]     row 2 sees everything so far
                  lower triangle kept, upper triangle masked
```

Two consequences worth knowing:

- **The first token has no choice.** Row 0 has one allowed key, so its weight is 1.0 on itself, whatever the scores are.
- **Masking is per-row.** A future key can have a huge raw score and it still gets zero. You'll see a raw score of 6 disappear in the worked example.

There's a second kind of mask you'll meet in practice: the **padding mask**. When sequences of different lengths are batched together, short ones are padded with filler tokens, and the padding mask sets their key columns to `−∞` so real tokens never attend to filler. The causal mask stops time-travel, and the padding mask stops attention to garbage. Bugs in either one are classic: a missing causal mask gives suspiciously low training loss, and a broken padding mask gives outputs that change with batch composition.

> 🎬 **Animation — the causal mask:** a 3×3 grid of scaled scores for rows/columns "the", "cat", "sat", values [[1, 3, 2], [0, 2, 1], [1, 3, 2]]. Step 1: a translucent upper-triangle overlay labelled "future" slides in, turning cells (0,1), (0,2), (1,2) into "−∞". Step 2: each row runs softmax left to right, filling in weights [1, 0, 0], [0.119, 0.881, 0], [0.090, 0.665, 0.245]. Step 3: highlight cell (0,1), whose raw score 3 was the largest in its row, fading to 0 with the caption "big score, still zero: masking wins".

## A complete worked example, by hand {#worked-example}

Let's do the whole thing with real numbers: three tokens "the cat sat", each a 4-feature vector, one attention head with `d_k = d_v = 4`. The weights are hand-picked 0/1 matrices, purely illustrative. Real models learn them and use widths in the hundreds or thousands. Rows are token positions and columns are features throughout.

**Step 0: the input.**

```text
           f1 f2 f3 f4
 X = the [ 1  0  1  0 ]
     cat [ 0  2  0  1 ]
     sat [ 1  1  0  1 ]
```

**Step 1: projections.** The weight matrices (each 4 × 4, rows = input feature, columns = output feature):

```text
 W_Q = [0 1 1 1]    W_K = [0 0 1 0]    W_V = [1 0 0 0]
       [0 0 0 1]          [1 1 0 0]          [0 1 0 1]
       [1 1 0 0]          [0 0 1 0]          [0 0 1 0]
       [1 1 0 0]          [0 0 0 0]          [0 1 0 0]
```

Multiplying a row of X by a 0/1 matrix just adds up the weight rows where X is non-zero, scaled by X's value. For "sat" = [1, 1, 0, 1] we add rows 1, 2 and 4 of W_Q: [0,1,1,1] + [0,0,0,1] + [1,1,0,0] = [1, 2, 1, 2]. For "cat" = [0, 2, 0, 1] we take 2 × row 2 plus row 4 of W_K: [2,2,0,0] + [0,0,0,0] = [2, 2, 0, 0]. Doing all nine rows:

```text
 Q = the [1 2 1 1]     K = the [0 0 2 0]     V = the [1 0 1 0]
     cat [1 1 0 2]         cat [2 2 0 0]         cat [0 3 0 2]
     sat [1 2 1 2]         sat [1 1 1 0]         sat [1 2 0 1]
```

**Step 2: raw scores `S = Q · Kᵀ`.** Check one row by hand. For query "sat" = [1, 2, 1, 2]:

- vs key "the" [0, 0, 2, 0]: 0 + 0 + 2 + 0 = **2**
- vs key "cat" [2, 2, 0, 0]: 2 + 4 + 0 + 0 = **6**
- vs key "sat" [1, 1, 1, 0]: 1 + 2 + 1 + 0 = **4**

```text
            the  cat  sat
 S = the  [  2    6    4 ]
     cat  [  0    4    2 ]
     sat  [  2    6    4 ]
```

**Step 3: scale by √d_k = √4 = 2.**

```text
            the  cat  sat
 S/2 = the [ 1    3    2 ]
       cat [ 0    2    1 ]
       sat [ 1    3    2 ]
```

**Step 4: causal mask.** Upper triangle becomes −∞:

```text
            the  cat  sat
       the [ 1   −∞   −∞ ]
       cat [ 0    2   −∞ ]
       sat [ 1    3    2 ]
```

Notice "the" had its biggest score (3) on "cat", a future token. The mask throws it away.

**Step 5: softmax each row** (subtract the row max, exponentiate, normalize):

| Row | Allowed scores | minus max | exp | sum | weights |
|---|---|---|---|---|---|
| the | [1] | [0] | [1] | 1 | [1.000, 0, 0] |
| cat | [0, 2] | [−2, 0] | [0.1353, 1] | 1.1353 | [0.119, 0.881, 0] |
| sat | [1, 3, 2] | [−2, 0, −1] | [0.1353, 1, 0.3679] | 1.5032 | [0.090, 0.665, 0.245] |

Each row sums to 1 (up to rounding). "sat" puts two-thirds of its attention on "cat", which is a nice story (verb finds its subject), but remember the weights were hand-picked to produce it.

**Step 6: blend the values, `output = weights · V`.** Each output row is a weighted sum of V's rows:

- the: 1.000 × [1, 0, 1, 0] = **[1, 0, 1, 0]** (it can only copy itself)
- cat: 0.119 × [1, 0, 1, 0] + 0.881 × [0, 3, 0, 2] = [0.119, 2.642, 0.119, 1.762]
- sat: 0.090 × [1, 0, 1, 0] + 0.665 × [0, 3, 0, 2] + 0.245 × [1, 2, 0, 1] = [0.335, 2.485, 0.090, 1.575]

Check the second feature of "sat": 0.665 × 3 = 1.995, plus 0.245 × 2 = 0.489, gives 2.485 (computing with unrounded weights gives 2.4852). Every output is still 4 features wide, one row per token. That's the contract: `T × d` in, `T × d` out, and now each row carries context.

> 🎬 **Animation — the full worked example:** a six-panel strip. Panel 1: X (3×4) with the values above. Panel 2: Q, K, V appear in blue, orange and green. Panel 3: the 3×3 grid S fills cell by cell, pausing on row "sat" to show 2, 6, 4 being computed as element-wise products. Panel 4: every cell halves ("÷2"). Panel 5: the upper triangle turns to −∞, then each row becomes weight bars ([1], [0.119, 0.881], [0.090, 0.665, 0.245]). Panel 6: for row "sat", the three V rows slide over scaled by 0.090, 0.665 and 0.245 and stack into the output [0.335, 2.485, 0.090, 1.575].

Senior-level observation: the output is a **blend of learned value vectors**, not a copy of the winning token. Even at 0.665 on "cat", 33% of the output comes from elsewhere. And notice what's missing: nothing here knew that "cat" came before "sat". Shuffle the rows of X (and apply the mask consistently) and the scores just shuffle too. Attention on its own is **order-blind**, which is why position information has to be injected. That's covered in **The transformer block: assembling the full model**.

## Many heads, one output projection {#multi-head}

A single head produces one set of weights per token, so it can follow one kind of relationship at a time. But "sat" might want its subject, its tense cues, and the previous word, all at once. **Multi-head attention** runs `h` smaller attention operations side by side, each with its own `W_Q`, `W_K`, `W_V`, and then combines them.

The analogy: instead of one reviewer reading a document, you have several reviewers, one checking grammar, one checking facts, one checking tone. Each writes a short report, and an editor merges them.

In practice the model's width `d` is split: `h` heads each of width `d_head = d / h`. LLaMA-7B, for example, has `d = 4096` and 32 heads, so `d_head = 128`. Each head computes its own scaled, masked softmax over its own `T × T` grid with `d_k = d_head`. The head outputs (each `T × d_head`) are **concatenated** back to `T × d`, then multiplied by an **output projection** `W_O` (`d × d`):

```formula
head_i = softmax(Q_i · K_iᵀ / √d_head + M) · V_i
MultiHead(X) = Concat(head_1, …, head_h) · W_O
```

- `Q_i, K_i, V_i` are the `i`-th head's slices, each `T × d_head`.
- `Concat` places the head outputs side by side along the feature axis.
- `W_O` is learned. It lets information from different heads be mixed and rewritten into whatever feature layout the rest of the model expects. Without it, head 1's output could only ever land in features 1–128.

Let's split our worked example into **two heads of width 2**: head 1 takes features 1–2 of Q, K and V, and head 2 takes features 3–4. Now `√d_head = √2 ≈ 1.414`. For query "sat":

| | Q slice | scores vs the, cat, sat | ÷ √2 | softmax weights |
|---|---|---|---|---|
| Head 1 | [1, 2] | 0, 6, 3 | 0, 4.243, 2.121 | 0.013, 0.882, 0.106 |
| Head 2 | [1, 2] | 2, 0, 1 | 1.414, 0, 0.707 | 0.576, 0.140, 0.284 |

(Head 1 keys are the = [0,0], cat = [2,2], sat = [1,1]. Head 2 keys are the = [2,0], cat = [0,0], sat = [1,0].) Same token, same input, and completely different attention patterns: head 1 locks onto "cat", while head 2 mostly looks at "the". That's the point of heads. Each learns a different notion of relevance, and `W_O` merges the results.

> 🎬 **Animation — heads split and rejoin:** show the "sat" row of Q as four cells [1, 2, 1, 2], and K and V likewise. Step 1: a vertical cut divides features 1–2 (head 1, purple) from 3–4 (head 2, teal). Step 2: each head runs its own mini attention; show weight bars for "sat": head 1 [0.013, 0.882, 0.106], head 2 [0.576, 0.140, 0.284]. Step 3: the two 2-wide outputs slide together into one 4-wide row (concat). Step 4: that row passes through a W_O box and comes out as a 4-wide vector with mixed colours, labelled "output projection mixes heads".

**Cost check:** splitting into heads doesn't add parameters. `W_Q`, `W_K`, `W_V` and `W_O` are each `d × d` in total (the per-head matrices are just column slices), so attention has `4d²` weights per layer, ignoring biases. For `d = 4096`, that's 4 × 16,777,216 = 67,108,864, about 67M parameters per layer. Many modern models reduce this by sharing key/value heads across several query heads (MQA and GQA), which shrinks `W_K`, `W_V` and, more importantly, the KV cache. We'll go through that in **Making attention cheaper: GQA, FlashAttention, and long context**.

## Shapes, end to end {#shapes}

Interviewers love asking you to trace shapes, because it exposes whether you really know what's happening. Here's one attention layer for a batch of `B` sequences, each `T` tokens long, width `d`, with `h` heads of width `d_head = d / h`.

```text
 X                         B × T × d            batch, positions, features
 │  · W_Q, W_K, W_V        (d × d each)
 ▼
 Q, K, V                   B × T × d
 │  reshape d → h × d_head, move heads forward
 ▼
 Q, K, V                   B × h × T × d_head   each head is independent now
 │  Q · Kᵀ
 ▼
 scores                    B × h × T × T        one T×T grid per head
 │  ÷ √d_head,  + mask,  softmax over last axis (the keys)
 ▼
 weights                   B × h × T × T        each row sums to 1
 │  · V
 ▼
 head outputs              B × h × T × d_head
 │  move heads back, merge h × d_head → d  (concat)
 ▼
 concat                    B × T × d
 │  · W_O
 ▼
 output                    B × T × d            same shape as X
```

Three details to be precise about:

1. **Softmax runs over the key axis** (the last one), separately for every query row, every head and every sequence. Getting the axis wrong is a silent bug: shapes still line up, and the model just trains badly.
2. **The reshape is free-ish.** Splitting `d` into `h × d_head` is a view of the same memory. The transpose that moves heads forward may force a copy, depending on the kernel.
3. **The mask broadcasts.** The causal mask is `T × T` and is shared across `B` and `h`. A padding mask is per sequence, `B × 1 × 1 × T`, and blocks key columns.

For our worked example, `B = 1`, `T = 3`, `d = 4`. With two heads, the scores were `1 × 2 × 3 × 3`, and the output was `1 × 3 × 4`. In PyTorch, `torch.nn.functional.scaled_dot_product_attention` takes these `… × T × d_head` tensors, defaults its scale to `1/√(last dim)`, and applies a lower-triangular mask when you pass `is_causal=True`, which is exactly the recipe above in one fused call.

> 🎬 **Animation — shape pipeline:** a vertical flow of labelled 3D blocks. Start with a slab B×T×d (B=2, T=3, d=4 drawn as cubes). Step 1: it triples into Q, K, V slabs. Step 2: each slab slices along its feature axis into h=2 thinner slabs of width 2 that rotate forward into a B×h×T×d_head stack. Step 3: Q·Kᵀ produces 2×2 square T×T tiles (one per sequence per head) with the upper triangle greyed out. Step 4: tiles multiply V and the thin slabs glue back to B×T×d, then pass through W_O. Every block shows its shape as a label.

## What it costs: the T² problem, and what comes next {#cost-and-preview}

Attention's great strength, letting every token look at every earlier token, is also its big bill. Let's size it for one layer, one sequence, counting a multiply-add as 2 FLOPs (floating-point operations).

| Piece | FLOPs (approx.) | Grows with |
|---|---|---|
| Projections Q, K, V, O | 4 × 2·T·d² = 8·T·d² | T (linear) |
| Scores `Q · Kᵀ` | 2·T²·d | T² |
| Weighted sum `weights · V` | 2·T²·d | T² |

(The per-head costs sum to `d` across heads, so head count doesn't change these totals. With the causal mask about half of the grid is wasted, and good kernels skip it, but the order stays T².)

The quadratic part (`4·T²·d`) overtakes the projections (`8·T·d²`) when `T > 2d`. For `d = 4096` that's around 8,192 tokens. Below that, attention layers are dominated by ordinary matrix multiplies. Above it, the `T²` term takes over. (The block's MLP adds more linear-in-T work, so the real crossover for the whole block comes later.)

**Memory** is the sharper problem. The score grid is `h × T × T` per sequence per layer. At `T = 8,192` with 32 heads in bf16 (2 bytes per number): 8,192² × 32 × 2 = 4,294,967,296 bytes, or 4 GiB, for *one layer of one sequence*, if you actually write it to GPU memory. Naive implementations do. **FlashAttention** avoids this by computing attention in tiles that fit in the GPU's fast on-chip memory, never storing the full grid, while producing exactly the same result. It changes the memory traffic, not the math. Details are in **Making attention cheaper: GQA, FlashAttention, and long context**.

```text
  T          score grid per head (T²)     all 32 heads, bf16
  1,024      1,048,576 entries             64 MiB
  8,192      67,108,864 entries            4 GiB
  32,768     1,073,741,824 entries         64 GiB   ← must never be materialized
```

**A preview of inference.** At generation time the model produces one token at a time. The new token needs its own query, but it attends to the keys and values of *all* earlier tokens. Those past keys and values never change, because the causal mask means earlier tokens never look at later ones. So instead of recomputing them every step, servers store them in a **KV cache**, per layer and per head. That turns each new step's attention into "one query against T cached keys": linear in T per step instead of recomputing everything, at the price of memory that grows with every token. That tradeoff runs the economics of LLM serving, and we'll build it up properly in **What happens at inference: prefill, decode, and the KV cache**.

> 🎬 **Animation — quadratic growth and the KV cache:** Left half: a T×T lower-triangular grid that doubles T from 4 to 8 to 16, with the area counter showing roughly 4× each time and a caption "double T → 4× work". Right half: generation mode, where a column of cached K and V rows (orange and green) grows by one row per step while a single new blue query row compares against the whole column; the caption shows "compute per step ∝ T, cache memory ∝ T".

So, to sum up what attention is: a learned, soft, content-based lookup where every position asks a question (query), every position advertises itself (key) and offers content (value), scores are scaled dot products, the future is masked, softmax turns scores into blend weights, and several heads do this in parallel before an output projection merges them. The rest of the transformer block, covering position, residuals, normalization and the MLP, is the subject of **The transformer block: assembling the full model**.

# Interview

## Question

Walk me through causal self-attention in a decoder-only language model, one layer, from the input hidden states to the output. Explain why queries, keys and values are separate, why the scores are divided by √d_k, where the causal mask goes, and how the cost scales with sequence length.

## Answer

The input is a `T × d` matrix of hidden states, one row per token position. Three learned matrices project it into queries, keys and values. The query says what a position is looking for, the key is what it advertises for matching, and the value is the content it contributes if selected. Keeping keys and values separate lets the model route on one set of features and deliver another. Keeping queries separate from keys lets relevance be asymmetric.

For each head, I compute scores `Q · Kᵀ`, a `T × T` grid. I divide by `√d_head`, because if query and key components were roughly independent with unit variance, the dot product's variance would grow like `d_head`. Without the division, softmax saturates into a near one-hot distribution with vanishing gradients. It's a heuristic justification, not a guarantee about trained activations.

Next I add the causal mask: `−∞` wherever key index > query index. It goes *before* softmax, so masked entries get exactly zero weight and each row still sums to one. That lets training compute predictions for all positions in parallel without any position seeing its own target. Softmax over the key axis gives weights, and multiplying by V gives each position a weighted blend of value vectors.

With `h` heads of width `d/h`, each head does this independently. The outputs are concatenated back to width `d` and multiplied by `W_O`, which mixes information across heads. The parameter count is about `4d²` per layer, regardless of head count.

On cost: projections are linear in T, about `8·T·d²` FLOPs, while the score and mixing matmuls are about `4·T²·d`, so attention compute becomes dominant roughly when T exceeds 2d. Naively, the score grid is `h·T²` memory per layer, which is why fused kernels like FlashAttention tile the computation instead of materializing it. At inference, past keys and values don't change under causal masking, so they're cached. Each new token then costs attention linear in the prefix, at the price of KV-cache memory that grows linearly with context. That memory is usually what limits concurrency.

## Follow-ups

- What goes wrong if you apply the mask after softmax instead of before?
- Why doesn't splitting into more heads change the parameter count or the FLOPs, and what does it change?
- Attention is permutation-equivariant. What does that mean, and how do models recover word order?
- How do MQA and GQA change the KV cache size, and what quality tradeoff do they make?
- If FlashAttention computes the same math, where does its speedup come from?

# Pitfalls

- **"Attention picks the most relevant token."** It doesn't pick. It blends every allowed value vector by softmax weight. Even a dominant key usually leaves a meaningful share to others, and a head that truly one-hots is a special learned case, not the definition.
- **Masking after softmax.** Zeroing future weights after softmax leaves rows that no longer sum to 1, and forgetting to renormalize distorts every output. The mask belongs inside the softmax as `−∞`.
- **Softmax over the wrong axis.** Normalizing over queries (columns) instead of keys (rows) still produces valid shapes and trains, just badly. Always say "softmax over the key axis, per query".
- **Claiming √d_k scaling is proven exact.** The variance argument assumes independent unit-variance components. It justifies the order of the correction, not a guarantee that trained scores are well scaled.
- **Thinking more heads means more compute or parameters.** With `d_head = d / h`, total projection parameters and score FLOPs are unchanged. Heads trade one wide similarity for several narrow ones.
- **Believing the KV cache makes decoding constant-time.** It removes recomputation of past keys and values, but each new query still reads and attends over the whole cached prefix, and the cache itself grows with every token.

# Checklist

- Explain in plain words why embeddings alone can't resolve "bank", and what attention adds.
- State the roles of Q, K and V and why they are separate projections.
- Write `softmax(Q · Kᵀ / √d_k + M) · V` from memory and define every symbol.
- Reproduce the variance argument for √d_k and state its limits.
- Compute a 3-token causal attention output by hand, including max-subtraction in softmax.
- Trace shapes from `B × T × d` through `B × h × T × T` and back, including concat and `W_O`.
- Estimate attention FLOPs and naive score memory for a given T, d and h, and say where the T² term takes over.
- Explain why keys and values can be cached at inference and what that costs.

# Sources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/html/1706.03762v7) — Scaled dot-product attention formula, the variance argument for 1/√d_k, −∞ masking of illegal connections in the decoder, multi-head attention with W^O, additive vs dot-product attention, and O(n²·d) per-layer self-attention complexity.
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/html/2302.13971v1) — Table 2 hyperparameters for the 7B model: dimension 4096, 32 heads, 32 layers (hence d_head = 128).
- [PyTorch: torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — Default scale 1/√(last dim), is_causal lower-triangular masking, and −inf handling for boolean masks.
- [Hugging Face Transformers: How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation) — Why past keys and values can be cached under causal masking, per-layer caches, and the linear per-step attention cost with caching.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — Exact attention computed with tiling to reduce reads and writes to GPU high-bandwidth memory.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) — Multi-query attention uses a single key/value head; grouped-query attention uses an intermediate number of key/value heads.

# Flashcards

## context-mixing

**Q:** Why can't a token's embedding alone tell the model what the token means in this sentence?

The embedding table is a context-free lookup: "bank" gets the same row next to "river" and next to "cash". Attention is the step that lets each position pull in information from other positions and build a context-dependent vector. It's the only place in a transformer block where positions exchange information.

## qkv-roles

**Q:** What are the jobs of queries, keys and values, and why are keys and values separate?

The query is what a position is looking for, the key is what a position advertises for matching, and the value is the content it hands over when matched. Keys only take part in the scores (routing), and values are what get blended (payload). Separate projections let the model match on one set of features and deliver another, for example matching "I'm the subject" but delivering "cat, singular".

## qkv-projections

**Q:** How are Q, K and V computed, and what are their shapes for one head?

Each is a learned linear projection of the same input: `Q = X·W_Q`, `K = X·W_K`, `V = X·W_V`. With X of shape `T × d` (rows are positions, columns are features) and `W` matrices of shape `d × d_head`, each of Q, K and V is `T × d_head`. The weights are shared across positions.

## dot-product-scores

**Q:** What does the score matrix `Q · Kᵀ` contain, and what is its shape?

Entry (i, j) is the dot product of query i with key j: large when they point the same way, near zero when unrelated, negative when opposed. It's `T × T` per head, with one row per query and one column per key. That square is why attention cost grows with T².

## sqrt-dk-scaling

**Q:** Why divide attention scores by √d_k before softmax?

If query and key components are independent with mean 0 and variance 1, their dot product has variance d_k, so raw scores spread wider as heads get wider. Large spreads saturate softmax into a near one-hot pick with tiny gradients. Dividing by √d_k restores variance of about 1. It's a heuristic under idealized assumptions, not a guarantee about trained activations.

## softmax-weights

**Q:** What does softmax do to a row of attention scores, and why is it computed with the row max subtracted?

It exponentiates each score and divides by the row total, giving non-negative weights that sum to 1, with a score 1 higher getting e ≈ 2.72× the weight. Subtracting the max first leaves the result unchanged (the factor cancels) but prevents `exp` overflow in floating point.

## causal-mask

**Q:** Where does the causal mask enter attention, and why −∞?

It's added to the scaled scores before softmax: `M[i, j] = 0` for `j ≤ i` and `−∞` for `j > i`. Since `exp(−∞) = 0`, future keys get exactly zero weight while each row still sums to 1. Applying it after softmax would leave rows that no longer sum to 1.

## mask-purpose

**Q:** Why does a decoder need a causal mask during training at all?

Training predicts the next token at every position of a sequence in parallel. Without the mask, position i could attend to position i+1, which is its own target, and learn to copy the answer. The model would then fail at generation time, when future tokens don't exist.

## attention-output

**Q:** What is the output of attention for one position, and what shape is the full output?

It's a weighted sum of all allowed value vectors, using that position's softmax weights. It's a blend, not a copy of the best-matching token. The full output has one row per position, `T × d_v` per head, and `T × d` after the heads are merged, the same shape as the input.

## worked-row

**Q:** A query's allowed scaled scores are [1, 3, 2] with value rows [1,0,1,0], [0,3,0,2], [1,2,0,1]. What are the weights and the output?

Subtract the max: [−2, 0, −1]. Exponentiate: [0.1353, 1, 0.3679], which sums to 1.5032. The weights are ≈ [0.090, 0.665, 0.245]. The output is 0.090·[1,0,1,0] + 0.665·[0,3,0,2] + 0.245·[1,2,0,1] ≈ [0.335, 2.485, 0.090, 1.575].

## multi-head

**Q:** Why use multiple attention heads, and what does W_O do?

One head gives each token one set of weights, so it can follow one relationship at a time. With h heads of width d/h, each head learns its own Q/K/V projections and attends differently (for example, one to the subject and one to the previous word). The head outputs are concatenated back to width d, and W_O (d × d) mixes them so information from any head can reach any output feature.

## head-cost

**Q:** Does using more heads increase attention's parameters or FLOPs?

Not with `d_head = d / h`. W_Q, W_K, W_V and W_O total `4d²` parameters per layer regardless of h, and the score and mixing FLOPs sum to about `4·T²·d` across heads. More heads trade one wide similarity for several narrower ones. Each also gets its own T × T grid, so naive score memory scales with h.

## quadratic-cost

**Q:** How does attention's cost scale with sequence length T, and when does it dominate?

Projections cost about 8·T·d² FLOPs (linear in T), and the score and mixing matmuls cost about 4·T²·d (quadratic). The quadratic part wins when T > 2d, around 8K tokens for d = 4096. A naive score grid at T = 8,192 with 32 heads in bf16 is 4 GiB per layer per sequence, which is why fused tiled kernels such as FlashAttention avoid materializing it.

## kv-cache-preview

**Q:** Why can keys and values be cached during generation, and what does the cache not save?

Under causal masking, earlier tokens never attend to later ones, so their keys and values never change once computed. The server stores them per layer and head and computes only the new token's Q, K and V each step. The new query still attends over the whole cached prefix, so per-step work and cache memory both grow linearly with context.

## order-blind

**Q:** Does attention by itself know the order of the tokens?

No. The scores depend only on the content of the query and key vectors, so permuting the input rows just permutes the outputs (permutation-equivariance). Order has to be injected separately through positional information, such as learned position embeddings, sinusoids or RoPE.
