---
{
  "slug": "tokens-and-embeddings",
  "title": "Text in, next token out: tokens, embeddings, and sampling",
  "category": "foundations",
  "summary": "The whole LLM pipeline from a bird's-eye view: how text becomes token IDs, IDs become vectors, a black-box stack turns vectors into scores, and a decoding policy turns scores into the next token, one step at a time.",
  "difficulty": "Core",
  "minutes": 30,
  "prerequisites": [],
  "learningObjectives": [
    "Explain the next-token-prediction contract and why generation is an autoregressive, one-token-at-a-time loop",
    "Run byte-pair encoding by hand on a toy corpus and explain why subword tokens beat characters or whole words",
    "Estimate token counts, cost, and context budget, and explain why languages, code, and whitespace change them",
    "Calculate the size of an embedding table and explain what tying input and output embeddings saves",
    "Compute softmax probabilities from logits, and show how temperature, top-k, and top-p change the choice of next token",
    "Defend a decoding configuration for a task, including stop conditions and what \"deterministic\" can and can't promise"
  ]
}
---

# Sections

## The only thing the model promises: a guess at the next token {#next-token-contract}

Let's start with the job description, because everything else in this article hangs off it. A large language model (LLM) does exactly one thing: you give it some text, and it gives you back a probability for **every possible next chunk of text**. That's it. Think of the autocomplete on your phone, except it has read a large slice of the internet and it scores tens of thousands of candidates at once instead of suggesting three.

Those "chunks" are called **tokens**. A token is a small piece of text: often a whole common word like ` the`, sometimes a fragment like `ing`, sometimes a single character or byte. Before the model sees anything, each token is replaced by an integer called its **token ID**. The set of all tokens the model knows is its **vocabulary**, and we write its size as `V` (real vocabularies run from tens of thousands to a few hundred thousand entries).

Here's a tiny example. Suppose the text so far is `The capital of France is`. The model might say:

| Candidate next token | Probability |
|---|---|
| ` Paris` | 0.59 |
| ` the` | 0.22 |
| ` a` | 0.13 |
| ` Lyon` | 0.05 |
| ` banana` | 0.01 |

(These are illustrative numbers; we'll derive them properly in the softmax section.) Notice the leading spaces: in most tokenizers the space belongs to the *following* word, so ` Paris` and `Paris` are different tokens with different IDs.

To produce more than one token, you run a loop. Pick one token from that distribution, glue it onto the end of the text, and ask again. This is called **autoregressive generation**: "auto" because the model's own output becomes its next input.

```text
 prompt:  The cat sat
          │
          ▼
 ┌──────────────┐   pick   ┌─────────────────────────┐
 │    model     │ ───────► │ " on"                   │
 └──────────────┘          └─────────────────────────┘
          ▲                            │ append
          └──── The cat sat on ◄───────┘
 step 2 → " the"   step 3 → " mat"   step 4 → <end-of-text>  → stop
```

> 🎬 **Animation — the autoregressive loop:** start with four boxes "The", " cat", " sat" feeding into a grey "model" box. Step 1: a bar chart pops out of the model with five bars (" on" 0.41, " down" 0.22, " there" 0.15, " quietly" 0.12, "." 0.10; illustrative). Step 2: the " on" bar is highlighted, lifts off, and slides to the end of the input row. Step 3: the model runs again on the four-token row and a new bar chart appears with " the" highest. Repeat for " mat", then a special `<end-of-text>` token wins and a red "STOP" stamp appears. Keep a counter "model calls: 1, 2, 3, 4" in the corner to show one full model call per generated token.

Written as a formula, the model defines the probability of a whole sequence as a chain of next-token guesses multiplied together:

```formula
P(x₁, x₂, …, x_T) = P(x₁) · P(x₂ | x₁) · P(x₃ | x₁, x₂) · … · P(x_T | x₁, …, x_{T−1})
```

Here `x₁ … x_T` are the tokens, `T` is the sequence length in tokens, and `P(a | b)` means "the probability of `a` given that `b` came before". This is just the chain rule of probability; the model's job is to supply each factor. Worked example: if the model gives ` The` 0.05 at the start, ` cat` 0.01 after ` The`, and ` sat` 0.2 after ` The cat`, the whole three-token sequence gets 0.05 × 0.01 × 0.2 = 0.0001. Because those products get tiny fast, practitioners add **log-probabilities** instead: ln 0.05 + ln 0.01 + ln 0.2 ≈ −3.00 − 4.61 − 1.61 = −9.21, and e^−9.21 ≈ 0.0001. Same number, no underflow.

**Why this framing matters in an interview.** A chat app looks like it's sending the model a structured conversation object. It isn't. The app *serializes* everything (the system prompt, each user and assistant turn, tool results) into one flat string using a model-specific **chat template**, which inserts special marker tokens for "a user turn starts here" and so on. The model sees one long list of token IDs and predicts what comes next. So "memory", "permissions" and "which tool is allowed" are application features built around the model, not properties of the network. Candidates who blur that line get pushed on it. We'll go deeper on templates in **From base model to assistant: SFT, RLHF, DPO, and verifiable rewards**.

**Why generation is sequential.** Token 5 can't be computed until token 4 has been chosen, because token 4 is part of token 5's input. So producing 500 tokens means 500 trips through the model, one after the other. Training doesn't have this problem, since the whole correct text is already known and all positions can be scored in parallel; we'll see why in **The transformer block: assembling the full model**. That asymmetry is the root of most inference engineering, which lives in **What happens at inference: prefill, decode, and the KV cache**.

Here's the full pipeline we'll walk through in this article. The middle box stays closed for now.

```text
 "The cat sat"                                     text
      │ tokenizer
      ▼
 [791, 8415, 7731]                                 T token IDs        (illustrative IDs)
      │ embedding lookup (table: V × d)
      ▼
 T × d matrix   rows = token positions, columns = features
      │ transformer stack (L blocks)  ◄── black box for now
      ▼
 T × d matrix   (same shape, "contextualized" vectors)
      │ unembedding (d → V)  on the last row
      ▼
 V logits  ──softmax──►  V probabilities  ──decoding policy──►  next token ID
```

## Why not just use characters, or whole words? {#why-subwords}

Before a model can do any maths, text has to become numbers. There are three obvious ways to chop it up, and it's worth seeing why the industry settled on the middle one.

**Option 1: whole words.** Give every word an ID. It's simple and each unit carries a lot of meaning. The problem is the long tail. English alone has hundreds of thousands of word forms, plus names, typos, URLs, `camelCaseIdentifiers`, and every other language. Any word not in the list becomes a single "unknown" token, and the model is blind to it. Imagine a dictionary that prints `???` every time you misspell something.

**Option 2: characters (or bytes).** Now nothing is ever unknown, since everything is made of letters. But sequences get long. `unbelievable` is 12 characters, and every one costs a model step when generating and a position of context. Worse, the model has to spend capacity learning that `u-n-b-e-l…` is a word at all, before it can learn anything about what it means.

**Option 3: subwords.** Keep frequent strings whole (` the`, ` France`) and break rare ones into reusable pieces (`un` + `believ` + `able`). You get short sequences for common text and a fallback path for anything weird. This is what virtually every modern LLM uses.

| Unit | Sequence length | Unknown inputs? | Vocabulary size | Typical use |
|---|---|---|---|---|
| Words | Shortest | Yes, anything unseen | Huge, and still incomplete | Old NLP pipelines |
| Characters / bytes | Longest (≈ 4× subwords on English) | Never | Tiny (256 for bytes) | Research models |
| Subwords | Short for common text | Never, with a byte fallback | 30k–250k-ish | Essentially all LLMs |

The analogy I like: subword tokens are like a well-stocked Lego set. You have big pre-built pieces for the shapes you use all the time, and small 1×1 bricks so you can still build anything odd.

There's a deeper tradeoff hiding in the vocabulary size `V`. A bigger vocabulary means each token covers more text, so sequences are shorter (cheaper attention, more text per context window). But every extra token adds a row to two `V × d` tables (we'll meet them shortly) and makes the final scoring step more expensive, and rare tokens get fewer training examples each. Picking `V` is a real engineering decision, not a detail.

## BPE by hand: how a tokenizer learns its pieces {#bpe-by-hand}

The most common way to learn those subword pieces is **byte-pair encoding (BPE)**. It was adapted for language models by Sennrich, Haddow and Birch in 2015 to handle rare words in translation. The idea is almost embarrassingly simple: start from single characters, find the pair of neighbours that appears most often, glue them into a new token, and repeat until you have as many tokens as you want.

It's like watching which letters always hang out together and giving each clique its own name tag.

Let's do it by hand. Our training corpus is four words with their counts. We split each word into characters and add an end-of-word marker `_` so the tokenizer can tell `est` at the end of a word from `est` in the middle.

```text
 word      count   starting pieces
 low         5     l o w _
 lower       2     l o w e r _
 newest      6     n e w e s t _
 widest      3     w i d e s t _
```

**Count every adjacent pair, weighted by word count.** For example `e s` appears in `newest` (6) and `widest` (3), so it scores 9.

```text
 pair   count          pair   count          pair   count
 e s      9            l o      7            n e      6
 s t      9            o w      7            e w      6
 t _      9            w e      8  (lower 2 + newest 6)
 w _      5            w i      3   i d  3   d e  3   e r  2   r _  2
```

Now merge, one rule at a time. Ties are broken by a fixed rule (here, first in the list), which is part of what makes a trained tokenizer deterministic.

| Merge # | Rule | Why | Words afterwards |
|---|---|---|---|
| 1 | `e` + `s` → `es` | count 9 | `n e w es t _`, `w i d es t _` |
| 2 | `es` + `t` → `est` | count 9 | `n e w est _`, `w i d est _` |
| 3 | `est` + `_` → `est_` | count 9 | `n e w est_`, `w i d est_` |
| 4 | `l` + `o` → `lo` | count 7 | `lo w _`, `lo w e r _` |
| 5 | `lo` + `w` → `low` | count 7 | `low _`, `low e r _` |

After five merges our vocabulary is the starting characters plus `es, est, est_, lo, low`. A real tokenizer runs tens of thousands of merges over gigabytes of text.

**Now the payoff: tokenizing a word it never saw.** Take `lowest`. Start with `l o w e s t _` and replay the merge rules *in the order they were learned*:

```text
 l o w e s t _
 l o w es t _        rule 1
 l o w est _         rule 2
 l o w est_          rule 3
 lo w est_           rule 4
 low est_            rule 5     →  2 tokens: [low] [est_]
```

`lowest` wasn't in the training data, yet it comes out as two meaningful pieces. That's the whole trick. A word like `xyzzy` would stay as single characters: long, but never unknown.

> 🎬 **Animation — BPE merges by hand:** show the four words `low ×5`, `lower ×2`, `newest ×6`, `widest ×3` as rows of letter tiles with a `_` tile at the end. Step 1: a pair-count table fades in; `e s = 9` pulses. Step 2: every adjacent `e`,`s` tile pair in all rows snaps together into one wider `es` tile, and "rule 1: e+s" is written into a growing "merge list" panel on the right. Step 3–5: repeat for `es+t`, `est+_`, `l+o`, `lo+w`, recounting the table each time. Final step: a new word `lowest` drops in as seven tiles, and the five rules replay from the merge list, snapping tiles until only `low` and `est_` remain.

**What production tokenizers add.** The Hugging Face Tokenizers library describes the full pipeline as four stages, and it's worth knowing the names because bugs live in each one:

```text
 raw text
   │ 1. normalizer      e.g. Unicode NFC/NFKC, maybe lowercasing (optional)
   ▼
   │ 2. pre-tokenizer   split into chunks the model may not merge across
   ▼                    (e.g. on whitespace; digits split from letters)
   │ 3. model           BPE / WordPiece / Unigram → token IDs
   ▼
   │ 4. post-processor  add special tokens (begin-of-text, separators)
   ▼
 IDs   ──(decoder reverses the mapping back to text)
```

Two variants you'll hear about. **Byte-level BPE**, introduced with GPT-2, starts from the 256 possible byte values instead of Unicode characters, so the base alphabet is tiny and no input is ever unknown. **WordPiece** (used by BERT) and **Unigram** are alternative subword algorithms with different ways of choosing pieces. Don't assume a particular model uses any of these; check its tokenizer config.

## Tokens are the unit you pay for: counting, cost, and context budget {#token-budget}

Here's where tokenization stops being academic. **Everything is metered in tokens.** API pricing is per token. The model's **context window**, the maximum number of tokens it can look at in one go (prompt plus generated output), is in tokens. Latency scales with output tokens, because each one is a separate trip through the model. So you need a feel for how many tokens your text is.

The rule of thumb from OpenAI's `tiktoken` README is that a token is about **4 bytes** of text on average (it's a rough English-ish average; your text will differ). So 1 MB of plain English prose is on the order of 1,000,000 / 4 = 250,000 tokens. Use that for napkin maths, and use the real tokenizer for anything that matters.

**Worked example: budgeting a context window.** Say you're building a support assistant on a model with a 128,000-token window. Prices here are illustrative: $3 per million input tokens and $15 per million output tokens.

| Piece of the prompt | Tokens |
|---|---|
| System prompt | 1,500 |
| Tool definitions | 2,500 |
| 8 retrieved document chunks × 800 | 6,400 |
| Conversation history | 20,000 |
| New user message | 600 |
| **Input total** | **31,000** |
| Reserved for the answer (max output) | 4,000 |
| **Committed** | **35,000** |
| Headroom | 93,000 |

Input cost per request: 31,000 × $3 / 1,000,000 = $0.093. Worst-case output: 4,000 × $15 / 1,000,000 = $0.06. Notice the history line: every turn re-sends the whole conversation, so a long chat gets more expensive per message even if each message is short. That's the kind of thing interviewers want you to spot.

**Why equal-looking text costs different amounts.** Tokenizers are trained on a corpus, and they get efficient at whatever that corpus had lots of. Everything else gets chopped finer.

- **Other languages.** In UTF-8, `hello` is 5 bytes, `héllo` is 6, the single Chinese character `猫` ("cat") is 3, and the Hindi word `नमस्ते` is 18 bytes for 6 code points. If the tokenizer saw little of a language, its text falls back towards bytes and the token count balloons. Petrov et al. (NeurIPS 2023) found that the same text translated into different languages can differ in tokenized length by up to 15× on some tokenizers. That's a cost gap, a latency gap and a context gap for those users.
- **Code.** Indentation, long identifiers, and punctuation-heavy syntax tokenize very differently from prose. Four spaces might be one token or four, depending on whether the tokenizer learned whitespace runs.
- **Numbers.** `12345` might be one token, or `123`+`45`, or five digits, depending on the pre-tokenizer. This is one reason arithmetic is awkward for LLMs: the model doesn't see digits in neat columns.
- **Whitespace and case.** ` Paris`, `Paris`, and ` paris` can all be different IDs. A stray leading space in a prompt template changes what the model sees.

```text
 same meaning, different token counts (illustrative, not a real tokenizer)

 "the cat"      ▕▔▔▔▔▕▔▔▔▔▔▕                     2 tokens
 "猫"            ▕▔▔▕▔▔▕▔▔▕                        3 byte tokens (rare in training)
 "    return x" ▕▔▔▔▔▕▔▔▔▔▔▔▔▕▔▔▕                   3 tokens (if 4 spaces merged)
 "    return x" ▕▔▕▔▕▔▕▔▕▔▔▔▔▔▔▔▕▔▔▕                6 tokens (if not)
```

> 🎬 **Animation — same text, different bills:** three horizontal strips labelled English, Hindi and Python, each holding a sentence of equal meaning. Step 1: token boundaries slice each strip into coloured segments (English ~10 segments, Hindi ~30, Python ~16; illustrative). Step 2: each strip drops a stack of coins proportional to its segment count, and a context-window bar at the bottom fills up faster for Hindi. Caption: "The model is billed and limited by tokens, not characters."

**The tokenizer is part of the model.** A checkpoint's weights were learned from the exact ID sequences its own tokenizer and chat template produced. Pair it with a different tokenizer, or change the template's whitespace or role markers, and you're feeding it sequences unlike anything it trained on. Quality can drop even though not one weight changed. In production, pin the tokenizer and template alongside the weights, and count tokens with the real tokenizer, not `string.length / 4`.

## From ID to vector: the embedding table is just a lookup {#embedding-lookup}

Token IDs are just labels. ID 17 isn't "bigger" than ID 3 in any meaningful sense, so you can't do maths on the raw integers. The model needs each token as a list of numbers it can learn to adjust. That list is a **vector**, and we call its length `d` (the model's hidden width, often written `d_model`). Real models use `d` in the thousands.

Where do those vectors come from? The **embedding table**, usually called `E`: a matrix with `V` rows (one per vocabulary entry) and `d` columns. Looking up a token means **taking row number = token ID**. That's it. It's a spreadsheet lookup, not a calculation.

Here's a toy table with `V = 6` and `d = 4` (rows = token IDs, columns = features, numbers made up):

```text
            f0     f1     f2     f3
 ID 0 <eot> 0.00   0.00   0.00   0.00
 ID 1 " the"0.10  -0.20   0.05   0.30
 ID 2 " cat"0.80   0.10   0.60   0.00
 ID 3 " sat"-0.30  0.70   0.20  -0.10
 ID 4 " dog"0.70   0.20   0.50   0.10
 ID 5 " car"0.10   0.90   0.00   0.40
```

Input `" the cat sat"` → IDs `[1, 2, 3]` → pick rows 1, 2, 3 → a `3 × 4` matrix where **row i is token position i** and **column j is feature j**. For a batch of `B` sequences each `T` tokens long, `B × T` IDs become a `B × T × d` block of numbers. That shape stays the same all the way through the stack.

> 🎬 **Animation — embedding lookup:** show the 6 × 4 toy table above as a grid. Step 1: the text " the cat sat" splits into three chips labelled 1, 2, 3. Step 2: each chip flies to the left edge of the table and its row lights up. Step 3: the lit rows slide out and stack into a new 3 × 4 grid labelled "rows = positions, columns = features". Step 4: a side panel shows the equivalent maths: a one-hot row vector [0,0,1,0,0,0] multiplied by the table returns row 2, with the caption "lookup = multiply by a one-hot vector, but way cheaper".

**Why "lookup" and "matrix multiply" are the same thing.** If you wrote ID 2 as a **one-hot vector** (all zeros except a 1 in position 2) and multiplied it by `E`, you'd get row 2 back. So an embedding layer is mathematically a linear layer, but nobody multiplies by a 100,000-wide vector of zeros. Frameworks just index the row. This matters because it means the table is **learned by gradient descent** like any other weight: rows start random, and training nudges them wherever they need to be so the rest of the model can predict well. (How the nudging works is in **How a model learns: loss, gradients, and optimizers**.) One consequence: a token that almost never appeared in training has a row that barely moved from random. Those "under-trained tokens" can make models behave oddly when they show up.

**Sizing it.** The table has `V × d` numbers. With a hypothetical `V = 50,000` and `d = 4,096`:

```formula
params = V × d = 50,000 × 4,096 = 204,800,000
bytes  = params × 2 (bf16) = 409,600,000 B ≈ 390.6 MiB
```

`bf16` is a 16-bit (2-byte) number format commonly used to store weights; a MiB is 2²⁰ = 1,048,576 bytes. Bump the vocabulary to `V = 128,000` with the same `d` and it's 524,288,000 parameters, exactly 1,000 MiB at 2 bytes each, about 7.5% of a 7-billion-parameter model *per table*. And there's usually a second table of the same shape at the other end of the model, which is our next stop. Those numbers exclude gradients and optimizer state during training, which multiply the memory several times over; that accounting lives in **Distributed training: fitting a training run onto a cluster**.

## What vectors buy you, and the black box in the middle {#vectors-and-the-stack}

Why go to the trouble of vectors at all? Because vectors have **geometry**. You can measure how close two of them are, and training discovers that it helps to put tokens that behave alike near each other. Think of each token as a pin on a map with thousands of dimensions instead of two, where "nearby" means "used in similar ways".

The standard closeness measure is **cosine similarity**: the cosine of the angle between two vectors. 1 means pointing the same way, 0 means unrelated (at right angles), −1 means opposite.

```formula
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

`a · b` is the **dot product** (multiply matching entries and add them up), and `‖a‖` is the vector's length, √(a · a). Let's check it on our toy rows:

```text
 cat = [0.8, 0.1, 0.6, 0.0]   ‖cat‖ = √1.01 ≈ 1.005
 dog = [0.7, 0.2, 0.5, 0.1]   ‖dog‖ = √0.79 ≈ 0.889
 car = [0.1, 0.9, 0.0, 0.4]   ‖car‖ = √0.98 ≈ 0.990

 cat·dog = 0.56 + 0.02 + 0.30 + 0.00 = 0.88   → cos ≈ 0.88 / (1.005 × 0.889) ≈ 0.985
 cat·car = 0.08 + 0.09 + 0.00 + 0.00 = 0.17   → cos ≈ 0.17 / (1.005 × 0.990) ≈ 0.171
```

`cat` and `dog` point nearly the same way; `car` is off in another direction, even though "cat" and "car" differ by one letter. The spelling similarity is irrelevant; only learned usage counts.

> 🎬 **Animation — vectors as arrows:** squash the toy vectors into a 2-D plot (use features f0 and f1 only: cat (0.8, 0.1), dog (0.7, 0.2), car (0.1, 0.9)). Step 1: draw each as an arrow from the origin. Step 2: shade the small angle between cat and dog and label it "cos ≈ 0.98 (4-D value)". Step 3: shade the wide angle between cat and car, label "cos ≈ 0.17". Step 4: fade in a cloud of other animal words near cat/dog and vehicle words near car, captioned "training pulls tokens that behave alike together".

Two cautions that interviewers like. First, the **input embedding of a single token is context-free**: ` bank` gets the same starting row whether the sentence is about rivers or money. Sorting out which meaning applies is the job of the layers in the middle. Second, the famous "king − man + woman ≈ queen" arithmetic came from older word-vector models; treat it as intuition, not a guarantee about any LLM's embedding table. Sentence-level embeddings used for search are a different thing again, which we cover in **RAG: giving the model the right evidence**.

**The black box.** Between the embedding lookup and the output sits a stack of `L` identical **transformer blocks**. For this article, all you need is its contract:

```text
 in:   T × d   one context-free vector per position
        │
        │  block 1: positions exchange information  ("attention")
        │           then each position is processed alone ("MLP")
        │  block 2 … block L: same again, refining
        ▼
 out:  T × d   same shape; each row now reflects its whole prefix
```

Each block lets every position pull in information from earlier positions, so by the top, the row for the last position of `The capital of France is` "knows" it's in a sentence about France's capital. How tokens pull information from each other is **Attention from scratch: how tokens talk to each other**. How blocks are wired together (position information, residual connections, normalization, the MLP) is **The transformer block: assembling the full model**. For now, treat it as a function that takes a `T × d` matrix and returns a smarter `T × d` matrix.

## Turning the last vector into scores: unembedding and logits {#unembedding}

At the top of the stack we have one `d`-long vector per position. To predict the next token we only need the **last** position's vector (during generation; in training every position predicts its own next token). Call it `h`. We need to turn `h` into one score per vocabulary entry.

The tool is another `V × d` table, usually called the **output embedding** or **unembedding** matrix, `W_out`. Each of its rows is a learned vector for one token, and the score for token `i` is just the dot product of `h` with row `i`. These raw scores are called **logits**.

```formula
z = W_out · h        z_i = (row i of W_out) · h
```

`z` is the vector of `V` logits; `z_i` is token `i`'s score. Logits can be any real number (negative, positive, huge), and they don't sum to anything in particular.

Intuition: each row of `W_out` is a "template" for its token, and the dot product asks "how well does the model's current state match this token's template?". Tiny example with `d = 3` and a three-token vocabulary:

```text
 h (final hidden vector, last position) = [1, 0, 1]

 W_out rows          dot with h                 logit
 " cat"  [1, 0, 1]   1·1 + 0·0 + 1·1        =   2
 " dog"  [0.5,0.5,0.5] 0.5 + 0 + 0.5        =   1
 " car"  [0, 1, 0]   0 + 0 + 0              =   0

 logits z = [2, 1, 0]
```

> 🎬 **Animation — unembedding:** show the hidden vector h = [1, 0, 1] as a vertical column of three cells on the left. On the right, the three W_out rows stacked. Step 1: h slides across the " cat" row; matching cells glow, products 1, 0, 1 appear and sum to 2. Step 2: same for " dog" (0.5, 0, 0.5 → 1) and " car" (0, 0, 0 → 0). Step 3: the three sums drop into a bar chart of logits [2, 1, 0], labelled "raw scores, not probabilities yet". Final frame zooms out to show the real shape: one d-vector against a V × d wall, producing V bars.

**Tied weights.** Notice `W_out` has exactly the same shape as the input table `E`. So an obvious question is: why not use the same matrix for both? That's **weight tying**, proposed by Press and Wolf (2016), and the original transformer paper did it too ("we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation"). The two tables answer related questions: `E` asks "what vector should represent this token going in?", while `W_out` asks "what final state should make this token likely next?".

| | Untied | Tied |
|---|---|---|
| Parameters for the two tables | 2 × V × d | V × d |
| Our V=50k, d=4,096 example (bf16) | ≈ 781 MiB | ≈ 391 MiB |
| Flexibility | Each table specializes | One table serves both jobs |
| Where you see it | Many large models | Common in smaller models, where vocabulary tables are a big share of all parameters |

It's an architecture decision fixed at training time, not a runtime flag. Also, the output projection is a real cost at inference: every generated token does a `V × d` multiply, which is why a huge vocabulary isn't free.

## Softmax and temperature: from scores to probabilities {#softmax-temperature}

Logits aren't probabilities. We need to turn `[2, 1, 0]` into positive numbers that sum to 1, keeping the order. **Softmax** does exactly that: exponentiate each score (which makes everything positive and exaggerates gaps), then divide by the total.

```formula
p_i = exp((z_i − max(z)) / τ) / Σ_j exp((z_j − max(z)) / τ)        for τ > 0
```

`p_i` is the probability of token `i`, `z_i` its logit, `Σ_j` sums over all `V` tokens, and `τ` (tau) is the **temperature**, explained below; plain softmax is `τ = 1`. Subtracting `max(z)` doesn't change the answer (it cancels top and bottom) but keeps `exp` from overflowing, which is how every real implementation does it.

**Worked example at τ = 1**, logits `[2, 1, 0]`:

```text
 exp(2) = 7.389   exp(1) = 2.718   exp(0) = 1.000     sum = 11.107
 p = [7.389, 2.718, 1.000] / 11.107 = [0.6652, 0.2447, 0.0900]
```

**Temperature** divides the logits before softmax. It's like the contrast knob on a photo. Turn it down (`τ < 1`) and the gaps between logits get bigger, so the top token dominates. Turn it up (`τ > 1`) and the gaps shrink, so the distribution flattens and unlikely tokens get more airtime.

| τ | Scaled logits | Probabilities |
|---|---|---|
| 0.5 | [4, 2, 0] | [0.8668, 0.1173, 0.0159] |
| 1.0 | [2, 1, 0] | [0.6652, 0.2447, 0.0900] |
| 2.0 | [1, 0.5, 0] | [0.5065, 0.3072, 0.1863] |

As `τ → 0`, all the mass goes to the top token (that's greedy decoding, below). As `τ → ∞`, it approaches uniform. For any positive `τ`, **the ranking never changes**: temperature changes how concentrated the choice is, not which token is best.

> 🎬 **Animation — the temperature knob:** a bar chart of three tokens (" cat", " dog", " car") with a dial underneath labelled τ. Start at τ = 1 showing bars 0.665 / 0.245 / 0.090. Turn the dial to 0.5: bars morph to 0.867 / 0.117 / 0.016 and the top bar turns bold. Turn to 2.0: bars morph to 0.507 / 0.307 / 0.186. Keep a small caption that never changes: "order stays cat > dog > car". Finish by sliding τ towards 0 and showing the top bar reach 1.0 with the label "greedy".

The big misconception to kill: **temperature doesn't add or remove knowledge.** Low temperature makes the model more *consistent*, not more *correct*. If the model's top guess is wrong, `τ = 0.1` gives you that wrong answer every time, confidently. And the model's probabilities aren't guaranteed to be calibrated (a 0.9 doesn't necessarily mean "right 90% of the time"), especially after post-training.

## Choosing a token: greedy, top-k, top-p, stopping, and determinism {#decoding}

You now have a probability for every token. The rule for actually picking one is the **decoding policy** (also called the sampling strategy). It's a choice you make at serving time; the model doesn't care. Let's use the five-token example from the first section, with logits `[3.0, 2.0, 1.5, 0.5, −1.0]` for ` Paris`, ` the`, ` a`, ` Lyon`, ` banana`:

```text
 token      logit   p (τ=1)   cumulative
 " Paris"    3.0    0.5912     0.5912
 " the"      2.0    0.2175     0.8087
 " a"        1.5    0.1319     0.9406
 " Lyon"     0.5    0.0485     0.9891
 " banana"  −1.0    0.0108     1.0000  (rounding)
```

**Greedy decoding:** always take the top token (` Paris`). It's deterministic in principle and good for short, factual or structured outputs. It's the default in Hugging Face Transformers' `generate()`. On long outputs it tends to fall into repetitive loops.

**Sampling:** roll a weighted die over the distribution, so ` the` comes up about 22% of the time. More varied, but occasionally the die lands on a tail token like ` banana`, and one bad token early can derail everything after it (since it becomes input).

That's why we usually **truncate the tail first**, then renormalize (rescale the survivors so they sum to 1):

- **Top-k** keeps the `k` most likely tokens. With `k = 3`: keep 0.5912, 0.2175, 0.1319 (sum 0.9406) → renormalized 0.6285, 0.2312, 0.1402.
- **Top-p (nucleus sampling)**, from Holtzman et al. (2019), keeps the *smallest* set of top tokens whose probabilities add up to at least `p`. With `p = 0.9`: 0.5912 isn't enough, 0.8087 isn't enough, 0.9406 is, so keep three tokens (same as top-k=3 here). With `p = 0.8`: 0.8087 already clears it, so keep two → 0.731 / 0.269.

The difference matters when the model's confidence changes. Top-k keeps `k` tokens whether the model is sure or clueless. Top-p adapts: when the model is confident, the nucleus might be one or two tokens; when it's unsure, it widens. And the knobs stack: at `τ = 0.5` the same logits give 0.8388 / 0.1135 / 0.0418 / …, so `p = 0.9` keeps just two tokens (cumulative 0.9523).

```text
 logits ──► ÷ τ ──► softmax ──► top-k / top-p filter ──► renormalize ──► sample (or argmax)
```

> 🎬 **Animation — truncating the tail:** show the five-bar chart (Paris 0.591, the 0.218, a 0.132, Lyon 0.049, banana 0.011) with a running cumulative line above it. Step 1 (top-k=3): a vertical cutter slides in after the third bar; Lyon and banana grey out; the remaining bars grow to 0.629 / 0.231 / 0.140. Step 2 (reset, top-p=0.8): a horizontal line at 0.8 on the cumulative axis; bars are admitted left to right until the cumulative line crosses it at " the" (0.809); survivors grow to 0.731 / 0.269. Step 3: a die rolls and lands on " Paris".

**Greedy isn't "best sequence".** Picking the best token at each step doesn't guarantee the most probable *whole* output. Toy case: step 1 offers A (0.6) or B (0.4). After A, the best continuation has 0.3; after B, it has 0.9. Greedy takes A and ends at 0.6 × 0.3 = 0.18, while B-then-best gives 0.4 × 0.9 = 0.36. **Beam search** keeps several candidate sequences alive to catch cases like this; it's common in translation and speech, less so in chat.

**Stop conditions.** Something has to end the loop, and in production it's always a combination:

| Stop condition | What it is | Gotcha |
|---|---|---|
| End-of-sequence token | The model emits a special "I'm done" token | Wrong template → model never emits it and rambles |
| Max new tokens | Hard cap on output length | Truncates mid-sentence; also your cost ceiling (HF's default cap is just 20 tokens unless configured) |
| Stop strings | Halt when e.g. `\nUser:` appears | Must be matched on decoded text, since a stop string can span token boundaries |
| Structured-output constraints | Mask out tokens that would break a JSON schema or grammar | Constrains form, not truth |

Repetition penalties (down-weighting tokens already used) are another common knob that edits logits before sampling.

**Determinism: what "temperature 0" can and can't promise.** Setting greedy decoding (or a fixed random seed for sampling) removes the *deliberate* randomness. It doesn't guarantee bit-identical outputs in a real serving system. GPU floating-point maths isn't perfectly associative (adding numbers in a different order can change the last bits), and the order can depend on batch size and which other requests share the batch. When two logits are nearly tied, a tiny difference flips the argmax, and because each token feeds the next, the outputs diverge from there. So "reproducible" in practice means: pin the model, tokenizer, template and decoding config, log them with every experiment, and design evaluations that tolerate small variation. How batching works on a server is in **Serving many users: batching, scheduling, and speculative decoding**.

**Picking a policy, the senior version.** Match it to the task and say why:

| Task | Typical choice | Reasoning |
|---|---|---|
| Extraction, classification, JSON | Greedy / low τ + schema constraint + tight max tokens | You want the single most likely answer, and a parse error is worse than blandness |
| Code | Low τ, maybe sample several and test them | Correctness is checkable, so diversity plus a verifier beats one guess |
| Brainstorming, fiction | τ ≈ 0.7–1.0 with top-p | Diversity is the point; top-p trims the nonsense tail |

Every one of these is a policy on top of the same model. Changing it changes behaviour without touching a single weight.

# Interview

## Question

Walk me through what happens between a user typing "The capital of France is" and the model returning " Paris". Where does cost come from, and which knobs change the output without retraining?

## Answer

First the application serializes the request, including any system prompt and conversation history, into one string using the model's chat template, and the tokenizer turns it into a list of token IDs. The tokenizer is typically byte-level BPE or similar: learned merge rules turn text into subword pieces, so common words are one token and rare strings break into smaller pieces, down to bytes if needed. Everything downstream is priced and limited in these tokens, not characters, and the ratio varies a lot across languages, code and numbers.

Each ID selects a row of the `V × d` embedding table, giving a `T × d` matrix (rows are positions, columns are features). That goes through `L` transformer blocks, where attention mixes information across earlier positions and MLPs process each position, and comes out as the same shape. The last position's vector is multiplied by the output embedding (possibly tied to the input table) to give `V` logits, one score per vocabulary token.

The decoding policy then turns logits into a choice: divide by temperature, softmax to probabilities, optionally truncate with top-k or top-p, renormalize, then sample or take the argmax. " Paris" is appended and the loop repeats until an end-of-sequence token, a stop string, or the max-token limit. Because every new token depends on the previous one, generation is sequential: one full model pass per output token, which is why output tokens dominate latency.

Cost comes from input tokens (processed once, in parallel), output tokens (sequential), and re-sending history each turn. Knobs that change output without retraining: the prompt and template, temperature, top-k/top-p, repetition penalties, stop conditions, max tokens and constrained decoding. None of them add knowledge; low temperature makes answers consistent, not correct.

## Follow-ups

- Why might the same prompt cost three times as much in Hindi as in English, and what could a model builder do about it?
- Your team sets temperature to 0 but sees different outputs for the same prompt in production. Explain why, and what you'd do about it.
- When would you untie the input and output embeddings, and what does it cost in memory for V = 128,000, d = 4,096 in bf16?
- Show a case where greedy decoding doesn't return the most probable sequence. What does beam search cost you?
- Top-k or top-p: which adapts to the model's confidence, and why does that matter?

# Pitfalls

- Treating tokens as words or characters. Context limits and prices are in tokens, and the ratio shifts with language, code, numbers and whitespace; `length / 4` is only a napkin estimate.
- Thinking an embedding lookup multiplies the token ID by something. The ID is a row index; ID 17 selects row 17 and has no numeric meaning.
- Believing low temperature makes the model more accurate. It only sharpens the distribution without changing the ranking, so a wrong top answer becomes a consistently wrong answer.
- Assuming temperature 0 means bit-identical outputs in production. Floating-point order and batching can flip near-ties, and differences compound token by token.
- Swapping tokenizers or editing the chat template and expecting the same quality. The weights were trained on a specific ID sequence format.
- Saying the model "remembers" the conversation. The app re-sends the history every turn; memory is an application feature, and the history costs tokens every time.
- Assuming greedy decoding finds the most likely full response. It's locally optimal per step, not globally.

# Checklist

- Explain the next-token contract and write the chain-rule factorization of a sequence's probability.
- Run five BPE merges by hand and tokenize an unseen word with the learned rules.
- Build a context-window budget for a real prompt and estimate its per-request cost.
- Explain why languages, code, numbers and leading spaces change token counts.
- Size an embedding table in parameters and MiB, and say what weight tying saves.
- Compute softmax for three logits at two temperatures, and apply top-k and top-p by hand.
- Pick and defend a decoding configuration, including stop conditions, for extraction vs creative tasks.
- Explain why generation is sequential and why "temperature 0" isn't a reproducibility guarantee.

# Sources

- [Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow, Birch, 2015/ACL 2016)](https://arxiv.org/abs/1508.07909) — Introduces BPE-based subword segmentation to handle rare and unseen words.
- [Hugging Face Tokenizers: Components](https://huggingface.co/docs/tokenizers/main/en/components) — The normalizer / pre-tokenizer / model / post-processor pipeline, byte-level pre-tokenization from GPT-2 with its 256-symbol base alphabet, and BPE vs WordPiece vs Unigram.
- [openai/tiktoken README](https://github.com/openai/tiktoken) — The "about 4 bytes per token" rule of thumb and an example of BPE splitting words into reusable subwords.
- [Language Model Tokenizers Introduce Unfairness Between Languages (Petrov et al., NeurIPS 2023)](https://arxiv.org/abs/2305.15425) — Token lengths for the same text can differ by up to 15× across languages, affecting cost, latency and context.
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/html/1706.03762v7) — Section 3.4: learned embeddings, a learned linear layer plus softmax for next-token probabilities, and one shared weight matrix for the embeddings and pre-softmax projection.
- [Using the Output Embedding to Improve Language Models (Press and Wolf, 2016)](https://arxiv.org/abs/1608.05859) — The output matrix is itself a word embedding; tying it to the input embedding helps perplexity and shrinks models.
- [The Curious Case of Neural Text Degeneration (Holtzman et al., ICLR 2020)](https://arxiv.org/abs/1904.09751) — Introduces nucleus (top-p) sampling to truncate the unreliable tail while keeping diversity.
- [Hugging Face Transformers: Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) — Greedy search as the default (20 new tokens unless configured), multinomial sampling, and beam search.

# Flashcards

## next-token-contract

**Q:** What does a decoder language model actually output, and how does it produce a long answer?

A probability for every token in its vocabulary, given the tokens so far. To produce a long answer you run a loop: choose one token, append it to the input, and run the model again. That's autoregressive generation.

Chat roles, tool results and history reach the model only because the app serializes them into one token sequence with a chat template. The model itself doesn't manage conversation state.

## chain-rule-probability

**Q:** How does next-token prediction give the probability of a whole sequence?

By the chain rule: P(x₁…x_T) = P(x₁) · P(x₂|x₁) · … · P(x_T|x₁…x_{T−1}). Each factor is one next-token prediction. For example, 0.05 × 0.01 × 0.2 = 0.0001.

In practice you sum log-probabilities instead (ln 0.05 + ln 0.01 + ln 0.2 ≈ −9.21), because multiplying many small numbers underflows.

## why-subwords

**Q:** Why do LLMs use subword tokens instead of words or characters?

Word vocabularies can't cover the long tail (names, typos, code, other languages), so unseen words become an unknown token. Characters never fail but make sequences several times longer and force the model to learn spelling before meaning.

Subwords keep frequent strings as single tokens and decompose rare ones into reusable pieces, with a byte fallback so nothing is unknown. You get short sequences for common text and full coverage.

## bpe-merges

**Q:** How does BPE learn its vocabulary, and how does it tokenize a word it never saw?

Training starts from characters (or bytes), counts every adjacent pair across the corpus weighted by frequency, merges the most frequent pair into a new token, and repeats until the vocabulary reaches the target size. The ordered list of merges is the tokenizer.

To tokenize new text, split it into base units and replay the merges in learned order. In the toy corpus (low, lower, newest, widest), the unseen word "lowest" becomes [low][est_].

## tokens-context-budget

**Q:** Why can two strings of the same length consume very different amounts of context and money?

Because tokenizers are efficient on what their training corpus had lots of. Under-represented languages fall back to smaller pieces or bytes (Petrov et al. found up to 15× differences for the same text on some tokenizers), and code, numbers and whitespace runs tokenize unpredictably.

Context limits and prices are in tokens, so count with the model's real tokenizer. About 4 bytes per token is only an English-ish rule of thumb.

## tokenizer-is-part-of-model

**Q:** Why can't you swap a model's tokenizer or tweak its chat template freely?

The weights were learned on the exact ID sequences that tokenizer and template produced. A different tokenizer maps text to different IDs (so different embedding rows), and a changed template produces marker sequences the model never saw.

Quality can drop even though no weight changed. Pin the tokenizer and template together with the checkpoint.

## embedding-lookup

**Q:** What does the embedding layer do to token ID 17?

It selects row 17 of the V × d embedding table and returns that d-long vector. It does not multiply anything by the number 17; the ID is just a row index.

Mathematically this equals multiplying a one-hot vector by the table, but frameworks just index the row. The rows are learned weights, so rarely seen tokens can end up with poorly trained vectors.

## embedding-size

**Q:** How big is a 50,000 × 4,096 embedding table in bf16?

50,000 × 4,096 = 204,800,000 parameters. At 2 bytes each that's 409,600,000 bytes, about 390.6 MiB.

That excludes gradients and optimizer state. An untied output embedding doubles it; with V = 128,000 and the same d, each table is 1,000 MiB, about 7.5% of a 7B model.

## cosine-similarity

**Q:** What is cosine similarity, and what does it tell you about two token embeddings?

cos(a, b) = (a · b) / (‖a‖‖b‖): the cosine of the angle between the vectors, from −1 (opposite) through 0 (unrelated) to 1 (same direction). Training tends to place tokens that are used similarly in similar directions.

In the toy example, cat·dog gives ≈ 0.985 and cat·car ≈ 0.171, even though "cat" and "car" are one letter apart. Input embeddings are context-free, though: " bank" has one row regardless of meaning.

## logits-unembedding

**Q:** How does the model turn its final hidden vector into scores for every token?

It multiplies the last position's d-long hidden vector h by the V × d output embedding (unembedding) matrix. Each logit is the dot product of h with that token's row.

Logits are unbounded real numbers that don't sum to 1; softmax turns them into probabilities. This V × d multiply runs for every generated token, so a large vocabulary adds inference cost.

## weight-tying

**Q:** What is weight tying, and what's the tradeoff?

Using the same V × d matrix as both the input embedding table and the output projection. It saves a whole V × d allocation (≈ 391 MiB for V = 50k, d = 4,096 in bf16) and was used in the original transformer, following Press and Wolf.

Untied tables cost that memory but let each specialize: one for "represent this token going in", one for "which state should make this token likely next". It's fixed by the architecture, not a runtime switch.

## temperature

**Q:** What does lowering temperature do, and what doesn't it do?

It divides logits by τ < 1 before softmax, widening the gaps so the distribution concentrates on the top tokens. Logits [2,1,0] give [0.665, 0.245, 0.090] at τ = 1 and [0.867, 0.117, 0.016] at τ = 0.5.

For any positive τ the ranking never changes. It doesn't add knowledge or fix a false premise; it makes answers more consistent, not more correct.

## top-k-top-p

**Q:** How do top-k and top-p (nucleus) sampling differ?

Top-k keeps a fixed number k of the most likely tokens. Top-p keeps the smallest set of top tokens whose cumulative probability reaches p, so its size adapts: small when the model is confident, larger when it's unsure. Both then renormalize and sample.

Example: probabilities [0.591, 0.218, 0.132, 0.049, 0.011] with p = 0.8 keeps two tokens (cumulative 0.809), renormalized to 0.731 / 0.269.

## greedy-not-optimal

**Q:** Why doesn't greedy decoding necessarily find the most probable sequence?

It picks the best token at each step without looking ahead. If A (0.6) leads to a best follow-up of 0.3, while B (0.4) leads to 0.9, greedy ends at 0.18 while B-then-best gives 0.36.

Beam search keeps several partial sequences to catch this, at extra compute. Greedy also tends to fall into repetitive loops on long outputs.

## determinism

**Q:** Why can "temperature 0" still give different outputs in production?

Greedy decoding removes deliberate randomness, but GPU floating-point addition isn't perfectly associative, and the order of operations can depend on batch size and co-scheduled requests. A near-tie between two logits can flip, and since each token feeds the next, outputs diverge from there.

For reproducibility, pin model, tokenizer, template and decoding config, log them, and build evals that tolerate small variation.

## stop-conditions

**Q:** What ends a generation loop, and what goes wrong with each?

An end-of-sequence token (a bad template can stop it ever being emitted), a max-new-tokens cap (truncates mid-sentence, but bounds cost), stop strings (must be matched on decoded text because they can span token boundaries), and structured-output constraints that mask invalid tokens (they enforce form, not truth).

Production systems combine several of these.
