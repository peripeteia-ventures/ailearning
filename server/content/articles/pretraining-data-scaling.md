---
{
  "slug": "pretraining-data-scaling",
  "title": "Pretraining: data, compute, and scaling laws",
  "category": "training",
  "summary": "How a base model gets made: turning a web crawl into a clean, deduplicated training mixture, estimating the compute bill with C ≈ 6·N·D, and using scaling laws (Kaplan, Chinchilla, and inference-aware variants) to pick a model size and token budget without over-trusting the fit.",
  "difficulty": "Advanced",
  "minutes": 30,
  "prerequisites": ["transformer-foundations", "optimization-generalization"],
  "learningObjectives": [
    "Describe a pretraining data pipeline (extraction, filtering, quality classifiers, dedup, contamination checks, mixtures) and compute how much data survives it.",
    "Estimate training compute with C ≈ 6·N·D, derive where the 6 comes from, and turn FLOPs into wall-clock time under a stated throughput assumption.",
    "Explain the Kaplan vs Chinchilla disagreement and use the ~20 tokens-per-parameter heuristic to size a compute-optimal run.",
    "Calculate when training a smaller model on more tokens pays for itself through cheaper inference.",
    "Defend a scaling recommendation by stating what the fitted curves can and cannot promise."
  ]
}
---

# Sections

## What pretraining actually buys you {#what-pretraining-produces}

Pretraining is the first and by far the most expensive stage of building an LLM. You take a transformer with random weights, show it trillions of tokens of text, and train it on one job: predict the next token. (A **token** is a chunk of text, usually a word or a piece of a word, that the tokenizer maps to an integer ID. We cover tokenizers in **Text in, next token out: tokens, embeddings, and sampling**.) There are no labels, no human ratings, no instructions. The text itself is the answer key: every position in every document is a little quiz whose correct answer is the token that actually comes next.

What comes out is a **base model**. Think of it as someone who has read an enormous library and absorbed its style, facts and reasoning patterns, but was never taught to hold a conversation. Give it "The capital of France is" and it continues with "Paris". Give it "How do I reverse a list in Python?" and it might answer the question, or it might continue with three more forum questions, because that's also what text on the web looks like. Turning that library-reader into a helpful assistant is post-training, which we'll go through in detail in **From base model to assistant: SFT, RLHF, DPO, and verifiable rewards**.

```text
  raw web, books, code, papers
             │
             ▼
   ┌───────────────────┐     clean, deduplicated, mixed token stream
   │  DATA PIPELINE    │ ──────────────────────────────┐
   └───────────────────┘                               ▼
                                     ┌──────────────────────────────┐
   random weights  ────────────────► │  PRETRAINING (next-token     │
                                     │  prediction, weeks-months)   │
                                     └──────────────┬───────────────┘
                                                    ▼
                                             BASE MODEL
                                     (knows a lot, follows nothing)
                                                    │
                                                    ▼
                                     post-training → assistant
```

The objective is **cross-entropy**: for each position, the negative log of the probability the model gave to the true next token, averaged over all positions. If the model gave the right token probability 0.5, that position costs −ln 0.5 ≈ 0.69 nats; if it gave 0.9, the cost is ≈ 0.105. Lower is better. We derive cross-entropy and the optimizer that minimizes it in **How a model learns: loss, gradients, and optimizers**; here you only need "loss = average surprise per token, measured on text the model hasn't trained on".

Here's the key mental shift for this article: **the model learns whatever lowers that average surprise on the data you feed it.** Reasoning patterns, code idioms, facts, but also spam templates, benchmark answers, and whatever boilerplate appears ten thousand times. The data *is* the specification. So pretraining is really three linked decisions, and the rest of this article takes them in order:

| Decision | Question | Sections |
|---|---|---|
| Data | Which tokens, how clean, in what proportions, how often repeated? | 2–4 |
| Compute | How many FLOPs can we afford, and how long will that take? | 5 |
| Allocation | For that budget, how big a model (N) and how many tokens (D)? | 6–8 |

One piece of vocabulary before we move on. A **training token** (or token *exposure*) is one position the model trained on. If a document appears twice, or you run through a dataset for three **epochs** (full passes), every one of those passes counts. So "trained on 2T tokens" says how much reading happened, not how much *distinct* text existed. Keep that distinction in your head; it comes back three times.

## Building the corpus: from crawl to clean text {#building-the-corpus}

The analogy I like: building a pretraining corpus is like refining crude oil. You start with a huge volume of messy stuff, each stage throws a fraction away, and what's left at the end is much smaller but much more valuable. The biggest single source for most open models is **Common Crawl**, a public archive of scraped web pages. RefinedWeb, the dataset behind the Falcon models, extracted about 5 trillion tokens from Common Crawl and showed that carefully filtered and deduplicated web data *alone* could train models that beat ones trained on hand-curated mixtures like The Pile. That was a big deal, because curated sources don't scale to tens of trillions of tokens and the web roughly does.

A typical pipeline looks like this:

```text
 crawl snapshot ─► extract text ─► language ID ─► heuristic filters ─► quality classifier
   (HTML, PDFs)    (strip menus,   (keep target   (too short, too many   (small model scores
                    ads, markup)    languages)     symbols, repeated      "does this look like
                                                   lines, bad words)      good text?")
                                                                              │
     shards ◄── tokenize ◄── mix & weight ◄── decontaminate ◄── deduplicate ◄─┘
  (fixed files the                              (remove eval     (exact + near
   trainer reads)                                material)        duplicates)
```

A few of these deserve a sentence each:

- **Extraction** turns HTML into plain text. It sounds boring, and it matters a lot: a bad extractor keeps cookie banners and navigation menus, and your model learns to write them.
- **Heuristic filters** are cheap rules: drop pages with very few words, a high ratio of symbols to words, lots of lines ending without punctuation, or the same line repeated many times.
- **Quality classifiers** are small models that score each document, trained to tell "reads like a good reference or textbook" from "reads like spam". Meta's Llama 3 post says they used Llama 2 to generate the training data for the text-quality classifiers behind Llama 3. The catch: a classifier trained on one notion of "good" can quietly delete whole dialects, genres or communities. Always audit a sample of what got *rejected*, not just what got kept.
- **Sharding** writes the final tokenized data into fixed files, with a manifest (source, date, filter versions, checksums). Think of the manifest as a lockfile for your data: it's how you rebuild the exact same corpus later and explain what the model saw.

Now the arithmetic people get wrong. Retention rates **multiply**, because each stage only sees what the previous one kept.

```formula
retained = raw × r₁ × r₂ × r₃ × …
```

Here `raw` is the starting token count and each `rᵢ` is the fraction that stage keeps *of its input*.

Worked example (illustrative rates). Start with 1,000B tokens of crawl. Extraction keeps 80%, quality filtering keeps 75% of that, dedup keeps 70% of that:

```text
 1,000B ──extract 80%──► 800B ──quality 75%──► 600B ──dedup 70%──► 420B
```

So you end with 420B tokens, not the 250B you'd get by subtracting 20 + 25 + 30 = 75 percentage points from the original. And the rates themselves depend on order: dedup before quality filtering sees different duplicates than dedup after. Real pipelines on web data can be far harsher than this, which is why "we crawled 100 trillion tokens" tells you little about the usable budget.

> 🎬 **Animation — the refinery funnel:** a wide horizontal bar labelled "1,000B raw tokens". Step 1: 20% of it greys out and falls away labelled "extraction junk", leaving 800B. Step 2: a quarter of the remaining bar greys out labelled "low quality", leaving 600B. Step 3: 30% of that greys out labelled "duplicates", leaving 420B. Step 4: a ghost bar at 250B appears with a red cross and the caption "subtracting percentages is wrong: rates multiply". Keep the bars left-aligned so the shrinkage is obvious.

## Duplicates, contamination, and repetition that's fine {#dedup-and-contamination}

Web text repeats itself constantly: mirrored news articles, license headers, forum signatures, scraped copies of Wikipedia. But "the same text showing up twice" covers three different situations with three different fixes, and interviewers love checking whether you can tell them apart.

| Kind of overlap | Example | Why it hurts | What to do |
|---|---|---|---|
| **Corpus redundancy** | The same press release on 400 sites | Wastes compute; raises memorization | Deduplicate: cluster copies, keep one |
| **Contamination** | A benchmark question and its answer on a blog | Inflates eval scores; you stop measuring generalization | Detect against eval sets and remove; report overlap |
| **Useful recurrence** | Common code idioms, many independent explanations of recursion | Nothing, it's how the model learns patterns | Leave it alone |

**Deduplication.** Exact dedup is easy: normalize the text (lowercase, collapse whitespace), hash it, drop repeat hashes. Near-duplicates are harder, since two copies of an article differing by a timestamp have different hashes. The standard trick is **MinHash**: break each document into overlapping word n-grams ("shingles"), and estimate how similar the two *sets* of shingles are with compact fingerprints, so you never compare billions of document pairs directly. The similarity being estimated is Jaccard similarity: shared shingles divided by total distinct shingles.

Tiny example with 3-word shingles:

```text
 A: "the cat sat on the mat"   → {the cat sat, cat sat on, sat on the, on the mat}
 B: "the cat sat on the rug"   → {the cat sat, cat sat on, sat on the, on the rug}

 shared = 3      union = 5      Jaccard = 3 / 5 = 0.6
```

Real systems use longer shingles on long documents, and flag pairs above a threshold (often around 0.8) as near-duplicates. Lee et al. found one 61-word English sentence repeated over 60,000 times in C4, a widely used web dataset. Training on their deduplicated versions made models emit memorized text about ten times less often and reach the same accuracy in fewer steps. They also found train-test overlap affecting over 4% of the validation sets of standard datasets. One warning: normalization must be domain-aware. Stripping punctuation is fine for prose and destroys the difference between two different programs.

**Contamination** means evaluation material leaked into training: benchmark questions, worked solutions, translations of them, tutorials built on them. It's the student who saw the answer key: a high score no longer tells you what they learned. Note that **dedup doesn't fix this**. A benchmark answer might appear exactly once in the corpus, so there's nothing to deduplicate. You need a separate check that compares training data against the protected eval sets (usually by n-gram overlap), plus eval sets built after the data cutoff so they *can't* be in the crawl.

Here's how much contamination can distort a number (illustrative). A benchmark has 100 questions, 20 of which have near-identical solved copies in the training data. The model scores 95% on those 20 and 60% on the other 80:

```text
 contaminated: 20 × 0.95 = 19 correct
 clean:        80 × 0.60 = 48 correct
 reported:     (19 + 48) / 100 = 67%      clean-only: 60%
```

The headline says 67%; the honest number is closer to 60%. (It's only "closer": the contaminated 20 might also be easier questions, so this isn't a causal estimate.) The fix is to report scores on the clean and contaminated subsets separately. Eval hygiene more broadly is covered in **Evaluation and observability: knowing whether it actually works**.

> 🎬 **Animation — three kinds of overlap:** three lanes side by side. Lane 1 "redundancy": 5 identical document icons collapse into 1, with a counter "memorization ↓". Lane 2 "contamination": a benchmark card labelled "Q17 + answer" slides from an eval box into the training pile, then the score gauge jumps from 60% to 67%; a detector highlights it and moves it back. Lane 3 "useful recurrence": five *different* documents all containing `for i in range(n):` stay in the pile with a green tick.

## The tokenizer quietly sets your budget {#tokenizer-budget}

All the budgets in this article are counted in tokens, and the tokenizer decides how many tokens a given piece of text costs. Modern tokenizers use **subwords**: common words are one token, rarer words are split into pieces ("tokenization" might become "token" + "ization"). SentencePiece is a widely used toolkit that learns these pieces directly from raw text without first splitting into words, which is why it handles languages without spaces between words. How BPE merges are learned is in **Text in, next token out: tokens, embeddings, and sampling**; for pretraining, three consequences matter.

**1. Freeze it before the main run.** Every row of the embedding table belongs to one token ID. Change the tokenizer after training starts and every learned vector points at the wrong piece of text, like swapping the keys on a keyboard someone already touch-types on. Normalization rules, byte fallback and special tokens (like the end-of-document token) are all part of that contract.

**2. Compression differs by language and domain.** Suppose (illustrative numbers) your tokenizer uses 1.3 tokens per English word but 2.6 per word in a language it saw less of while being built. Give each language a 50B-token allocation:

```text
 English:  50B tokens ÷ 1.3 tokens/word ≈ 38.5B words
 Lang X:   50B tokens ÷ 2.6 tokens/word ≈ 19.2B words
```

"Equal tokens" meant half the actual text for language X, and every prompt in X also burns twice the context window and twice the serving cost. That's why labs grow vocabularies: Meta reports Llama 3's 128K-token vocabulary yields up to 15% fewer tokens than Llama 2's tokenizer. Bigger vocabularies aren't free, though. The embedding table is V × d parameters: at d = 4,096, going from V = 32,000 to V = 128,000 grows it from about 131M to about 524M parameters, and the output layer has the same shape (unless the two are tied).

**3. Per-token loss isn't comparable across tokenizers.** **Perplexity** is e raised to the average per-token cross-entropy. A tokenizer with bigger tokens has fewer, harder predictions per sentence, so its per-token loss goes up even if the model is just as good at the *text*. To compare tokenizers, normalize to the text: **bits per byte** (total loss in bits divided by the number of raw bytes) or downstream task scores.

## Mixing sources: weights are sampling probabilities {#data-mixtures}

You rarely train on one source. A mixture might be web text, code, papers, books, math, and multilingual data, each with a **mixture weight**. The subtle part: a weight is the *probability of drawing the next training batch from that source*, not a share of the data you own. So a small, high-quality source with a big weight gets repeated.

```formula
Dᵢ = pᵢ · D        passesᵢ = Dᵢ / Uᵢ
```

`D` is the total training tokens, `pᵢ` is source i's mixture weight, `Dᵢ` is how many tokens the model reads from it, and `Uᵢ` is how many *unique* tokens that source has. `passesᵢ` is how many epochs it effectively gets.

Worked example. A 300B-token run with 50% web, 30% code, 20% technical writing:

| Source | Weight | Exposure Dᵢ | Unique Uᵢ | Passes |
|---|---|---|---|---|
| Web | 0.50 | 150B | 2,000B | 0.075 |
| Code | 0.30 | 90B | 200B | 0.45 |
| Technical | 0.20 | 60B | 15B | **4.0** |

The mixture looks balanced, but the technical corpus is seen four times while most of the web is never seen at all. Four passes turns out to be roughly the edge of "free" repetition (more on that in the last section), and memorization risk grows with every extra pass. So log unique tokens and passes per source, not just weights.

How do labs pick the weights? Mostly with **ablations**: small training runs that change one weight at a time and compare on a fixed eval suite. Meta says it ran extensive mixing experiments for Llama 3 and ended up with four times more code than Llama 2. A common trick is to change the mixture late in training, upweighting the highest-quality sources during the final stretch when the learning rate is low (this is often called annealing), though the right recipe is empirical and model-specific.

> 🎬 **Animation — mixture weights vs unique data:** three buckets labelled Web (huge), Code (medium), Technical (small), drawn to scale by unique tokens. A sampling wheel split 50/30/20 spins and drops tokens into a "training stream". Counters under each bucket show exposure climbing to 150B, 90B, 60B. The Technical bucket's pass counter ticks 1, 2, 3, 4 and turns orange; the Web bucket shows most of its area never highlighted ("0.075 passes").

## Counting compute: where 6·N·D comes from {#compute-6nd}

Now the bill. The workhorse estimate for dense transformers (every parameter used for every token) is:

```formula
C ≈ 6 · N · D
```

`C` is total training compute in FLOPs (floating-point operations), `N` is the parameter count, and `D` is the number of training tokens.

Why 6? Picture a single weight `w` in a matrix multiply. On the **forward pass**, each token multiplies its activation by `w` and adds it into a sum: 1 multiply + 1 add = **2 FLOPs per parameter per token**. On the **backward pass** you do two similar jobs: compute the gradient with respect to the input (to keep backpropagating) and the gradient with respect to `w` itself (to update it). That's roughly **4 FLOPs**. Total: 2 + 4 = 6 FLOPs per parameter per token.

```text
             forward          backward (≈ 2× forward)
 per param:  x·w  (2 FLOPs)   ∂L/∂x (2 FLOPs) + ∂L/∂w (2 FLOPs)
             ──────────────────────────────────────────────────
             total ≈ 6 FLOPs per parameter per token
```

That's also why inference is cheaper per token: generating only needs the forward pass, about **2·N FLOPs per token**.

What 6·N·D leaves out: attention's own T² work (which matters at long context; see **Making attention cheaper: GQA, FlashAttention, and long context**), recomputing activations to save memory, communication between GPUs, and all the time spent not computing. For a mixture-of-experts model, `N` should be the *active* parameters per token, not the total (see **Mixture of experts: more parameters, same compute per token**).

**From FLOPs to calendar time.** Divide by the throughput you actually *sustain*, not the peak on the spec sheet. Real runs typically sustain a fraction of peak because of memory traffic, communication and stalls; how that fraction (called MFU) is measured and improved is covered in **Distributed training: fitting a training run onto a cluster**, along with how the weights, gradients and optimizer state get split across GPUs and how a weeks-long run survives failures and restarts.

Worked example. N = 7B parameters, D = 140B tokens:

```text
 C = 6 × 7×10⁹ × 140×10⁹ = 5.88 × 10²¹ FLOPs

 assume the cluster sustains 1 × 10¹⁷ FLOP/s in aggregate (illustrative)
 time = 5.88×10²¹ / 1×10¹⁷ = 58,800 s ≈ 16.3 hours
```

Now scale the data up to where modern small models live. Llama 3 8B was trained on over 15T tokens:

```text
 C ≈ 6 × 8×10⁹ × 15×10¹² = 7.2 × 10²³ FLOPs
 at the same 1×10¹⁷ FLOP/s: 7.2×10⁶ s ≈ 83 days
```

Same-sized model, about 120× the compute, purely because of D. (That's the formula applied to published sizes, not Meta's reported compute.) Calendar time then grows further with evaluations, checkpoints, restarts and debugging.

> 🎬 **Animation — where the 6 comes from:** one weight cell in a matrix glows. Forward: an activation arrow passes through it with "×, +" and a counter "2". Backward: two arrows flow back, one to the previous layer ("∂L/∂x: 2") and one into the weight's gradient slot ("∂L/∂w: 2"); the counter becomes 6. Then zoom out: the single cell multiplies into a grid of N cells (counter "6N per token") and a conveyor belt of D tokens runs through it (counter "6·N·D"). Finish with a gauge showing 5.88×10²¹ FLOPs ÷ 10¹⁷ FLOP/s = 16.3 h.

## Scaling laws: Kaplan, Chinchilla, and the 20-tokens-per-parameter rule {#scaling-laws}

You have a compute budget C. Since C ≈ 6·N·D, you can spend it on a big model reading few tokens or a small model reading many. Which is best? **Scaling laws** answer this empirically: train many models at different sizes and token counts, fit a smooth curve that predicts held-out loss, and pick the allocation the curve says is lowest.

The analogy: you have a fixed number of study hours and must split them between "hire a smarter student" (N) and "let them read more books" (D). A genius who reads one pamphlet does badly; an average student who reads everything but can't retain it also does badly. The best split is somewhere in the middle.

Chinchilla (Hoffmann et al., 2022) fit this form:

```formula
L(N, D) = E + A / N^α + B / D^β
```

`L` is held-out loss in nats per token. `E` is the irreducible floor (the entropy of text itself: even a perfect model can't predict every next word). `A / N^α` is the penalty for having too few parameters; `B / D^β` is the penalty for seeing too few tokens. `α` and `β` say how fast each penalty shrinks. Their published fit (for their data and setup) was E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28. So 10× the parameters shrinks the model term by 10^0.34 ≈ 2.2×: a **power law**, meaning each constant *multiple* of resources buys a constant *fraction* of improvement, with diminishing returns in absolute terms.

**The Kaplan vs Chinchilla story.** Kaplan et al. (2020) found clean power laws and concluded that bigger models are so much more sample-efficient that, as compute grows, you should grow the model much faster than the data. In the Chinchilla paper's summary table, Kaplan's allocation is N_opt ∝ C^0.73 and D_opt ∝ C^0.27. That's why the GPT-3 era produced huge models trained on a few hundred billion tokens. Chinchilla redid the experiment and got roughly N_opt ∝ C^0.5 and D_opt ∝ C^0.5: **scale parameters and tokens equally**, doubling the tokens every time you double the model. A big reason for the difference: Kaplan used the same number of training tokens and learning-rate schedule for all their runs, while Chinchilla found that matching the schedule's length to each run's token count gives the best final loss. A cosine schedule sized for a longer run hasn't finished decaying when you stop early, so short runs looked worse than they really were.

They proved it with a head-to-head. Gopher: 280B parameters on 300B tokens. Chinchilla: 70B parameters on 1.4T tokens. Check the budgets:

```text
 Gopher:     6 × 280×10⁹ × 300×10⁹  = 5.04 × 10²³ FLOPs     (~1 token/param)
 Chinchilla: 6 ×  70×10⁹ × 1.4×10¹² = 5.88 × 10²³ FLOPs     (20 tokens/param)
```

A similar budget, a model 4× smaller, and Chinchilla beat Gopher across the board (67.5% on MMLU, a 7-point gain), while being far cheaper to serve.

**The heuristic.** Chinchilla's estimated table works out to about 20 tokens per parameter (1B → 20.2B tokens, 10B → 205.1B, 67B → 1.5T). Combine D = 20·N with C = 6·N·D and you get a sizing formula you can do on a whiteboard:

```formula
C = 6 · N · (20 · N) = 120 · N²    ⟹    N_opt ≈ √(C / 120),   D_opt ≈ 20 · N_opt
```

Check it with our earlier budget, C = 5.88×10²¹: C/120 = 4.9×10¹⁹, √ = 7×10⁹. So N ≈ 7B, D ≈ 140B, exactly the example above. Quadruple the compute and both N and D double (14B, 280B), because each scales with √C.

Two caveats that separate a senior answer from a memorized one. First, the optimum is a **flat basin**. Plugging the published fit into a fixed budget of 5.88×10²¹ FLOPs:

| N | D | tokens/param | predicted loss |
|---|---|---|---|
| 1B | 980B | 980 | 2.224 |
| 2B | 490B | 245 | 2.189 |
| 3.5B | 280B | 80 | 2.177 |
| 7B | 140B | 20 | 2.184 |
| 14B | 70B | 5 | 2.212 |
| 28B | 35B | 1.25 | 2.262 |

Everything from 2B to 7B lands within about 0.01 nats. Second, notice that this particular fit's minimum sits near 3.5–4B (about 60 tokens/param), not at 20: Chinchilla's three fitting methods gave slightly different exponents (0.46 to 0.50 for parameters). The "20" is a compute-optimal *heuristic* for their data and recipe, not a constant of nature. Better data, a different tokenizer, or a different architecture moves it.

> 🎬 **Animation — the IsoFLOP valley:** x-axis model size N (log scale, 1B to 28B), y-axis predicted loss. Draw the six points from the table as a U-shaped curve with the label "fixed budget 5.88×10²¹ FLOPs". Step 1: a slider moves right along the curve; a paired bar underneath shows D shrinking as N grows (980B → 35B). Step 2: shade the flat bottom (2B–7B, within 0.01 nats) as "roughly equivalent". Step 3: add a second, lower U for 4× the budget whose minimum sits at 2× the N, with an arrow labelled "√C scaling: 4× compute → 2× N and 2× D".

## Training past "optimal": paying up front for cheap inference {#overtraining}

Chinchilla-optimal answers one narrow question: *what's the lowest loss for this training budget?* But a model you ship also gets **served**, and serving costs roughly 2·N FLOPs for every token processed, forever. A smaller model trained on more tokens can reach the same quality with a smaller N, which is cheaper on every request. That's called **over-training** (training well past the ~20 tokens/param point), and it's now standard practice.

The analogy: buying a car. A fuel-efficient model costs more up front, and only pays off if you drive enough miles. Meta's Llama 3 post is a clean real example: they note the Chinchilla-optimal budget for an 8B model is about 200B tokens, yet trained on over 15T tokens (about 1,875 tokens per parameter) and saw both the 8B and 70B models keep improving log-linearly. Sardana et al. ("Beyond Chinchilla-Optimal") put inference into the scaling-law optimization directly and concluded that anyone expecting reasonably large inference demand (on the order of a billion requests) should train smaller and longer than Chinchilla-optimal. They also found quality keeps improving at extreme ratios up to 10,000 tokens per parameter.

Worked break-even (hypothetical candidates that pass the **same** quality bar):

```text
 Model A: N = 10B, D = 200B   train C = 6 × 10¹⁰ × 2×10¹¹ = 1.2 × 10²² FLOPs
 Model B: N =  5B, D = 800B   train C = 6 × 5×10⁹ × 8×10¹¹ = 2.4 × 10²² FLOPs

 inference per token:  A = 2 × 10¹⁰ FLOPs    B = 1 × 10¹⁰ FLOPs
 B's extra training:   1.2 × 10²² FLOPs
 B's saving per token: 1 × 10¹⁰ FLOPs
```

```formula
Q_break-even = (C_train,B − C_train,A) / (c_infer,A − c_infer,B)
```

`Q` is the number of lifetime tokens served, `C_train` is each model's training compute, and `c_infer` is each model's per-token inference compute. Here Q = 1.2×10²² / 10¹⁰ = **1.2×10¹² tokens**. If your product serves 1M requests a day at 1,000 tokens each (10⁹ tokens/day), you break even after 1,200 days, about 3.3 years, and A wins. At 10M requests/day it's 120 days, and B wins easily. Frontier labs serve far more than that, which is why their small models are so heavily over-trained.

Be honest about what this calculation ignores. FLOPs aren't dollars. Serving is often limited by memory bandwidth and KV-cache memory rather than raw FLOPs, and prompt processing and token generation stress hardware differently (see **What happens at inference: prefill, decode, and the KV cache**). Smaller models also fit on fewer GPUs and give lower latency, which can matter more than cost. And the whole comparison assumes the two candidates actually reach equal quality, which is the part you must measure, not assume. Treat demand as a range and check that the decision holds at both ends.

> 🎬 **Animation — the break-even line:** x-axis lifetime tokens served (0 to 3×10¹²), y-axis cumulative total FLOPs. Model A's line starts at 1.2×10²² and rises with slope 2×10¹⁰; Model B's starts at 2.4×10²² with half the slope. Animate a dot moving along x; the lines cross at 1.2×10¹² tokens, where a label pops up "break-even". Shade left of the crossing "A cheaper" and right "B cheaper". Add two tick marks: "1M req/day → 3.3 years" and "10M req/day → 120 days".

## What the fits can't promise {#limits-of-fits}

Scaling laws are among the most useful tools in ML planning, and they're still curve fits. A senior engineer uses them the way you'd use a load test: a strong prior that you validate, not a guarantee.

**They predict loss, not abilities.** A curve tells you average surprise on a held-out distribution. It doesn't promise reliable tool use, calibrated uncertainty, good behaviour on a small but critical slice (say, medical dosing questions), or knowledge of anything after the data cutoff. An average hides the tails. Meta mentions building separate scaling laws for downstream benchmark scores to predict the largest models' task performance before training them; that's the right instinct, keep loss curves and task gates both.

**They hold for the recipe they were fit on.** Change the data mixture, the tokenizer, the architecture, the context length or the learning-rate schedule, and the coefficients move. A fit from a different tokenizer isn't even in the same units (remember, per-token loss isn't comparable). That's why teams refit at small scale whenever the recipe changes.

**Repeated data is worth less than fresh data.** The frontier assumes every token is new. When you run out of unique data, you repeat it. Muennighoff et al. found that up to about 4 epochs of repeated data gives a negligible change in loss compared with unique data, but with more repetition the value of extra compute decays toward zero. So a 300B-token budget on 100B unique tokens (3 epochs) is probably fine; 30 epochs of a small domain is not. Track unique tokens and total exposures separately, always.

```text
 value of the next token
   ▲
   │ ████████████████  fresh data
   │ ███████████████   epochs 1–4: nearly as good as fresh
   │ ████████
   │ ███                 more epochs: diminishing, → 0
   │ █
   └──────────────────────────────────────────► passes over the same data
```

**How to make a scaling recommendation defensible:**

1. Run a ladder of small models (say 5–6 sizes spanning 10–100×) at several token counts, each with a learning-rate schedule sized to its own run length.
2. Fit on most of them and **hold some runs out** to check the fit's prediction error, like a validation set for the curve itself.
3. Measure real throughput for the candidate shapes; an awkward shape can waste more wall-clock than its FLOPs suggest.
4. Present a *range* of candidate (N, D) pairs, sensitivity to inference demand and data quality, a pilot that could prove you wrong, and a stopping rule.

That turns "the curve says 7B" into a decision someone can audit.

> 🎬 **Animation — trusting the extrapolation:** a log-log plot of compute vs loss. Six small-run points appear one by one; a straight fitted line is drawn through four of them. Step 1: the two held-out points appear close to the line, with small error bars ("validated"). Step 2: the line extends far to the right toward a star labelled "target run", with a widening cone of uncertainty. Step 3: a toggle "tokenizer changed" shifts all the points and the line snaps to a new position, with the caption "new recipe → refit".

# Interview

## Question

You have a fixed compute budget of about 6×10²³ FLOPs and roughly 1.5T unique tokens after filtering. A colleague proposes training the largest model that fits in memory for as long as the budget allows, repeating data if needed. The model will serve very heavy traffic. How do you decide the model size and token count?

## Answer

I'd start by pinning down what "tokens" means. Is 1.5T counted with the tokenizer we'll actually ship, and after dedup and decontamination? How is it split by source, and what does each source's repetition look like under the mixture we want? Then I'd set the quality bar we care about: task evals, clean of contamination, plus the latency and memory limits for serving.

For a first anchor, the Chinchilla heuristic: C ≈ 6·N·D with D ≈ 20·N gives C = 120·N², so N ≈ √(6×10²³ / 120) ≈ √(5×10²¹) ≈ 71B and D ≈ 1.4T. That's essentially Chinchilla itself, and it would fit our 1.5T unique tokens in about one epoch. The "largest model that fits" proposal lands on the wrong side of the curve: at a fixed budget, a much bigger N means far fewer tokens, and Gopher vs Chinchilla showed that loses.

But Chinchilla-optimal minimizes training compute only. With heavy inference, I'd seriously consider a smaller model trained longer. For example, a 20B model at the same budget reads 6×10²³ / (6 × 2×10¹⁰) = 5T tokens, which is over three epochs of our unique data. That's inside the roughly 4-epoch range where repetition costs little, but it's close enough that I'd watch it. Serving it saves about 2 × (71B − 20B) ≈ 10¹¹ FLOPs per token. If it falls a little short of the 71B's quality at this budget, I'd compute the break-even: the extra training needed to close the gap, divided by that per-token saving, checked at the low and high ends of the demand forecast. Then translate FLOPs into real serving cost with measured throughput, since decode is usually memory-bandwidth-bound.

Before committing, I'd run a ladder of small pilots with schedules sized to each run, fit the curve on our own data and tokenizer, validate it on held-out runs, and compare the candidates on the same clean evals. The recommendation is the cheapest lifetime option that clears the quality gate, not the biggest model or the lowest pretraining loss. How the chosen model is then laid out across the cluster and checkpointed is a separate plan.

## Follow-ups

- How would your choice change if inference demand turned out to be 10× lower than forecast?
- You only have 400B unique tokens but the compute-optimal plan wants 1.4T. What are your options, and what does the data-constrained scaling work say about each?
- Why might switching to a new tokenizer invalidate your existing scaling fit?
- Which experiment would tell you whether a weak result came from a bad data mixture rather than too little model capacity?
- How would you detect and report benchmark contamination in your final corpus?

# Pitfalls

- Treating raw crawl size, unique tokens after filtering, and training tokens (exposures, including repeats) as the same number.
- Subtracting pipeline retention percentages instead of multiplying them.
- Assuming deduplication removes benchmark contamination; a leaked answer can appear exactly once.
- Comparing per-token perplexity across models with different tokenizers instead of normalizing to bits per byte or tasks.
- Quoting "20 tokens per parameter" as a law of nature rather than a compute-optimal heuristic tied to one data setup and recipe, and one that ignores inference cost.
- Converting 6·N·D directly into days or dollars using peak hardware specs instead of measured sustained throughput.
- Reading a mixture weight as "share of the data" and missing that a small source with a large weight is being repeated many times.
- Treating lower pretraining loss as proof of better downstream abilities.

# Checklist

- Sketch a pretraining data pipeline and say what each stage removes and why.
- Compute retained tokens through a chain of filter rates.
- Distinguish corpus redundancy, contamination, and useful recurrence, and name a fix for each.
- Compute per-source exposure and number of passes from mixture weights and unique counts.
- Derive the 6 in C ≈ 6·N·D and turn FLOPs into wall-clock time under a stated throughput.
- Size a compute-optimal run with N ≈ √(C/120) and D ≈ 20·N, and explain why it's a heuristic.
- Explain why Kaplan and Chinchilla disagreed and what the Gopher vs Chinchilla comparison showed.
- Compute an inference break-even for a smaller, over-trained model and list what the FLOP-only calculation ignores.
- State what scaling fits can't promise and how to validate one before a big run.

# Sources

- [Hoffmann et al. — Training Compute-Optimal Large Language Models (Chinchilla), 2022](https://arxiv.org/abs/2203.15556) — Parametric loss fit and constants, 6ND approximation, equal scaling of N and D, Kaplan comparison (0.73/0.27), learning-rate schedule explanation, Table 3 token estimates, Gopher vs Chinchilla sizes and MMLU result.
- [Kaplan et al. — Scaling Laws for Neural Language Models, 2020](https://arxiv.org/abs/2001.08361) — Original power-law fits for loss vs model size, data and compute, and the conclusion that compute-efficient training favours very large models on modest data.
- [Sardana et al. — Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — Inference-aware scaling; train smaller and longer given large inference demand (~1B requests); gains up to 10,000 tokens per parameter.
- [Muennighoff et al. — Scaling Data-Constrained Language Models, 2023](https://arxiv.org/abs/2305.16264) — Up to ~4 epochs of repeated data is nearly as good as unique data; value of further repetition decays toward zero.
- [Lee et al. — Deduplicating Training Data Makes Language Models Better, 2021](https://arxiv.org/abs/2107.06499) — 61-word sentence repeated 60,000+ times in C4; 10× less memorized output after dedup; over 4% train-test overlap in standard validation sets.
- [Penedo et al. — The RefinedWeb Dataset for Falcon LLM, 2023](https://arxiv.org/abs/2306.01116) — Filtered and deduplicated web data alone (5T tokens from Common Crawl) can outperform curated corpora like The Pile.
- [Kudo and Richardson — SentencePiece, 2018](https://arxiv.org/abs/1808.06226) — Language-independent subword tokenizer trained directly from raw text.
- [Meta — Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/) — 15T+ training tokens, ~200B Chinchilla-optimal for 8B, continued log-linear gains, Llama 2-generated data for quality classifiers, 128K vocabulary with up to 15% fewer tokens, data-mix experiments and downstream scaling laws.

# Flashcards

## base-model

**Q:** What does pretraining produce, and why isn't it an assistant yet?

A base model: a network trained only to predict the next token over a huge corpus. It has absorbed facts, style and reasoning patterns, but it continues text rather than following instructions, so a question might be answered or might be followed by more questions, because both look like web text. Post-training (SFT, preference tuning, RL) turns it into an assistant.

## exposure-diversity

**Q:** Why is "trained on one trillion tokens" not a measure of one trillion tokens of novel information?

Training tokens count exposures: every position the model trained on, repeats included. Duplicated documents, oversampled sources and multiple epochs all inflate the count while the unique text stays fixed. Track unique tokens and total exposures separately.

## retention-math

**Q:** A 1,000B-token crawl goes through stages keeping 80%, then 75%, then 70%. How much remains?

420B tokens: 1,000 × 0.8 × 0.75 × 0.7. Each rate applies to the previous stage's output, so rates multiply. Subtracting 20 + 25 + 30 points from the original (giving 250B) is the common mistake.

## near-dup-minhash

**Q:** How do you find near-duplicate documents at web scale, and what similarity is being measured?

Split each document into overlapping word n-grams (shingles) and estimate the Jaccard similarity of the shingle sets (shared ÷ total distinct) with compact MinHash fingerprints, grouping candidates so you never compare all pairs. Pairs above a threshold are clustered and one copy kept. Exact hashing misses these because a single changed timestamp changes the hash.

## dedup-leakage

**Q:** Why doesn't deduplicating the training corpus prove an evaluation is uncontaminated?

A benchmark question or solution may appear only once in training, or as a paraphrase or translation, so there's nothing to deduplicate. Contamination needs a separate comparison against protected eval sets, plus evals built after the data cutoff, and results reported separately on clean and overlapping subsets.

## tokenizer-perplexity

**Q:** Why can't perplexity be compared naively across tokenizers?

Perplexity averages loss per token, and different tokenizers cut the same text into different numbers of tokens of different difficulty. A tokenizer with bigger tokens has fewer, harder predictions, so per-token loss rises even if the model models the text equally well. Normalize to the text (bits per byte) or compare on downstream tasks.

## mixture-passes

**Q:** A 300B-token run gives 20% weight to a source with 15B unique tokens. How much repetition is that?

60B exposures (0.2 × 300B), which is 60 ÷ 15 = 4 passes. Mixture weights are sampling probabilities, not data shares, so a small source with a big weight gets repeated; four passes is roughly where repetition stops being nearly free.

## compute-estimate

**Q:** What does C ≈ 6·N·D estimate, where does the 6 come from, and what does it leave out?

Total training FLOPs for a dense model with N parameters trained on D tokens. Forward is ~2 FLOPs per parameter per token (multiply + add); backward is ~4 (gradients for inputs and for weights). It omits attention's T² work, activation recomputation, communication and idle time, and for MoE N should be active parameters. Inference is ~2·N per token.

## kaplan-vs-chinchilla

**Q:** Why did Kaplan (2020) and Chinchilla (2022) recommend different allocations, and who was right in practice?

Kaplan's allocation grew the model much faster than data (N ∝ C^0.73, D ∝ C^0.27). Chinchilla found roughly equal scaling (both ∝ C^0.5), partly because Kaplan used one token count and learning-rate schedule for all runs, while matching the schedule to each run's length gives better final loss. Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at a similar budget.

## balanced-scaling

**Q:** Using D ≈ 20·N, size a compute-optimal run for C = 5.88×10²¹ FLOPs. What happens if compute quadruples?

C = 6·N·20N = 120·N², so N = √(5.88×10²¹ / 120) = 7B and D = 140B. Both scale with √C, so 4× compute gives 2× each: 14B parameters on 280B tokens. This depends on the fitted frontier and isn't guaranteed for every recipe.

## heuristic-limit

**Q:** Why is "20 tokens per parameter" not enough to plan a training run?

It's a compute-optimal heuristic from one data setup and recipe; the loss basin is flat and even Chinchilla's own fitting methods disagree on the exact ratio. Data quality, repetition, tokenizer and architecture move it, and it ignores inference cost entirely, which is why production models are often trained far past it.

## inference-optimum

**Q:** When does training a smaller model on many more tokens make economic sense?

When it reaches the required quality and the serving savings (~2·ΔN FLOPs per token, plus lower memory and latency) over the model's lifetime outweigh the extra training compute. With heavy inference demand, this favours training smaller and longer than Chinchilla-optimal, as Llama 3 8B did with 15T tokens.

## break-even

**Q:** How do you compute the inference break-even for a smaller model that cost more to train?

Q = (extra training FLOPs) ÷ (per-token inference saving). Example: 5B on 800B (2.4×10²² FLOPs) vs 10B on 200B (1.2×10²²); saving 10¹⁰ FLOPs/token; break-even at 1.2×10¹² served tokens. It assumes equal quality, and FLOPs aren't dollars, so confirm with measured serving costs.

## repeated-epochs

**Q:** How much can you repeat data before it stops helping?

Muennighoff et al. found up to about 4 epochs of repetition gives a negligible change in loss vs unique data, but beyond that the value of extra compute decays toward zero. So a few epochs is a fine way to stretch a limited corpus; many epochs mostly buys memorization.

## scaling-validation

**Q:** How do you make a scaling-law prediction defensible?

Fit it on your own data, tokenizer and recipe with a ladder of small runs at several sizes and token counts (schedules sized to each run), hold out some runs to check prediction error, measure real throughput, give a range of candidates with sensitivity to demand and data, and gate the final decision on clean downstream tasks, not loss alone.
