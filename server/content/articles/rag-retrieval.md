---
{
  "slug": "rag-retrieval",
  "title": "RAG: giving the model the right evidence",
  "category": "applications",
  "summary": "Retrieval-augmented generation turns a closed-book exam into an open-book one. Learn to build it as an evidence pipeline: ingest and chunk documents, search with embeddings and keywords, enforce permissions, pack the prompt, and then check whether the answer actually follows from the evidence.",
  "difficulty": "Advanced",
  "minutes": 32,
  "prerequisites": ["tokens-and-embeddings", "transformer-foundations"],
  "learningObjectives": [
    "Explain why RAG exists and name every stage where a RAG answer can go wrong.",
    "Design ingestion with structure-aware chunks, stable IDs, versioning, and deletion manifests.",
    "Calculate cosine similarity, raw vector-index memory, and reciprocal rank fusion scores by hand.",
    "Explain how HNSW trades exactness for speed, and what M, ef_construction, and ef control.",
    "Combine BM25, dense retrieval, and a cross-encoder reranker, and enforce permissions before any text leaves the search layer.",
    "Separate retrieval failures from generation failures with retrieval metrics and oracle-context tests.",
    "Defend a choice between RAG, fine-tuning, long context, or a plain database query."
  ]
}
---

# Sections

## Why a smart model still needs an open book {#why-rag}

A large language model (LLM) knows only what was in its training data, and that knowledge froze the day training finished. It has never read your company wiki, last night's pricing change, or the support ticket filed an hour ago. Ask it anyway and it does what it always does: it predicts plausible next tokens. Plausible isn't the same as true, so you get a confident answer that may be invented. People call that a *hallucination*.

**Retrieval-augmented generation (RAG)** fixes this by turning a closed-book exam into an open-book one. Before the model answers, a search system looks up passages that are likely relevant and pastes them into the prompt. The model still writes the answer, but now it can copy from the textbook instead of relying on memory.

The idea was named in Lewis et al. (2020). They paired a sequence-to-sequence generator (its "parametric" memory, the knowledge stored in its weights) with a dense vector index of Wikipedia searched by a neural retriever (its "non-parametric" memory, knowledge you can swap without retraining). Modern systems usually run those two halves as separate services, but the split is the same:

- The **retriever** decides *which evidence* the model sees.
- The **generator** (the LLM) turns that evidence into an answer.

```text
                 ┌──────────── offline (ingestion) ────────────┐
 source docs ──► parse ──► chunk ──► embed + index ──► search index
                                                          │
                 ┌──────────── online (per request) ──────┼──────┐
 user question ──► retrieve (+ permission filter) ◄───────┘      │
                     │                                           │
                     ▼                                           │
                 rerank ──► pack prompt ──► LLM ──► answer + citations
```

Here's the part interviewers care about: every arrow above is a place the answer can break. The source can be wrong or stale. The parser can mangle a table. The chunker can split a rule from its exception. Search can miss the right passage. The permission filter can leak a document the user shouldn't see. And the model can ignore perfectly good evidence. A senior answer about RAG is mostly about **knowing which of those broke**.

So before you pick a vector database, write down the **evidence contract**: what counts as a correct answer for this product. For example, a support bot might answer only from published docs, show the doc version, and refuse to talk about unreleased features. An internal HR assistant must also use only documents the asking employee is allowed to read. Under that contract, a fluent answer copied from last year's manual is *wrong*, even if its citation link works.

Then build a small labelled question set, say 100 to 200 questions, before tuning anything. Include ordinary questions, exact identifiers ("what does error E-4417 mean?"), vague wording, questions with no answer in the corpus, questions where two documents disagree, and questions that should be blocked by permissions. For each one, record which passages would justify the answer. Without this set, every later knob (chunk size, embedding model, top-k, prompt) becomes a competing guess.

> 🎬 **Animation — closed book vs open book:** split screen. Left: a user asks "What's our parental leave policy?" and the LLM alone answers "16 weeks" with a red "invented" stamp. Right: the same question flows into a search box (step 1), three passage cards slide out of a document stack (step 2), they drop into the prompt next to the question (step 3), and the LLM answers "20 weeks [HR-Policy v7 §3.2]" with a green "supported" stamp (step 4). Finally, highlight each arrow of the pipeline in turn with a small red ✗ labelled with its failure mode: stale source, bad parse, bad chunk, missed retrieval, leaked permission, ignored evidence.

## Getting documents in without breaking them {#ingestion-chunking}

**Ingestion** is the offline job that turns raw files (PDFs, HTML, wiki pages, tickets) into searchable pieces. Most RAG quality problems are born here, and they're invisible if you only look at the final answer.

**Parse structure first.** Pull out headings, lists, tables, and page boundaries *before* you split text. A table cell reading "30" is useless on its own. "Refund window (days): 30" is evidence. Keep the original text next to any cleaned-up version, plus a locator (URL, page number, section path) so a citation can take someone to the exact spot. Be careful with deduplication: dropping repeated navigation menus helps, but deleting a warning because it appears on every page can remove the only restriction attached to a procedure.

**Chunking** means splitting each document into smaller passages, called **chunks**. The chunk is the unit that search finds and that gets pasted into the prompt. Think of cutting a textbook into index cards. If the cards are too small, one card says "Refunds are allowed within 30 days" and the next says "except for customised orders", so you can retrieve the rule without its exception. If they're too big, one card covers refunds, shipping, and warranties. Its embedding (the search vector, next section) becomes a blurry average of three topics, and it eats prompt tokens.

```text
 too small                    about right                   too big
 ┌──────────────┐            ┌────────────────────┐        ┌───────────────────┐
 │ Refunds OK   │            │ §4 Refunds         │        │ §4 Refunds …      │
 │ within 30d.  │            │ Allowed within 30d │        │ §5 Shipping …     │
 └──────────────┘            │ except custom      │        │ §6 Warranty …     │
 ┌──────────────┐            │ orders.            │        │ (one fuzzy vector │
 │ …except      │            └────────────────────┘        │  for 3 topics)    │
 │ custom orders│                                          └───────────────────┘
 └──────────────┘
 rule and exception           one idea, one card           diluted meaning,
 can be split apart                                        wasted tokens
```

A good default is **structure-aware splitting**: cut at section and paragraph boundaries, cap the size (a few hundred tokens is a common starting point, but that's a starting point, not a law), and add a little **overlap** so a sentence on a boundary shows up in both neighbours. A quick check on the numbers: a 10,000-token document with 500-token chunks and 50 tokens of overlap has a stride of 450 tokens. That gives ⌈(10,000 − 500) / 450⌉ + 1 = 22 + 1 = **23 chunks**, and about 10% of the tokens are stored twice.

A popular pattern is **small-to-big** (also called parent expansion): search over small, precise child chunks, then hand the model the larger parent section they came from. You get sharp matching and full context. Just remember that the expansion must still respect permissions and the token budget. There's no universally right chunk size. It depends on how your documents are structured and what people ask, so test questions whose answers cross a chunk boundary.

**Give everything an identity.** Each document gets a stable ID. Each chunk gets an ID that includes the document version. Useful metadata includes tenant (the customer or organisation that owns the data), source URI, section path, content hash, effective date, ingestion time, parser version, embedding model version, and permission attributes. These fields are what make citations, debugging, incremental updates, and deletion possible.

Finally, keep a **manifest**: a packing list mapping each document version to all of its chunks. Here's the bug it prevents. Version 1 of a policy produced chunks 1–12. Version 2 is shorter and produces chunks 1–9. If your update just upserts (inserts or overwrites) chunks by ID, chunks 10–12 from the old version stay searchable forever, and the bot happily quotes a deleted clause. With a manifest, the update is "delete everything listed for v1, publish v2's list", ideally as an atomic swap.

> 🎬 **Animation — the orphan chunk bug:** a document card "Policy v1" splits into 12 numbered chunk tiles that fly into an index grid (step 1). "Policy v2" arrives, shorter, and splits into 9 tiles that overwrite tiles 1–9 (step 2). Tiles 10–12 stay, now glowing red with the label "orphans, still searchable" (step 3). Rewind and replay with a manifest clipboard: the clipboard lists v1 → {1…12}, all 12 are removed, then v2's 9 are inserted, and the grid ends clean (step 4).

## Embeddings: turning meaning into coordinates {#embeddings-similarity}

To search by meaning, we turn text into numbers. An **embedding model** reads a passage and outputs a fixed-length list of numbers, a **vector** (often a few hundred to a few thousand numbers long). The model is trained so that texts with similar meaning land near each other. Think of it as a map: every passage gets a location, "how do I reset my password" and "forgot login credentials" end up in the same neighbourhood, and search becomes "find the points closest to the question's point". (Token embeddings inside an LLM are a related idea. We cover vectors and similarity intuition in **Text in, next token out: tokens, embeddings, and sampling**.)

The usual way to measure "close" is **cosine similarity**, which asks how closely two vectors point in the same direction:

```formula
cosine(q, d) = (q · d) / (‖q‖ × ‖d‖)
```

Here `q` is the query vector and `d` is a document vector. `q · d` is the dot product (multiply matching entries and add them up). `‖q‖` is the vector's length, √(sum of squares). The result runs from −1 (opposite) through 0 (unrelated) to 1 (same direction). If every vector is normalised to length 1, cosine is just the dot product, which is why many systems normalise once and then use inner product.

A tiny 2-D example (real embeddings have hundreds of dimensions, but the arithmetic is identical):

| Vector | Values | q · d | ‖q‖·‖d‖ | cosine |
|---|---|---|---|---|
| query q | (3, 4) | — | — | — |
| d₁ | (6, 8) | 18 + 32 = 50 | 5 × 10 = 50 | **1.00** |
| d₂ | (4, 3) | 12 + 12 = 24 | 5 × 5 = 25 | **0.96** |
| d₃ | (4, −3) | 12 − 12 = 0 | 5 × 5 = 25 | **0.00** |

Notice that d₁ is twice as long as q but still scores a perfect 1.00. Cosine ignores length and only cares about direction.

**Bi-encoders.** The retrieval embedding model is usually a *bi-encoder*, also called a dual encoder (Karpukhin et al.'s Dense Passage Retrieval is the classic example). The question and the passage are encoded *separately*. That's the key property: you can embed a million passages once, offline, and at query time you only embed the short question. The price is that the question and passage never "see" each other token by token, so fine distinctions get lost. We'll come back to that with rerankers.

Two traps interviewers love:

1. **Same length doesn't mean same space.** Two models that both output 768 numbers put meaning in completely different places. Matching dimensions only makes the arithmetic *possible*. Query and documents must be embedded by a compatible model with the same preprocessing (some models even want a prefix such as "query:" versus "passage:"). Switching embedding models means re-embedding the whole corpus into a *new* index, evaluating it, and then flipping traffic over, so store the model version with every vector.
2. **Similar isn't the same as relevant.** "How do I cancel my subscription?" and "How do I *not* cancel my subscription?" can be very close in embedding space. Nearest in vector space is a proxy for relevance, not a guarantee.

**Sizing.** Raw vector storage is simply count × dimensions × bytes per number:

```formula
raw bytes = N_vectors × dims × bytes_per_value
```

For 1,000,000 chunks at 768 dimensions in float32 (4 bytes each), that's 1,000,000 × 768 × 4 = 3,072,000,000 bytes ≈ **3.07 GB**, before graph links, metadata, replicas, and allocator overhead. Storing each value as one byte (int8) would cut it to ≈ 0.77 GB at some cost in accuracy. We cover how that works in **Quantization: spending fewer bits per weight**.

> 🎬 **Animation — cosine on a 2-D map:** draw axes with the query arrow q = (3, 4) in blue. Step 1: add d₁ = (6, 8) along the same line but twice as long, with the angle arc reading 0° and "cos = 1.00". Step 2: add d₂ = (4, 3), with an arc of about 16° and "cos = 0.96". Step 3: add d₃ = (4, −3) at a right angle, with "cos = 0.00". Step 4: scatter 30 small grey passage dots, draw a cone around q, and highlight the 3 dots inside it as "top-3 results".

## Finding the nearest neighbours fast: ANN and HNSW {#ann-hnsw}

**Exact search** (also called brute-force or flat search) compares the query with every vector. At 1M vectors × 768 dimensions, that's 768 million multiply-adds per query. A GPU or a well-vectorised CPU can do it, and for small corpora (tens of thousands of chunks) exact search is often the right answer. But cost grows linearly with the corpus, and filters, replicas, and high query rates add up.

**Approximate nearest neighbour (ANN)** search gives up a little accuracy for a lot of speed. The most common ANN index in RAG stacks is **HNSW, Hierarchical Navigable Small World graphs** (Malkov & Yashunin). Each vector is a node, linked to some of its near neighbours. On top of that, a random, exponentially thinning subset of nodes is also placed in higher layers, which have fewer nodes and longer links.

The analogy is driving to a new address. You take the motorway to the right part of the country (top layer), then main roads to the right town (middle), then local streets to the exact house (bottom layer). At each layer, the search greedily hops to whichever neighbour is closer to the query, then drops down a layer.

```text
 layer 2   A ─────────────────────── F             (few nodes, long links)
           │                         │
 layer 1   A ────── C ────── E ───── F             (more nodes)
           │        │        │       │
 layer 0   A ─ B ─ C ─ D ─ E ─ G ─ F ─ H           (every node, short links)

 search for query ★ (nearest to G):
   L2: start A → F is closer            → drop at F
   L1: F → E is closer, neighbours no better → drop at E
   L0: explore around E, keeping the ef best candidates → returns G, then E, F
```

The hnswlib library's documentation names the three knobs:

| Knob | What it controls | Turn it up and… |
|---|---|---|
| `M` | links created per node during construction (range 2–100 per the docs) | better recall on hard, high-dimensional data, more memory (roughly M × 8–10 bytes per element for links), slower builds |
| `ef_construction` | how hard the builder searches for good links | better graph quality, longer build time |
| `ef` | size of the candidate list kept during a query (must be ≥ k) | higher recall, slower queries |

Memory check with M = 16: about 16 × 8–10 = 128–160 bytes of links per vector, so roughly 0.13–0.16 GB for 1M vectors. That's small next to the 3.07 GB of raw float32 vectors.

Now a crucial distinction. **ANN recall** asks: of the *true* nearest neighbours in vector space, how many did the approximate search return? It compares HNSW against exact search. It says *nothing* about whether those neighbours actually answer the question. That's **relevance recall**, which we measure against human labels in the evaluation section. You can have 99% ANN recall and still retrieve the wrong evidence because the embedding model ranks irrelevant text highly. Measure both, separately.

These are empirical tradeoffs, not guarantees. Benchmark on your own data, including with filters on (next sections), and look at tail latency (p95/p99), not just averages.

> 🎬 **Animation — one HNSW search:** a 3-layer graph with ~20 nodes on layer 0, 7 on layer 1, and 2 on layer 2. A gold star (the query) appears. Step 1: on the top layer, a cursor starts at the entry node and hops once toward the star, then stops because no neighbour is closer. Step 2: the cursor drops through a dashed vertical line to the same node on layer 1 and hops twice. Step 3: on layer 0, a highlighted "ef = 4" candidate list box fills and updates as neighbours are checked. Step 4: the final top-3 light up. Then show exact search as a contrast: a sweeping beam checks all 20 nodes, and a counter compares "nodes visited: 9 vs 20".

## Two searches beat one: hybrid retrieval and reranking {#hybrid-reranking}

Embedding search is called **dense retrieval** because every passage becomes a dense vector of numbers. It's great at paraphrase: "forgot login" matches "reset password". It's weak at exact strings. An error code like `E-4417`, a part number, or a rare surname might tokenise into fragments that the embedding treats as noise.

**Lexical retrieval** is classic keyword search, and the standard scoring function is **BM25** (Elasticsearch's default similarity). For each query word that appears in a document, BM25 adds up a score that rewards two things: *rare* words (the IDF term, "inverse document frequency") and more occurrences of the word, *with diminishing returns*. It also penalises long documents a little.

```formula
BM25(D, Q) = Σ over query terms t:  IDF(t) · f(t,D)·(k₁ + 1) / ( f(t,D) + k₁·(1 − b + b·|D| / avgdl) )
```

`f(t,D)` is how many times term t appears in document D. `|D|` is D's length and `avgdl` is the average document length. `k₁` controls how fast repeated occurrences stop helping, and `b` controls the length penalty (Elasticsearch defaults: k₁ = 1.2, b = 0.75). For a document of exactly average length, the fraction becomes f·2.2 / (f + 1.2):

| occurrences f | term-frequency factor |
|---|---|
| 1 | 2.2 / 2.2 = 1.00 |
| 3 | 6.6 / 4.2 ≈ 1.57 |
| 10 | 22 / 11.2 ≈ 1.96 |
| ∞ | → 2.2 (the ceiling) |

So repeating a keyword 10 times buys you less than twice the credit of saying it once. That saturation is why keyword stuffing doesn't break BM25.

| | Dense (embeddings) | Lexical (BM25) |
|---|---|---|
| Wins on | paraphrases, synonyms, fuzzy questions | exact IDs, codes, names, rare terms |
| Loses on | exact strings, negation, new jargon | different wording for the same idea |
| Needs | embedding model, vector index | inverted index (word → documents) |

**Hybrid retrieval** runs both and merges the results. But you can't just add the scores: a BM25 score of 14.2 and a cosine of 0.83 are on unrelated scales. **Reciprocal rank fusion (RRF)** sidesteps this by using only *positions*:

```formula
RRF(d) = Σ over result lists containing d:  1 / (k + rank(d))
```

`rank(d)` starts at 1 for the top result. `k` is a smoothing constant (Elasticsearch's default is 60) that controls how much rank 1 beats rank 5. A document missing from a list gets nothing from that list.

Worked example with k = 60:

| Doc | dense rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | 10 | 1/61 + 1/70 ≈ 0.01639 + 0.01429 = **0.03068** |
| B | 3 | 3 | 1/63 + 1/63 = 2/63 ≈ **0.03175** |
| C | 2 | absent | 1/62 ≈ **0.01613** |

B wins because both retrievers agree it's good. A was the top dense hit, but BM25 barely liked it. C only appeared in one list, so it scores roughly half. These are ranking scores, **not** probabilities of relevance. Don't threshold on them as if they meant "83% confident". Also note that the candidate window size (how deep you take each list) changes which documents get two contributions, so tune it too.

**Reranking.** After fusion, you might have 50–100 candidates. A **reranker** takes a closer look at each one. It's usually a **cross-encoder**: a transformer that reads the question and the passage *together* as one input, so every question token can attend to every passage token, and it outputs a relevance score. Nogueira & Cho (2019) showed that a BERT cross-encoder reranking passages beat the previous state of the art on MS MARCO by 27% relative in MRR@10. This is why the pattern stuck.

The analogy: retrieval is a recruiter skimming 10,000 CVs in seconds with keyword and vibe matching. Reranking is the proper interview, which you only have time to do for the shortlist. Cross-encoders are far more accurate per pair, but they can't precompute anything, so you run one forward pass per (question, passage) pair at query time. That's why it's bounded to a small pool.

```text
  authorised corpus (1M chunks)
     │                         │
  BM25 top-50              dense top-50
     └──────────┬──────────────┘
           RRF fuse → ~80 unique
                │
       cross-encoder rerank → top-8
                │
          pack into prompt
```

The iron rule is that **a reranker can only reorder what it's given.** If the right passage isn't in the candidate pool, no reranker can rescue it. Keep three numbers separate and tune them one at a time: how many you *retrieve*, how many you *rerank*, and how many you *put in the prompt*. Bumping all three at once multiplies cost without telling you what helped.

> 🎬 **Animation — retrieve, fuse, rerank:** two columns of ranked cards, "Dense" and "BM25", each 10 deep. Step 1: doc B glows at rank 3 in both; doc A is rank 1 in dense and rank 10 in BM25. Step 2: the cards slide into a merge funnel, and each card's RRF sum is written beside it (A 0.03068, B 0.03175, C 0.01613), after which the list re-sorts so B is on top. Step 3: the top 8 enter a "cross-encoder" box, where the question and passage are shown side by side with attention lines crossing between them, and the scores reshuffle again. Step 4: a card that was never in either list stays greyed out outside the funnel, labelled "can't be rescued by reranking".

## Permissions and freshness are part of retrieval {#permissions-freshness}

Here's where RAG systems cause real incidents. Suppose an intern asks the internal assistant "what's the CEO's salary?" and the compensation spreadsheet is in the index. If the only protection is a line in the system prompt saying "don't reveal documents the user can't access", you've already lost. The text reached the model. It may also have reached an external reranker API, a log line, a trace, or a cache. **Access control is a property of retrieval, not of the prompt.**

The rules:

- Resolve who the user is and which groups they belong to through trusted infrastructure (your identity provider), never from anything the user or model typed.
- Apply tenant and permission filters on **every** retrieval path: BM25, vector search, parent expansion, and any fallback tool. One unfiltered path is a leak.
- Enforce the filter **before** any text leaves the search layer. The index internals may walk graph nodes the user can't see, and that's fine, as long as their text is never returned.

Search products implement this in different ways. Azure AI Search, for example, documents both simple "security filters" (your app passes the user's group IDs as a filter string) and preview features that check the caller's identity token against permission metadata synced into the index. Notice from those docs that permission changes appear in results only *after* the metadata is re-synced. Check your backend's exact semantics. We go deeper on the threat model, including prompt injection through retrieved documents, in **LLM security: treating model output as untrusted**.

**Filtering changes search quality.** Naive post-filtering goes like this: fetch the 10 nearest chunks globally, then drop the ones the user can't see. If 9 of the 10 belong to other tenants, you're left with 1 result, even though dozens of good authorised chunks sat just outside the top 10.

```text
 post-filter (bad)                       filter-aware (good)
 top-10 global: ✗✗✓✗✗✗✗✗✗✗              search only inside the user's
 after filter:  ✓            (1 left)    allowed set: ✓✓✓✓✓✓✓✓✓✓ (10)
```

Fixes include filter-aware ANN (the index respects the filter while traversing), per-tenant indexes or partitions, and controlled over-fetching (ask for 100 to keep 10). Selective filters, where the user can see only 0.1% of the corpus, are exactly where HNSW recall and latency can fall apart, so benchmark them explicitly.

**Freshness has two directions, with different stakes.** A new manual showing up 10 minutes late is an inconvenience. A *revoked* permission taking 10 minutes to apply is a disclosure. Give them separate service-level objectives. Deletions and revocations should propagate as **tombstones** (records that say "this is gone") to the BM25 index, the vector index, the parent-chunk store, and every cache. Then reconcile regularly against the source system's inventory to catch anything missed.

Caches deserve special suspicion. If you cache answers, the cache key must include the user's authorisation scope and the corpus version. Otherwise user A's cached answer, built from HR-only documents, gets served to user B. And keep audit logs per your retention policy without quietly keeping deleted document text in debug logs.

> 🎬 **Animation — post-filter starvation:** a grid of 100 chunk dots coloured by tenant (blue for the user's tenant, grey for others). A query point appears and a circle grabs the 10 nearest; 9 are grey. Step 1: a "post-filter" sieve removes the grey ones and 1 blue dot falls through, with a counter reading "results: 1". Step 2: rewind; the grey dots fade out first ("filter-aware search"), then the circle expands among only blue dots and catches 10, with the counter reading "results: 10". Step 3: a "revoke access" event hits one blue document; tombstone icons fly to four boxes labelled BM25, vectors, parent store, and answer cache, each with a lag timer.

## Packing the prompt and making citations mean something {#packing-citations}

You have a ranked list of candidates. Now you have to decide what actually goes into the prompt. This is **context packing**, and it's a budgeting problem, not "take the top k".

Start from the model's **context window**, the maximum number of tokens it can read at once. Reserve space for everything that isn't evidence *first*, then spend what's left.

```text
 context window: 16,000 tokens
 ┌──────────────┬───────────────┬──────────┬──────────────┬──────────────────────────┐
 │ system 1,000 │ history 1,500 │ question │ output       │ evidence: 12,000         │
 │              │               │   200    │ reserve 1,300│                          │
 └──────────────┴───────────────┴──────────┴──────────────┴──────────────────────────┘
   1,000 + 1,500 + 200 + 1,300 = 4,000 reserved → 16,000 − 4,000 = 12,000 for evidence
```

If each chunk is about 400 tokens plus about 30 tokens of ID label and delimiters, you could fit ⌊12,000 / 430⌋ = 27 chunks. But you probably shouldn't. More evidence means more cost, more latency (prefill work grows with prompt length), and more distractors. The better goal is **coverage without redundancy**. A two-part question ("what's the refund window, and does it apply to EU customers?") needs two *different* facts, not five near-copies of the best refunds paragraph. Deduplicate near-identical chunks, and prefer diversity when scores are close.

Placement matters too. Liu et al.'s "Lost in the Middle" found that models they tested did best when the relevant information was at the start or end of a long context, and noticeably worse when it sat in the middle. So test where your key evidence lands instead of assuming a bigger window means the model uses all of it. We cover long context in **Making attention cheaper: GQA, FlashAttention, and long context**.

Formatting rules that pay off:

- Wrap evidence in a clearly delimited data block, with each passage tagged by a stable ID, such as `[doc:HR-7 §3.2 v7]`.
- Tell the model that text inside documents is **data, not instructions**. A retrieved web page saying "ignore previous instructions" is an indirect prompt injection. The instruction reduces the risk but doesn't eliminate it, which is why permissions and tool limits live outside the model.
- When trimming or compressing passages, keep dates, scope ("applies to EU only"), exceptions, and units. A summary that drops a qualifier can turn a correct answer into an unsupported one.

**Citations.** Require a citation next to each factual claim, using the IDs you supplied. Then *resolve* those IDs back to real source locations in code, and never accept a URL the model invented. But a citation that resolves is only step one. There are three checks:

| Check | Question | Failure example |
|---|---|---|
| Resolves | Does the ID point to a passage we actually retrieved? | model invents `[doc:HR-9]` |
| Supports | Does that passage *entail* (logically imply) the claim? | cites the refunds page for a shipping claim |
| Covers | Is every important claim cited, and are conflicts surfaced? | the key number has no citation, or v6 and v7 disagree silently |

Finally, design for **abstention**. When the evidence is missing, contradictory, or out of date for the question, the right answer is a bounded one: "I can't find a current policy on X; the closest is Y (2023)". Or the system declines. Calibrate when to abstain on labelled cases where you know the answer isn't in the corpus. Don't treat a cosine score of 0.8 as "80% confident". Similarity scores aren't calibrated probabilities.

> 🎬 **Animation — the context budget bar:** a horizontal 16,000-token bar. Step 1: grey blocks fill in from the left for system 1,000, history 1,500, question 200, and output reserve 1,300, and a label shows "12,000 left". Step 2: evidence cards (400 + 30 tokens each) snap in from a ranked list; the 3rd and 4th cards are near-duplicates of the 1st and flash yellow, then get swapped for a card about a different sub-question. Step 3: the final answer appears with superscript citations, and each citation shoots a line back to its card, turning green (supports), amber (resolves but doesn't support), or red (doesn't resolve).

## Is it the retriever or the generator? Evaluating RAG {#evaluating-rag}

When an answer is wrong, the first question is *which half failed*. Evaluate them separately.

**Retrieval metrics** use your labelled set, where for each question you know which chunks are relevant:

```formula
Precision@k = relevant chunks in the top k / k
Recall@k    = relevant chunks in the top k / all labelled relevant chunks
RR          = 1 / rank of the first relevant chunk   (averaged over questions → MRR)
```

Worked example: a question has **4** labelled relevant chunks. The top 5 results are relevant at ranks 1, 3, and 5.

- Precision@5 = 3 / 5 = **0.60**
- Recall@5 = 3 / 4 = **0.75**
- Reciprocal rank = 1 / 1 = **1.0**

The reciprocal rank looks perfect, but a quarter of the evidence is missing. If that missing chunk holds the exception, the answer will be confidently wrong. That's why you look at more than one number. For graded labels (very relevant, somewhat relevant, not relevant), **nDCG** (normalised discounted cumulative gain) rewards putting the most useful evidence first. Always report the cutoff k and the label scale so the number means something.

Caveats: labels are usually incomplete (unlabelled chunks may be relevant too), and overlapping chunks can inflate hit rates. Measuring "did the top k cover each *required fact*" is often more honest than counting chunks. And again, this is **relevance** recall, not the ANN recall from the HNSW section.

**Generation metrics** are separate: answer correctness, **faithfulness** (is every claim supported by the provided context?), citation support, and abstention quality (does it decline when it should, and *only* when it should?). The Ragas paper (Es et al.) frames exactly this split: whether retrieval found relevant, focused context, whether the LLM used it faithfully, and the quality of the answer itself. Automated LLM judges make this cheap, but they need human spot-checks and stable rubrics. We cover judges and their calibration in **Evaluation and observability: knowing whether it actually works**.

**The oracle-context test** is the most useful diagnostic you have. For each labelled question, skip retrieval and hand the generator the *known correct* evidence:

```text
                         normal pipeline
                     correct          wrong
 oracle   correct │ all good     │ RETRIEVAL problem  │
 context          │              │ (evidence missing) │
          wrong   │ (lucky/      │ GENERATION problem │
                  │  check data) │ (prompt, model,    │
                  │              │  instructions)     │
```

If the model fails even with perfect evidence, no retrieval tuning will help. Fix the prompt, the model, or the format. If it succeeds with the oracle but fails in the real pipeline, the problem is upstream: parsing, chunking, search, or packing.

Slice every metric by tenant, language, question type, document freshness, and permission selectivity. Averages hide the tenant whose PDFs never parsed. Track latency and cost next to quality, and keep a held-out test set you never tune on.

> 🎬 **Animation — precision, recall, reciprocal rank:** five result slots with a checkmark at slots 1, 3, and 5, and a separate "missing" tray holding one relevant chunk. Step 1: the precision gauge fills to 3/5 = 0.60. Step 2: the recall gauge counts 3 of 4 and fills to 0.75, with the missing chunk pulsing. Step 3: the RR gauge jumps straight to 1.0 because slot 1 is a hit. Step 4: the missing chunk flips over to reveal "except custom orders", and the generated answer below turns red. The caption reads "a perfect RR can still give a wrong answer".

## RAG, fine-tuning, or neither {#rag-vs-finetuning}

This comes up in almost every interview. **Fine-tuning** means continuing to train the model's weights on your examples. It changes *behaviour*: tone, output format, domain terminology, how to follow a task. The details, including LoRA, are in **Fine-tuning on a budget: LoRA and QLoRA**. RAG changes *what evidence the model sees* on each request.

| Need | RAG | Fine-tuning |
|---|---|---|
| Facts that change daily | ✓ re-index in minutes | ✗ retrain for every change |
| Cite the exact source | ✓ you know which passage was used | ✗ knowledge is smeared across weights |
| Per-user permissions | ✓ filter per request | ✗ weights are shared by all users |
| Delete a document (legal request) | ✓ remove the chunks | ✗ no clean per-document "unlearn" |
| Consistent format and style, domain phrasing | partly, via prompt | ✓ |
| Lower per-request tokens | ✗ evidence costs tokens | ✓ behaviour is baked in |

The rule of thumb: **put volatile, attributable, permissioned facts in retrieval, and put stable behaviour in weights.** They combine well. For example, you can fine-tune a model to cite consistently and to abstain when evidence is thin, while all the facts stay in the index.

Also defend the *simplest* system that meets the contract:

- **Tiny, stable corpus?** It may fit straight into the prompt, perhaps with prefix caching so the shared document isn't re-processed every request (see **What happens at inference: prefill, decode, and the KV cache**).
- **Structured facts?** "What's order 8812's status?" is a database query, not a similarity search.
- **Multi-hop questions** (answering needs a first lookup to decide the second)? Consider query rewriting or iterative retrieval. That drifts into agent territory, covered in **Agents and tools: from model decisions to safe actions**.

Add rewriting, multi-step retrieval, or reranking only when failure analysis says you need them. Version everything (parser, chunker, embedding model, index, prompt), roll out index changes behind an alias you can flip back, and keep enough corpus snapshots to reproduce a bad answer. A strong interview answer isn't a list of components. It explains how you'd *find out* why a specific answer was wrong, and how you'd fix it.

> 🎬 **Animation — where knowledge lives:** two containers side by side, "Model weights" (a sealed block) and "Search index" (a filing cabinet). Step 1: a "policy updated" event arrives; the cabinet swaps one folder in 2 seconds, while the weights block shows a "retrain: days" progress bar. Step 2: a "user lacks access" event arrives; the cabinet locks one drawer for that user, while the weights have no drawer to lock. Step 3: a "respond in JSON with citations" requirement arrives; a fine-tuning arrow reshapes the weights block, labelled "behaviour". End on a combined picture: fine-tuned model + filing cabinet.

# Interview

## Question

Design an internal policy assistant for a company with many business units (tenants). Policies change daily, group membership changes take effect immediately, and every answer must cite its evidence. How do you keep it accurate, and how do you make sure it never shows someone a document they're not allowed to see, including right after their access is revoked?

## Answer

I'd start with the contract, not the stack. Answers come only from current, published policy. Every factual claim cites a policy section and version. If the evidence is missing or conflicting, the assistant says so. Then I'd set two separate freshness objectives: new content visible within, say, 15 minutes, and revocations enforced within seconds or before the next request, because a late revocation is a disclosure. I'd build a labelled set of a couple hundred questions, including unanswerable ones, conflicting versions, and permission-boundary cases, before tuning anything.

**Ingestion:** parse structure (headings, tables, exceptions), chunk at section boundaries with modest overlap, give every document a stable ID and every chunk a versioned ID, and keep a manifest from document version to chunks so replacements and deletions remove every derived chunk atomically. Every chunk carries tenant and ACL metadata plus parser and embedding versions.

**Retrieval:** identity and groups come from the identity provider, never the prompt. Every path (BM25, dense, parent expansion, fallbacks) filters on tenant and permissions *inside* the search engine, before any text reaches the reranker, the model, logs, or caches. I'd use filter-aware ANN or per-tenant partitions, because post-filtering starves results for selective users. I'd run hybrid BM25 + dense, fuse with RRF, rerank the top ~50 with a cross-encoder, and pack about 6–10 diverse passages within a fixed token budget, each tagged with an ID.

**Revocation:** a membership change emits an event. Group resolution happens at query time, so the next query is filtered correctly. Answer caches are keyed on authorisation scope and corpus version, or revalidated before reuse. Tombstones propagate to every index and cache, with lag monitored and nightly reconciliation against the source systems.

**Generation:** citations are resolved in code, and I'd check claim support with sampled human review plus an automated entailment check. The model abstains when evidence is thin or contradictory.

**Evaluation:** I'd track relevance recall@k and nDCG separately from ANN recall, plus faithfulness, citation support, and abstention accuracy. Oracle-context runs tell me whether failures are retrieval or generation. Everything is sliced by tenant and permission selectivity, with p95 latency and cost alongside. Index and model changes ship behind versioned aliases with rollback.

## Follow-ups

- How do very restrictive tenant filters affect HNSW recall and latency, and what would you do about it?
- A user says the bot ignored an exception in the policy. How do you tell whether the exception was never retrieved or retrieved and ignored?
- Two versions of the same policy (current and historical) both match the query. How do you decide what to show?
- How do you invalidate a cached answer when someone's group membership changes?
- You want to switch embedding models. Walk me through the migration without downtime or a quality regression.

# Pitfalls

- Treating cosine similarity or RRF scores as calibrated confidence. They're ranking signals, so calibrate abstention on labelled cases instead.
- Enforcing permissions with a prompt instruction, or filtering only after unauthorised text has reached a model, reranker, log, or cache.
- Post-filtering a small global top-k and starving users with narrow permissions.
- Switching embedding models (or preprocessing) without re-embedding the corpus into a new, compatible index.
- Updating documents by upserting chunk IDs with no manifest, so orphaned chunks from a longer old version stay searchable.
- Assuming a reranker can fix bad recall. It can only reorder the candidates it receives.
- Stuffing the context window with near-duplicate chunks and forgetting to reserve tokens for instructions and output.
- Counting a citation that resolves as proof that the passage supports the claim.
- Confusing ANN recall (approximate vs exact neighbours) with relevance recall (retrieved vs truly relevant evidence).
- Optimising one end-to-end answer score without separating missing evidence from misused evidence.

# Checklist

- Write an evidence contract covering authority, freshness, access, citations, and abstention.
- Build a labelled question set that includes unanswerable, conflicting, and permission-boundary cases.
- Chunk at structural boundaries, and verify tables, exceptions, and cross-boundary questions.
- Version document and chunk IDs, and keep a manifest so deletes and replacements remove every derived chunk.
- Compute cosine similarity and raw vector memory by hand, and explain what HNSW's M, ef_construction, and ef trade off.
- Compare BM25, dense, hybrid (RRF), and reranked baselines on held-out questions.
- Enforce tenant and ACL filters inside every retrieval path, and test with revoked users and very selective filters.
- Budget the context window, reserving system, history, question, and output tokens before adding evidence.
- Audit citations for resolution, claim support, and coverage.
- Use oracle-context runs to tell retrieval failures from generation failures, and slice metrics by tenant and question type.
- Defend RAG vs fine-tuning vs long context vs a database query for a given requirement.

# Sources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — Original RAG paper: a seq2seq generator (parametric memory) combined with a dense vector index of Wikipedia searched by a neural retriever (non-parametric memory).
- [Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906) — The dual-encoder (bi-encoder) approach to dense retrieval, compared against Lucene BM25.
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320) — HNSW's multi-layer proximity graphs, with layer levels drawn from an exponentially decaying distribution.
- [hnswlib — Algorithm parameters](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) — Definitions of M (range 2–100, roughly M × 8–10 bytes per element), ef_construction, and ef (must be ≥ k).
- [Elasticsearch — Similarity settings](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity) — BM25 is the default similarity, with defaults k1 = 1.2 and b = 0.75.
- [Elasticsearch — Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — The RRF formula, with ranks starting at 1 and a default rank constant of 60.
- [Passage Re-ranking with BERT (Nogueira & Cho, 2019)](https://arxiv.org/abs/1901.04085) — Cross-encoder reranking, including the 27% relative MRR@10 gain on MS MARCO.
- [Azure AI Search — Document-level access control](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview) — Security-filter trimming vs token-based permission enforcement, and the fact that permission changes apply only after metadata sync.
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172) — Performance was highest with relevant information at the start or end of the context and degraded in the middle.
- [Ragas: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217) — Separates context relevance, faithfulness, and answer quality as evaluation dimensions.

# Flashcards

## rag-contract

**Q:** What does RAG add to ordinary generation, and what does it cost you?

It retrieves external evidence at request time and puts it in the prompt, turning a closed-book exam into an open-book one. You gain knowledge you can update without retraining, citations to specific sources, and per-user access control.

The cost is new failure points: the source can be stale, parsing can mangle it, chunking can split it, search can miss it, permissions can leak it, and the model can still ignore it. Debugging RAG means finding which stage broke.

## chunk-boundaries

**Q:** Why can chunks that are too small make answers worse?

Small chunks can separate a rule from its exception ("refunds within 30 days" / "except custom orders") or a table value from its column heading. Search may then retrieve a chunk that matches well but has lost the qualifier that changes its meaning.

Too big has the opposite problem: several topics get averaged into one fuzzy embedding, and the chunk wastes prompt tokens. Structure-aware splits with modest overlap, and sometimes small-to-big parent expansion, balance the two.

## chunk-manifest

**Q:** Why keep a manifest mapping each document version to its chunks?

Because replacements and deletions must remove *every* derived chunk. If v1 of a document produced 12 chunks and v2 produces 9, upserting by chunk ID leaves chunks 10–12 from v1 searchable, so the bot keeps quoting a deleted clause.

With a manifest, an update is "remove everything listed for v1, publish v2's list", ideally as an atomic swap.

## acl-boundary

**Q:** Why is "tell the model not to reveal restricted documents" not access control?

By the time the model reads the instruction, it has already received the restricted text. The same text may also have reached a reranker, logs, traces, or caches. Prompts are suggestions, and prompt injection can override them.

Permissions must be enforced by trusted infrastructure inside the retrieval layer, on every path, before any text is passed downstream.

## postfilter-recall

**Q:** How can post-filtering search results destroy recall?

If you fetch the global top 10 and then drop what the user can't see, unauthorised neighbours use up your fixed budget. With 9 of the 10 belonging to other tenants, you return 1 result even though many good authorised chunks sat just outside the top 10.

Fixes include filter-aware ANN, per-tenant partitions, or deliberate over-fetching. Benchmark the most selective filters explicitly.

## embedding-compatibility

**Q:** Two embedding models both output 768-dimensional vectors. Can you search one's index with the other's query vectors?

No. Each model learns its own space, so the same meaning lands at unrelated coordinates. Equal dimensions only make the dot product computable, not meaningful.

Queries and documents must use a compatible model and the same preprocessing. Changing models means re-embedding the corpus into a new index, evaluating it, and then switching traffic.

## hnsw-controls

**Q:** What do HNSW's M, ef_construction, and ef control?

**M** is how many links each node gets. More links give better recall on hard data, at the cost of memory (roughly M × 8–10 bytes per element in hnswlib) and slower builds. **ef_construction** is how hard the builder searches for good links: better graph quality, longer builds. **ef** is the candidate list size at query time (≥ k): higher ef means higher recall and slower queries.

All of these are empirical tradeoffs to benchmark on your own data, including with filters and at tail latency.

## two-recalls

**Q:** How is ANN recall different from relevance recall?

ANN recall compares the approximate index's results with *exact* nearest-neighbour search in the same vector space. It measures the index. Relevance recall compares retrieved chunks with human-labelled relevant evidence. It measures whether retrieval found what answers the question.

You can have 99% ANN recall and poor relevance recall if the embedding model ranks irrelevant text as near. Measure both separately.

## hybrid-value

**Q:** Why combine BM25 with dense retrieval?

They fail differently. Dense embeddings match paraphrases ("forgot login" ≈ "reset password") but can blur exact strings like error codes, SKUs, and rare names. BM25 nails exact and rare terms but misses different wording for the same idea.

Running both and fusing the results widens the candidate pool. Whether it actually helps your corpus is something you confirm on held-out questions.

## rrf-meaning

**Q:** What problem does reciprocal rank fusion solve, and what doesn't its score mean?

BM25 scores and cosine similarities live on unrelated scales, so adding them is meaningless. RRF uses only ranks: each list contributes 1 / (k + rank), with k = 60 by default in Elasticsearch. A doc at ranks 3 and 3 (2/63 ≈ 0.0317) beats one at ranks 1 and 10 (≈ 0.0307), because agreement is rewarded.

The result is a ranking score, not a probability of relevance or correctness. Don't threshold it as confidence.

## reranker-limit

**Q:** What failure can a reranker never fix?

A missing candidate. A cross-encoder only reorders the pool it's given, so if first-stage retrieval never surfaced the right passage, reranking can't recover it.

Make sure candidate recall is good first, and tune retrieve count, rerank count, and final context count separately.

## bi-vs-cross-encoder

**Q:** Why use a bi-encoder for retrieval but a cross-encoder for reranking?

A bi-encoder embeds the question and passages separately, so you can embed millions of passages offline and search them in milliseconds. But the two texts never interact token by token. A cross-encoder reads the question and passage together, so attention can compare every token pair. That's much more accurate, but it needs one forward pass per pair at query time.

So you use the cheap model to shortlist from millions and the expensive one to order the top tens.

## context-packing

**Q:** Your model has a 16,000-token window. How do you decide how much evidence to include?

Reserve tokens first: for example, system 1,000 + history 1,500 + question 200 + output 1,300 = 4,000, which leaves 12,000 for evidence, including the ID labels and delimiters.

Then optimise for *coverage without redundancy*, not raw count: distinct facts for each part of the question, near-duplicates removed, and qualifiers kept. More context costs latency and money and adds distractors. Models may also underuse evidence placed in the middle of a long context.

## citation-support

**Q:** Why isn't a citation that resolves to a real document enough?

Resolving only proves the ID exists. The cited passage must also *support* (entail) the claim, and every important claim must be covered. The passage could be about something else, out of scope, or an obsolete version.

Check three things: it resolves (in code, never accepting invented URLs), it supports the claim, and the answer's claims are covered, with conflicts surfaced.

## oracle-context

**Q:** What does an oracle-context experiment tell you?

You skip retrieval and give the generator the known-correct evidence. If it still fails, the problem is generation: the prompt, the instructions, or the model. If it succeeds with the oracle but fails in the real pipeline, the problem is upstream: parsing, chunking, search, filtering, or packing.

It's the quickest way to stop tuning the wrong half.

## rag-finetuning

**Q:** When do you use RAG, fine-tuning, or both?

Use RAG for facts that are volatile, need citations, need per-user permissions, or must be deletable. Weights can't be filtered per request or cleanly unlearn one document. Use fine-tuning for stable behaviour: format, tone, domain phrasing, and consistent use of evidence.

Combine them by fine-tuning the model to cite and abstain well while the facts stay in the index. Also consider simpler options: a tiny corpus in the prompt, or a database query for structured facts.
