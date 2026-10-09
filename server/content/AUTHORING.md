# Article authoring guide

Articles live in `server/content/articles/SLUG.md`, one per file. Legacy articles in `server/content/SLUG.json` still load. A Markdown file with the same slug replaces its JSON file, and once an article is rewritten its JSON is deleted. `shared/article-markdown.ts` parses the Markdown into the `Article` shape in `shared/content.ts`, so the frontend is unchanged. To check a file, run `node scripts/check-article.ts SLUG`. After content changes, run `npm.cmd test` and `npm.cmd run db:setup`.

## Who we're writing for, and how

The reader is a capable software engineer who is **new to ML** and is preparing for **senior-level AI/LLM interviews**. Write the way a senior engineer teaches a sharp new teammate at a whiteboard: relaxed, direct, and a bit conversational ("here's the trick", "this is where people get tripped up"). Explain things properly. Don't dumb them down, and skip the corporate filler.

- **Self-contained.** Define every term in plain words the first time you use it, before you depend on it. If a concept belongs to another article, give a one- or two-sentence intuition and point ahead, e.g. "*we'll go through this properly in* **Serving many users: batching, scheduling, and speculative decoding**." Don't use an idea you haven't explained or pointed to.
- **Intuition → example → mechanism → consequence.** For each idea: start with an analogy or picture, then a tiny concrete example with real numbers, then the precise mechanism or formula, then why it matters (the tradeoff, failure mode, or what an interviewer will probe).
- **Draw a lot.** Put napkin sketches in ` ```text ` blocks as ASCII diagrams (tensor shapes, data flow, timelines, memory layouts). Use Markdown tables for comparisons. Put equations in ` ```formula ` blocks as plain Unicode math (`softmax(QKᵀ / √d_k) · V`), not LaTeX, and explain every symbol right after.
- **Mark where a real visual belongs.** Richer animations and diagrams are drawn later. Leave a placeholder paragraph wherever one would teach well (at least 3 per article, usually 5–8):
  `> 🎬 **Animation — Q, K and V projections:** show a 3-token × 4-feature matrix multiplied by W_Q… step 1 …, step 2 …`
  Describe exactly what the visual should show and in what steps, including the toy numbers, so an illustrator can build it without rereading the article.
- **Worked numbers are checked.** Recompute every arithmetic example. Give units and state your assumptions.
- **Senior depth.** Still reach interview-grade substance: tradeoffs, failure modes, back-of-envelope sizing, and how to defend a choice. Plain language isn't the same as shallow.
- **No stale facts.** Don't make claims about the "newest" model or fixed prices. Verify version-dependent claims against primary sources you actually opened (papers, official docs). Never fabricate a benchmark or measurement, and label illustrative numbers as illustrative.

Length: roughly **2,500–4,500 words** of section prose, in **6–10 sections**. Farm guides can run longer. Headings should read like what a teacher would say, e.g. "Why attention needs three different vectors", not "Q/K/V Projections".

## File format

````markdown
---
{
  "slug": "attention-from-scratch",
  "title": "Attention from scratch: how tokens talk to each other",
  "category": "foundations",
  "summary": "One or two sentences shown under the title and on cards.",
  "difficulty": "Core",
  "minutes": 25,
  "prerequisites": ["tokens-and-embeddings"],
  "learningObjectives": ["Explain …", "Calculate …", "Defend …"]
}
---

# Sections

## Why a word needs its neighbours {#why-context}

Paragraphs are separated by blank lines. Normal Markdown: **bold**, `code`, lists, tables.

```text
 "the" "cat" "sat"
   │     │     │
   └──►  mix  ◄┘
```

> 🎬 **Animation — context mixing:** three token vectors … step 1 … step 2 …

```formula
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

## Next section title {#next-section}

…

# Interview

## Question

The whiteboard question, one paragraph.

## Answer

A model answer: several paragraphs, Markdown allowed.

## Follow-ups

- A deeper question an interviewer would ask next
- Another one

# Pitfalls

- A common wrong answer, and why it's wrong.

# Checklist

- Something the reader should now be able to do.

# Sources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — What this source supports.

# Flashcards

## scaled-dot-product

**Q:** Why divide by √d_k before the softmax?

The answer, explaining *why* rather than just *what*. It can run to several paragraphs.
````

Rules the parser and tests enforce:

- Top-level `#` headings are exactly: Sections, Interview, Pitfalls, Checklist, Sources, Flashcards. Don't use `#` headings anywhere else outside code fences. Inside sections, use `###` or bold for sub-headings.
- Every section heading ends in a unique `{#kebab-id}`.
- `difficulty` is one of `Core`, `Advanced`, `Systems`. Prerequisites must be real slugs.
- 3+ learning objectives, 6+ sections, 1,200+ words of prose, 3+ `> 🎬` placeholders, 2+ follow-ups, 3+ pitfalls, 3+ checklist items, and 4+ HTTPS sources in the `- [Title](url) — note` format.
- 12–16 flashcards with unique kebab-case keys. When a card's concept carries over from the legacy JSON, **reuse its old key** so review history is kept. Cards removed from the source are retired, not deleted.
- Every code fence needs a language: `text` for drawings, `formula` for equations, or a real language for code.

## Categories and curriculum order

The canonical order is in `shared/catalog.ts`. Each article should know what its neighbours cover, so it can point ahead instead of re-teaching them.

## Corporate AI farm guides

The farm is **educational content**, not this app's infrastructure. `FARM-CONTRACT.md` and the files in `examples/ai-farm/` are the source of truth, and the guides must match them. Distinguish runnable examples from fragments. Don't claim anything was executed or measured. Never embed secrets; use environment-variable placeholders.
