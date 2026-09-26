# Article authoring contract

Write exactly one article per assignment to `server/content/SLUG.json`. Use valid JSON matching `shared/content.ts`. All content must be original explanatory prose, verified by web search AND opening primary sources. Include source URLs, descriptive title, and the claim each source supports. No citations to search result URLs. Avoid stale claims about newest models and fixed prices.

Target a senior software/ML interview audience: define terms, mechanisms, equations with units/assumptions, a worked numerical example, tradeoffs, failure cases, and how to defend design choices. 1,400–2,200 words of article prose (farm guides may exceed this). At least six substantial sections, at least FIVE (prefer six) accurate diagrams placed where they teach. 12–16 distinct flashcards; answers should explain why. Do not write superficial bullet-only summaries. Include prerequisites (slug strings from list below), learningObjectives, one interview scenario with answer and follow-up questions, pitfalls, actionable checklist, 4+ primary sources, estimated reading minutes.

Diagram types:
- flow / steps: 3–6 nodes `{label,detail}`, rendered as numbered connected cards.
- compare: 2–4 contrasting nodes with substantive detail.
- bars: nodes with `value` numeric; specify units/normalization in caption, avoid misleading metrics.
- curve: `xLabel`, `yLabel`, `series:[{name,points:[[x,y],...]}]`. Points need >=3 samples, finite numbers. Caption MUST say if illustrative rather than measured. Values reflect the mechanism.
Every diagram requires kind, title, caption. No external images/HTML. Equations in `formula` as legible plain Unicode math, not LaTeX. Optional code `{language,title,value}` uses real newline characters. Each section can have one code block. Use separate sections for separate files when needed. Prose can use Markdown inline formatting, but no inline HTML.

## Structural visual walkthroughs

Introduce transformer mechanisms with a concrete picture alongside the prose. A section can include `visuals: [{"id":"foundation-embedding","afterParagraph":2}]`; paragraph indexes start at zero. Keep the existing `diagram` for its complementary overview or chart. Walkthrough IDs resolve through `shared/visuals/index.ts` and use the typed, original SVG scenes in `shared/visuals/`. They are rendered by `src/VisualLesson.tsx`.

Each walkthrough needs a title, summary, toy-example note and distinct steps with a title, explanatory description and visual elements. Show actual cells, vectors, connections or geometry, not just shape notation or text cards. Use stable element IDs across frames for transitions, small matrices and legible labels. The canvas is 720 × 380; leave margins and space for row/column labels. State whether rows mean token positions, features, vocabulary choices or stored state. Verify numerical examples and distinguish model parameters, activations and optimizer/cache state. The player provides manual stepping, explicit playback, slower playback, reduced-motion behavior and text equivalents. Run `npm.cmd test`, build, seed and inspect affected article layouts after adding placements.

The local corporate AI farm is EDUCATIONAL CONTENT, not a deployment of actual inference servers. Linux Docker examples. The app itself is a Windows-hosted learning application. Distinguish runnable core examples from pseudocode or fragments explicitly. Do not claim examples executed or performance measured. No embedded secrets; environment variables with placeholders.

Categories / slugs:
foundations: transformer-foundations, optimization-generalization, pretraining-data-scaling
alignment: post-training-alignment, parameter-efficient-finetuning
architecture: efficient-attention, moe-and-parallelism, quantization-and-compression
inference: serving-kv-cache, inference-scheduling
applications: rag-retrieval, agents-tool-systems
evaluation: evaluation-observability, llm-security
ai-farm: farm-blueprint, farm-deployment, farm-routing-consistency
system-design: interview-design

Preserve given slug/category. Do not modify shared files or other article files. Tell parent when done with word/diagram/card count, verification links, and caveats. If a tool becomes unavailable, report honestly rather than fabricate verification.
