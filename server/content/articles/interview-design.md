---
{
  "slug": "interview-design",
  "title": "The senior LLM system design interview",
  "category": "system-design",
  "summary": "One full mock interview, start to finish: turn \"build us an assistant\" into contracts, token rates, an architecture, a GPU-sized capacity plan that survives a failure, a cost per successful task, and a release you can take back.",
  "difficulty": "Systems",
  "minutes": 30,
  "prerequisites": ["serving-kv-cache", "inference-scheduling", "rag-retrieval", "agents-tool-systems", "evaluation-observability", "llm-security", "farm-routing-consistency"],
  "learningObjectives": [
    "Translate a vague product ask into separate quality, reliability, and data-policy contracts with numbers attached.",
    "Turn user counts into request rates, input and output token rates, and in-flight concurrency using Little's law.",
    "Size a replica from first principles (weights, KV cache, decode bandwidth, prefill FLOPs), then size a fleet that still meets the peak after a failure.",
    "Compare designs by cost per successful task rather than price per token.",
    "Defend a release process (evaluation gates, canary, versioned bundle, rollback) and name the evidence that would change each decision."
  ]
}
---

# Sections

## What the interviewer is actually grading {#what-is-graded}

Here's the secret of the senior LLM system design interview: nobody is grading whether you know the name of every serving framework. They're grading whether your **assumptions turn into decisions**, whether those decisions come with **numbers**, and whether the design **bends instead of shattering** when they change a requirement halfway through. A junior answer is a pretty box diagram. A senior answer is a box diagram where every box exists for a reason you can state, every arrow has a failure you've thought about, and every number has a unit.

Think of it like a house inspection rather than an architecture competition. The inspector doesn't care that you picked a fashionable brand of pipe. They care that the pipes are sized for the water pressure, that there's a shutoff valve, and that you know what happens when one bursts.

This article is the capstone of the curriculum, so it leans on everything before it. Rather than re-teach, it restates the one fact it needs each time and points you to the article that owns the details. Then it walks through **one mock interview from the first minute to the last**, with every number worked out.

Time is the scarce resource, so walk in with a plan. Here's one 45-minute budget. Treat it as something you negotiate with the interviewer ("I'd like to spend a few minutes on requirements before drawing, does that work?"), not a script.

| Minutes | Phase | What you produce |
|---|---|---|
| 0–6 | Requirements | User task, exclusions, data boundary, quality and latency targets |
| 6–12 | Workload | Request rates, token budgets, concurrency |
| 12–22 | Architecture | Components, who owns which state, where permissions are enforced |
| 22–31 | Capacity | Per-replica napkin math, fleet size with failure headroom, cost |
| 31–39 | Validation | Evaluation set, load tests, canary, rollback |
| 39–45 | Defense | Curveballs, alternatives, what would change your mind |

One habit makes the whole thing work: keep a **decision ledger** in the corner of the whiteboard. It's a running list of "decided X because Y". When the interviewer says "actually, traffic triples", you scan the ledger, find the lines that depended on traffic, and revise just those. You don't erase the diagram and start over.

```text
 DECISION LEDGER
 ─────────────────────────────────────────────────────────
 D1  self-host the model      ← restricted docs can't leave
 D2  RAG, not fine-tuning     ← policies change weekly
 D3  6 replicas               ← 8 req/s burst, survive 1 loss
 D4  cap input at 12k tokens  ← protects KV memory + TTFT
 ...
```

> 🎬 **Animation — the decision ledger under a curveball:** show the whiteboard with a box diagram on the left and the four-line ledger above on the right. Step 1: the interviewer's speech bubble says "restricted docs may now go to an approved cloud API". Step 2: ledger line D1 flashes red; a dotted line from D1 highlights the "self-hosted model" box in the diagram. Step 3: D3 flashes amber because capacity now splits between local and API. Step 4: D2 and D4 stay green. Step 5: only the highlighted box is redrawn; the rest of the diagram never moves. Caption: "revise the affected decisions, not the whole design".

## Minutes 0–6: turning "build us an assistant" into three contracts {#requirements}

The interviewer opens with: *"Design an internal assistant that answers employees' policy questions. We have 5,000 employees."*

That's deliberately vague. Your first job is to make it concrete enough to compute with. Here's the problem we'll carry through the whole article:

- The assistant answers HR, travel, and expense **policy** questions from current company documents.
- Every answer **cites** the passages it used, and the assistant **abstains** ("I couldn't find that in current policy") when the evidence isn't there.
- It **cannot act**: no approving expenses, no editing records. That one exclusion removes most of the scary security surface, and you should say so out loud.

**Traffic.** "5,000 employees" is not a request rate. Ask how many are active and how often they ask. Suppose the interviewer accepts these illustrative assumptions:

```text
 5,000 employees × 40% daily active        = 2,000 active users/day
 2,000 users × 10 questions each           = 20,000 requests/day
 spread over an 8-hour workday (28,800 s)  = 0.69 requests/s average
 busiest hour ≈ 3× average                 ≈ 2 requests/s   ("busy")
 all-hands announces a new travel policy   = 8 requests/s for 10 minutes ("burst")
   → 8 × 600 s = 4,800 requests in the burst
```

The burst is the number that sizes the system. Always ask about synchronized events (an all-hands, a policy deadline, Monday 9 a.m.), because averages hide them.

Now pin down "good". The trick is that there are really **three separate contracts**, and passing one tells you almost nothing about the others:

| Contract | Example target (hypothetical, for discussion) | Who cares |
|---|---|---|
| Product quality | ≥ 90% task success on a held-out test set, where the rubric checks correctness, citation support, permission compliance, and abstaining when it should | Users, legal |
| Service reliability | p95 time to first token ≤ 2 s; p95 full answer ≤ 15 s; 99.5% of eligible requests complete without infrastructure failure over a rolling 30 days | On-call, users |
| Data policy | restricted documents never leave company infrastructure; logs retained N days; only named roles can read traces | Security, compliance |

A few terms. **Time to first token (TTFT)** is how long until the reply starts streaming. **p95** means 95% of requests do at least that well. A **held-out test set** is a set of examples you never tuned on, so the score isn't flattering itself. A **service level objective (SLO)** is an explicit reliability target like the row above, measured over a stated window. Google's SRE workbook recommends **rolling** windows because users don't forget an outage just because the calendar month flipped.

SLOs also give you an **error budget**, which is 100% minus the target. At 20,000 requests × 22 workdays = 440,000 requests/month, a 99.5% target allows 0.5% × 440,000 = **2,200 failed requests a month**. That number turns reliability into a decision: if a risky deploy burns 1,500 of them, you freeze deploys.

Two details separate senior answers. First, say whether **rejected** requests (the system is full and says "try again") count against availability. If you promise capacity for the burst, they should. Second, a **permission leak** (showing someone a document they can't see) is not "a 1% quality dip". It's a release blocker. It gets its own gate, not a line in an average.

> 🎬 **Animation — three independent contracts:** three gauges side by side labeled Quality (needle at 91%, threshold 90%), Reliability (p95 TTFT 1.8 s vs 2 s limit), Data policy (a padlock). Step 1: all three green. Step 2: a new, bigger model is swapped in: Quality rises to 94%, but the Reliability needle swings to 3.1 s and turns red. Step 3: swap to a cloud API: Quality 95%, Reliability green, but the padlock breaks open and turns red because restricted documents would leave. Caption: "passing one contract never implies the others".

## Minutes 6–12: from requests to tokens {#workload}

Models don't process requests. They process **tokens**, chunks of text of roughly three-quarters of an English word each. Everything about cost and capacity is measured in tokens, so convert. *Tokenization is covered properly in* **Text in, next token out: tokens, embeddings, and sampling**.

Budget the whole model input, not just the user's question:

```text
 input tokens per request (illustrative means)
 ┌──────────────────────────────┬───────┐
 │ system prompt + rules        │   500 │ ██
 │ user's question              │   300 │ █
 │ conversation history         │ 1,200 │ █████
 │ retrieved policy passages    │ 2,000 │ ████████
 ├──────────────────────────────┼───────┤
 │ total input                  │ 4,000 │
 └──────────────────────────────┴───────┘
 output: mean 400 tokens, hard cap 600
```

Averages are for throughput; limits are for safety. Cap input at, say, 12,000 tokens (trim or summarize history beyond that) and reserve 600 for output, so the worst case needs a context window of at least 12,600 tokens. Pick a model with room to spare, say 16K. The **context window** is the most tokens a model can handle in one request, input plus output together.

Now multiply by the burst rate. Use **λ** (lambda) for arrival rate in requests per second:

```formula
input demand  = λ × mean input tokens  = 8 req/s × 4,000 = 32,000 tokens/s
output demand = λ × mean output tokens = 8 req/s × 400   =  3,200 tokens/s
```

Why keep them separate? Because an LLM serves each request in two very different phases. **Prefill** reads the whole prompt in one big parallel pass. It's limited by raw arithmetic (FLOPs). **Decode** writes the answer one token at a time, and each step has to stream the model's weights out of GPU memory again. It's limited by memory bandwidth. It's like reading a letter (you take in the whole page at once) versus writing the reply (one word at a time, and you reach for the dictionary before every word). The two phases hit different hardware limits, so "35,200 tokens/s" is a meaningless sum. *We go through this properly in* **What happens at inference: prefill, decode, and the KV cache**.

**How many requests are in flight?** Use **Little's law**: in any stable system, the average number of things inside equals the arrival rate times the average time each one spends inside.

```formula
L = λ × W
```

**L** is the average count in the system, **λ** is arrivals per second, and **W** is the average time in the system. The picture is a coffee shop: 8 customers walk in per second and each stays 10 seconds, so about 80 people are inside at any moment. The law holds regardless of how arrivals or service times are distributed, but it needs a *stable* system (arrivals don't outrun departures) and you have to say **which boundary** you're measuring.

Suppose the end-to-end time is about 10 s: roughly 2 s until the first token (queue + retrieval + prefill), then 8 s to decode the remaining ~400 tokens. That implies each stream must run at about 400 ÷ 8 = **50 tokens/s**. Now apply the law at two boundaries:

```text
 whole system:       L = 8 req/s × 10 s   = 80 requests in flight
 on the GPU (decode):L = 8 req/s × ~8.2 s ≈ 66 sequences holding GPU memory
                         (8 s decode + ~0.2 s prefill)
```

Here's where people get tripped up. The 80 is an **average** count, not a p95 count. Multiplying the peak rate by the p95 latency does **not** give you "p95 concurrency". And 80 in the system doesn't mean 80 GPU slots, since some requests are waiting in a queue or on retrieval. Measure concurrency per stage.

> 🎬 **Animation — Little's law at two boundaries:** a pipeline drawn left to right: "gateway queue" → "retrieval" → "GPU: prefill | decode" → "done". Dots enter at 8 per second (a ticker counts arrivals). Step 1: dots spend ~1.5 s in the queue and retrieval zone, ~0.2 s in prefill, 8 s in decode. Step 2: a dashed box around the whole pipeline shows a live counter settling near 80. Step 3: a second dashed box around only the GPU shows its counter settling near 66. Step 4: a speech bubble: "L = λW: pick the box first".

## Minutes 12–22: the architecture, one independent decision at a time {#architecture}

A common mistake is framing the design as "local model **or** API **or** RAG **or** fine-tuning", as if those were four rival architectures. They're answers to three **different** questions:

| Axis | Question it answers | Options |
|---|---|---|
| Hosting | Where does the computation run? | Self-hosted GPUs, managed API, or both split by data class |
| Knowledge | Where do facts come from? | Retrieval, or just the prompt when everything fits |
| Behavior | How do we shape the style and format of answers? | Prompting first, fine-tuning if labeled failures justify it |

You pick on each axis separately. A self-hosted, fine-tuned model can also use retrieval.

**Hosting: self-host.** Restricted documents can't leave company infrastructure, so the core model runs on our own GPUs. Mention the alternative: an approved API could serve the *public*-document slice if it wins on quality per cost in evaluation. But a restricted request must **never** fail over to it, even in an outage.

**Knowledge: retrieval.** Policies change weekly, and the answer has to cite a specific current version. **Retrieval-augmented generation (RAG)** means the system first searches a document index for relevant passages and pastes them into the prompt. It's an open-book exam: the model doesn't memorize the policy manual; it looks up the right page. The original RAG paper (Lewis et al., 2020) framed this as combining knowledge stored in the weights ("parametric memory") with an external index you can swap without retraining. That swap is exactly what we want. Updating a policy means re-indexing a document, not retraining a model. And document permissions can be enforced **at retrieval time**, per user. *Chunking, embeddings, hybrid search and reranking are all in* **RAG: giving the model the right evidence**.

**Behavior: prompt first.** Fine-tuning only enters when you have labeled examples showing a *recurring* behavior problem that prompting and better retrieval didn't fix, like the model keeps botching the citation format. Then **LoRA** is the usual tool. It freezes the base model and trains small low-rank "adapter" matrices on top. Fewer trainable parameters don't remove the costs of preparing data, running regression evals, and keeping adapters compatible with the serving stack. And fine-tuning never supplies fresh facts or enforces permissions. *Details in* **Fine-tuning on a budget: LoRA and QLoRA**.

Now draw the request path, and for each box say **what state it owns**:

```text
 browser
    │  HTTPS, SSE stream back
    ▼
 ┌──────────┐   TLS, size limits, rate limits
 │ gateway  │   least-connections → API replicas
 └────┬─────┘
      ▼
 ┌─────────────────────┐      ┌──────────────┐
 │ API (stateless)     │─────►│  Postgres    │  durable: turns, revisions,
 │ auth, conversation  │◄─────│              │  idempotency keys, status
 │ authz, deadlines,   │      └──────────────┘
 │ provider policy     │      ┌──────────────┐
 │                     │─────►│ retrieval    │  filters by the USER's
 │                     │◄─────│ (versioned)  │  document permissions
 │                     │      └──────────────┘
 │                     │      ┌──────────────┐
 │                     │─────►│ inference ×N │  owns only compute and
 │                     │◄─────│ same model   │  disposable KV caches
 └─────────────────────┘      └──────────────┘
```

The rule of thumb: **correctness lives in the database; speed lives in caches**. The API writes the user's turn to Postgres *before* generation starts, and later marks the assistant's turn `complete`, `error`, or `canceled`. Two small mechanisms stop double-sends. An **idempotency key** is a unique ID per send, so a retried click doesn't create a second turn. An **expected revision** is the client saying "I'm replying to version 7 of this conversation", so two tabs can't silently fork it. Neither requires holding a database transaction open for the 10 seconds of generation.

Replies stream over **Server-Sent Events (SSE)**, a simple one-way HTTP streaming format. Once tokens are on the user's screen, you can't silently retry: a fresh generation would say something different, and the user would see the answer rewrite itself. Mark the turn as partial and let the client reconcile against the stored history. A retry *before* any token is shown is fine if the deadline allows it.

Routing each conversation to the same inference host can reuse its cached prompt prefix and cut prefill work. That's a nice speedup, but it's a *performance* hint, not a source of truth. When the host dies, the new one starts cold and still produces a correct answer from Postgres history. *This exact pattern is built in* **AI farm, part 3: routing and keeping conversations consistent**, *on the two-server layout from* **AI farm, part 1: designing a two-server Qwen deployment**.

What you **don't** draw matters too. Don't put Kafka between chat and inference "for scale". A synchronous streaming request doesn't need a durable log, and adding one creates delivery problems you then have to solve. And since the assistant can't take actions, there are no tool credentials to scope. If the interviewer adds "let it file expense reports", that's a new trust boundary. *See* **Agents and tools: from model decisions to safe actions** *and* **LLM security: treating model output as untrusted**. One security point applies even now: retrieved documents are *evidence*, not *instructions*. A policy PDF that says "ignore previous rules" gets no authority.

> 🎬 **Animation — one request through the ownership map:** animate the box diagram above. Step 1: a request dot enters the gateway; the API box lights up and a row "turn 8: user, pending" appears in Postgres. Step 2: the dot visits retrieval; three document cards fly back, and a fourth card stamped "HR-confidential" is visibly blocked by a permission filter. Step 3: the dot reaches inference host #2; tokens stream back through the API to the browser as a growing text line. Step 4: host #2 flashes red mid-stream; the browser shows "partial answer, retry?" and Postgres marks turn 9 "error". Step 5: the retry goes to host #3 (cold cache icon), rebuilds from Postgres history, and succeeds. Caption: "the cache died; the conversation didn't".

## Minutes 22–27: does it fit on a GPU? {#napkin-sizing}

Now the interviewer asks "how many GPUs?" Don't guess. Do the napkin math for **one replica**. A **replica** is one complete copy of the model that can serve requests on its own, possibly spread over several GPUs. We'll assume a hypothetical dense 8-billion-parameter model on one illustrative 80 GB GPU.

**Step 1: weights.** In bf16, a 16-bit format, each parameter takes 2 bytes.

```text
 8 × 10⁹ params × 2 bytes = 16 GB of weights
 80 GB card − 16 GB weights − ~8 GB runtime/activations ≈ 56 GB for KV cache
```

**Step 2: KV cache.** During decode, the model keeps each past token's attention keys and values so it doesn't recompute them. That's the **KV cache**. Its size per token is:

```formula
KV bytes per token = 2 × L × h_kv × d_head × s
```

**2** is one key plus one value. **L** is the layer count. **h_kv** is the number of key/value heads. **d_head** is the width of each head. **s** is bytes per number. With L = 32, h_kv = 8, d_head = 128, s = 2:

```text
 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token

 average request at its end: 4,000 in + 400 out = 4,400 tokens
   4,400 × 131,072 = 576,716,800 bytes ≈ 0.58 GB
   56 GB ÷ 0.58 GB ≈ 97 average sequences fit

 worst case (12,000 in + 600 out = 12,600 tokens)
   12,600 × 131,072 ≈ 1.65 GB  →  56 ÷ 1.65 ≈ 33 worst-case sequences fit
```

Little's law said ~66 decode sequences fleet-wide at the burst. Spread over 5 replicas, that's about 13 each. Even 33 worst-case sequences is plenty, so **memory isn't our bottleneck**. That's worth saying explicitly, because at long contexts it often is. *The formula and paging tricks are in* **What happens at inference: prefill, decode, and the KV cache**; *grouped-query attention (the reason h_kv is 8, not 32) is in* **Making attention cheaper: GQA, FlashAttention, and long context**.

**Step 3: decode speed.** Each decode step streams all the weights plus every active sequence's KV cache out of GPU memory once, and produces one token for every sequence in the batch. Take ~13 sequences averaging ~4,200 cached tokens mid-answer (≈ 0.55 GB each) and an illustrative 3 TB/s of memory bandwidth (check your card's datasheet):

```text
 bytes per step ≈ 16 GB weights + 13 × 0.55 GB KV ≈ 23.2 GB
 time per step  ≈ 23.2 GB ÷ 3,000 GB/s ≈ 7.7 ms
 ceiling        ≈ 1 ÷ 0.0077 s ≈ 130 tokens/s per stream
```

Real kernels reach only part of peak bandwidth, but even at 60% we'd get about 78 tokens/s. That's above our 50 tokens/s requirement. ✔

**Step 4: prefill.** A forward pass costs about 2 FLOPs per parameter per token (one multiply and one add per weight). Assume an illustrative 1,000 TFLOP/s bf16 peak, of which 40% (400 TFLOP/s) is usable:

```text
 per token:    2 × 8 × 10⁹       = 16 GFLOP
 per request:  4,000 × 16 GFLOP  = 64 TFLOP  →  64 ÷ 400 = 0.16 s of prefill
 burst total:  32,000 tok/s × 16 GFLOP = 512 TFLOP/s fleet-wide
               ÷ 5 replicas ≈ 102 TFLOP/s each ≈ 26% of usable compute
```

0.16 s leaves most of the 2 s TTFT budget for queueing and retrieval. ✔

So on paper one replica looks like it could do *more* than 2 requests/s. But napkin math gives a **ceiling**, not a plan. It ignores prefill stalling decode when they share a batch, long-prompt tails, scheduler overhead, and p95 rather than mean behavior. Now suppose a load test (hypothetical, for this exercise) shows one replica holds **2 requests/s** for this exact traffic mix while meeting both p95 targets. That measured number is the one you size with.

If the model didn't fit, you'd have levers, each with a catch. **Quantization** (fewer bits per weight) shrinks memory but needs a quality re-check. **Tensor parallelism** (splitting each layer across GPUs) makes big models fit but adds communication. A **mixture-of-experts** model computes with only some parameters per token but must keep all of them in memory. *See* **Quantization: spending fewer bits per weight**, **Distributed training: fitting a training run onto a cluster**, *and* **Mixture of experts: more parameters, same compute per token**.

> 🎬 **Animation — filling one 80 GB GPU:** a vertical bar representing 80 GB of GPU memory. Step 1: a 16 GB block labeled "weights (8B × 2 bytes)" fills the bottom. Step 2: an 8 GB block "runtime" stacks on top. Step 3: 0.58 GB slivers labeled "request" drop in one at a time; a counter ticks to 13 (typical load, bar mostly empty), then keeps going to 97 (bar full). Step 4: reset and drop 1.65 GB "worst-case" slivers; counter stops at 33. Step 5: a side panel shows a decode step as a sweep reading 23.2 GB at 3 TB/s → 7.7 ms. Caption: "memory fits; bandwidth sets the speed".

## Minutes 27–31: buying capacity that survives a failure {#capacity}

We need 8 req/s at the burst. Each replica is validated at 2 req/s. The naive answer is 8 ÷ 2 = 4 replicas. The senior answer asks: **what happens when one dies during the burst?**

```formula
utilization after failure = λ_peak ÷ ((N − f) × r)
```

**λ_peak** is the peak arrival rate, **N** is the replica count, **f** is how many replicas you assume can fail, and **r** is validated throughput per replica.

| Replicas N | Capacity after 1 loss | Utilization at 8 req/s | Verdict |
|---|---|---|---|
| 4 | 3 × 2 = 6 req/s | 133% | Overloaded: queue grows without bound |
| 5 | 4 × 2 = 8 req/s | 100% | Zero margin: any variance spills into latency |
| 6 | 5 × 2 = 10 req/s | 80% | Survives the loss with some slack |

Why not run at 100%? Queueing theory says waiting time explodes as utilization approaches 100%, because random bursts have no slack to drain into. *The queueing math and admission control are in* **Serving many users: batching, scheduling, and speculative decoding**.

**"Can't we just queue the burst?"** Good interviewers ask this. Check it with numbers. With 4 replicas and one dead, capacity is 6 req/s against 8 arriving. The backlog grows by 2 req/s × 600 s = 1,200 requests by the end of the burst. Draining at 6 req/s, the last one waits 1,200 ÷ 6 = **200 seconds** for its first token. That blows the 2 s TTFT target by 100×. So the choice is either buy headroom, or **shed load deliberately**: return HTTP 429 ("too many requests") with a retry-after, shorten max output during overload, or send public-document questions to the approved API. Never let requests wait forever.

The 80% figure is only as good as its assumptions. It assumes identical replicas and no **shared bottleneck**. If all six replicas hit the same retrieval service or the same Postgres, those need the same failure analysis.

**Load testing that doesn't lie.** Use **open-loop** tests: requests arrive on a fixed schedule no matter how fast responses come back. A **closed-loop** tester (N virtual users, each waiting for its answer before asking again) quietly slows down when your system slows down, which hides exactly the overload you're trying to find. Replay realistic *joint* distributions of prompt length, output length, and arrival bursts. Include cold caches, one killed replica, slow retrieval, cancellations, and database pressure. Record SLO-compliant throughput, rejection rate, queue time, TTFT, and per-stream decode speed. When reporting decode speed, know the difference between vLLM's two metrics: **time per output token** is one number per finished request, while **inter-token latency** is one sample per streamed chunk. They're weighted differently and can disagree.

> 🎬 **Animation — four, five, or six replicas:** three rows of GPU icons (4, 5, 6). An arrival meter at the top reads "8 req/s". Step 1: all healthy; each row's load bar is below 100%. Step 2: one GPU in each row turns red. Row 4: load bar reads 133%, and a queue tube beside it fills steadily, with a timer counting to "wait 200 s" at the end of the 10-minute burst. Row 5: bar at exactly 100%, jittering into red at every random spike. Row 6: bar at 80%, spikes absorbed. Caption: "size for the peak with one replica gone".

## Pricing successful tasks, not tokens {#economics}

Next question: *"This seems expensive. Is it worth it?"* The right unit is **cost per successful task**, not price per token.

```formula
cost per successful task = total attributable cost in a window ÷ successful tasks in the same window
```

"Attributable cost" means everything, over the same window: GPUs (including idle reserved hours), retrieval, storage, networking, retries, human review, and on-call time. "Successful" has to use the same rubric for every candidate.

Why this matters, with invented numbers. System A costs 240 units for 1,000 attempts and succeeds 800 times: 240 ÷ 800 = **0.30 per success**. System B is "cheaper" at 200 units, but succeeds only 500 times: 200 ÷ 500 = **0.40 per success**. The cheap system is 33% more expensive per useful answer, before counting the employees who gave up and emailed HR anyway.

Our fleet, with an illustrative 2.50 units per GPU-hour:

```text
 always-on:  6 GPUs × 720 h/month = 4,320 GPU-h × 2.50 = 10,800 units
 successes:  440,000 requests × 90% success          = 396,000
 GPU cost per success: 10,800 ÷ 396,000 ≈ 0.027 units

 scheduled:  6 GPUs × 220 working h (22 days × 10 h) = 1,320 GPU-h
             2 GPUs × 500 other h                   = 1,000 GPU-h
             2,320 GPU-h × 2.50 = 5,800 units   (46% less)
```

Here's the senior observation: average load is 0.69 req/s, and we bought for 8 req/s with a failure, so the fleet sits mostly idle. Scaling down off-hours is the obvious win. Two replicas at night still survive one failure at night-time load, as long as it stays under 2 req/s. The deeper question is whether the 10-minute burst deserves six GPUs all day. That's a product conversation ("is 10 minutes of degraded answers after an all-hands acceptable?"), and saying so shows judgment.

A **cascade** is another lever: try a small, cheap model first and escalate hard questions to a bigger one, like a help desk where tier one handles routine tickets. With illustrative per-attempt costs of 0.01 (small) and 0.03 (large), and 30% of questions escalated, the average cost is 0.01 + 0.30 × 0.03 = **0.019** instead of 0.03. But the escalated 30% pay double latency, and a small model that answers confidently when it *should* have escalated costs you quality, not money. Measure the escalation rate and misrouting on held-out traffic before you believe the savings.

**Retries are a hidden cost multiplier.** If the browser, the API, and the inference client each retry 3 times, one failing request can become 3 × 3 × 3 = **27** attempts, exactly when the system is already struggling. Retry at one layer only, cap the attempts, pass the remaining deadline downstream so no one works on a request the user has abandoned, and use exponential backoff with **jitter** (random spread in retry timing). AWS's analysis of jittered backoff showed that randomizing the waits spreads retry spikes into a roughly even rate and substantially cuts total calls.

> 🎬 **Animation — cheap per token, expensive per success:** two cash registers, A and B. Step 1: A takes 240 coins for 1,000 attempts; 800 green checkmarks pop out. B takes 200 coins; 500 checkmarks pop out, and 500 red crosses fall into a bin. Step 2: each register divides coins by checkmarks: A shows 0.30, B shows 0.40. Step 3: B's red crosses turn into retry arrows that feed back into B's register and add coins. Caption: "divide by successes, not attempts".

## Minutes 31–39: shipping changes you can take back {#release}

LLM regressions are sneaky. A new prompt or model can make answers worse without throwing a single error, so your release process is part of the design.

**Evaluation set.** Build a versioned set that covers common questions, rare policies, questions with **no** answer in the docs (to test abstention), conflicting policy revisions, long histories, users *without* access to the relevant document (to test permission compliance), and adversarial documents. Keep the development split separate from the final test split, and look at **slices**, not just the average. A model that goes from 91% to 93% overall while dropping from 99% to 95% on permission cases has *failed*. Use cheap deterministic checks where possible (does the cited passage exist? is it one the user may see?), expert review for correctness, and an **LLM judge** (another model grading answers) only once it has been calibrated against human labels. *See* **Evaluation and observability: knowing whether it actually works**.

**The release bundle.** Version everything that changes behavior **together**: model weights, tokenizer, prompt template, retrieval index and embedding model, permission logic, and decoding settings like temperature. If you only version the model, you can't tell which of five changes caused the regression, and you can't cleanly roll back.

```text
 bundle v14 ──► offline eval ──► load + failure test ──► canary 5% ──► 100%
                   │ gate            │ gate                 │ gate
                   ▼                 ▼                      ▼
                 stop              stop              roll back to v13
                                                   (but keep TODAY's
                                                    permission revocations)
```

**Canary.** Release to a small slice of eligible users first. The name comes from miners' canaries, an early warning. Infrastructure metrics like TTFT and error rate show up in minutes. Quality needs labels, and labels are slow. Suppose you can label 200 canary answers a day at ~90% success. The standard error of one day's estimate is √(0.9 × 0.1 ÷ 200) ≈ 0.021, so ±4.2 points at 95% confidence. Comparing canary with baseline adds the two uncertainties, which makes it worse. After two days (400 labels per side), each estimate's standard error is √(0.09 ÷ 400) = 0.015, and the difference's standard error is √2 × 0.015 ≈ 0.021, or ±4.2 points. So detecting a 5-point drop takes **a few days**, not an afternoon. Promote slowly on quality even when latency looks great on day one.

**Rollback.** Keep the previous bundle deployable and rehearse the rollback. There's one trap: restoring last week's retrieval index must **not** restore last week's permissions. If someone lost access to a document yesterday, rolling back can't give it back. Permissions are enforced from the live source of truth, not baked into the index version.

> 🎬 **Animation — a canary that catches a slow regression:** two lines over a 5-day x-axis: baseline task success (flat at 90%) and canary (true value 85%). Step 1: day 1 shows the canary point at 88% with a wide error bar (±4 points) overlapping the baseline, labeled "can't tell yet". Step 2: days 2–3 add points and the error bars narrow. Step 3: by day 3 the bars separate; a gate icon turns red and "roll back v14 → v13" appears. Step 4: an inset shows a document permission revoked on day 2 staying revoked after rollback.

## Minutes 39–45: the curveball and the defense {#defense}

The last minutes test whether your design is **testable** rather than something you're just loyal to. Two moves do this.

**Move 1: handle the change through the ledger.** Say the interviewer asks: *"Security now allows restricted documents on one approved cloud API. Do you switch?"* Don't redesign. Walk the ledger. D1 (self-host) was justified by the data boundary, and that justification just weakened. So run the API on the *same* evaluation set, including the permission and abstention slices, and compare cost per successful task and p95 latency. Maybe keep a hybrid: the API absorbs the burst, and self-hosted capacity covers the baseline, which could cut the fleet from six replicas to fewer. D2 (RAG) and the release process don't change at all. That's the payoff of choosing on independent axes.

**Move 2: name what would change your mind.** A senior defense states the constraint, the decision, the evidence, the downside, and the observation that would reverse it. For diagnosis, have this table ready:

| Symptom | Likely cause | Where to look |
|---|---|---|
| Right passage not retrieved (low recall on labeled queries) | Chunking, embeddings, index freshness | Retrieval, not the model |
| Right passage retrieved, answer still wrong | Model or prompt | Prompting, then fine-tuning |
| TTFT high, decode speed steady | Queueing or prefill | Admission, chunked prefill, capacity |
| TTFT fine, decode slowing | KV memory pressure or too-large batches | Batch limits, context caps |
| Smaller quantized model passes the rubric | Economics can improve | Only if it also passes the critical slices |

The last row captures the whole philosophy. Memory savings are worthless if the model fails the permission slice. Every lever—model size, quantization, caching, cascades—gets judged by the same three contracts you wrote in minute three.

A strong closing sentence ties it together: *"The biggest remaining uncertainty is whether one replica really sustains 2 req/s on our real prompt-length tail; the first experiment I'd run is an open-loop load test with a replayed week of traffic and one replica killed mid-burst."* That's an interviewer's favorite ending: a specific uncertainty paired with the experiment that resolves it.

# Interview

## Question

Design the internal policy assistant above in 45 minutes. After you finish, the interviewer makes two changes: average retrieved context doubles (they believe it improves answers), and any one inference **host**, a physical machine that runs two replicas, must be removable during working hours for maintenance. What changes, and by how much?

## Answer

Start by protecting the contracts. Before accepting "doubling context improves answers", ask for evidence on the held-out set. More context also means more room for distraction and more cost, so I'd want the quality gain measured, and I'd test reranking or passage compression as a cheaper way to get it.

**Token math.** Only the retrieval component doubles: 500 + 300 + 1,200 + 4,000 = **6,000** input tokens (up from 4,000, a 1.5× increase, not 2×). Peak input demand becomes 8 × 6,000 = **48,000 tokens/s**. Output demand stays 3,200 tokens/s. Prefill per request becomes 6,000 × 16 GFLOP = 96 TFLOP, or about 0.24 s at 400 usable TFLOP/s, which still fits the 2 s TTFT budget. KV per average request becomes 6,400 × 128 KiB ≈ 0.84 GB, so a 56 GB pool holds about 66 average sequences. That's still above the ~13 per replica we need, so memory remains fine. The per-step decode read grows to roughly 16 + 13 × 0.81 ≈ 26.6 GB, so decode slows a bit.

**The old 2 req/s benchmark is now invalid.** It was measured on a different mix. I'd re-run the load test. Suppose, hypothetically, it now shows 1.6 req/s per replica.

**Failure unit changes from replica to host.** Losing a host now removes two replicas at once. To serve 8 req/s at 1.6 req/s each, I need 8 ÷ 1.6 = 5 surviving replicas at 100% utilization. With 2 replicas per host, 3 hosts (6 replicas) minus one host leaves 4 replicas, or 6.4 req/s: overloaded. 4 hosts (8 replicas) minus one leaves 6 replicas, or 9.6 req/s, so 8 ÷ 9.6 ≈ **83%** utilization. That's my answer: 4 hosts. Planned maintenance also means I can drain a host gracefully: stop routing new requests to it and let in-flight streams finish.

Everything else stays: bounded admission with explicit 429s, partial-generation status instead of silent retries, and restricted requests never failing over to an unapproved provider. The new configuration ships as a new bundle through the same gates (quality slices, open-loop load test with one host removed, canary), with the previous bundle ready for rollback.

## Follow-ups

- The product team wants a three-step agent (search, draft, verify). How do input tokens per task, latency, and cost per successful task change?
- Which observations tell you retrieval failed rather than the generator?
- Why doesn't doubling the replica count halve p95 latency?
- The cloud API beats your local model on restricted questions. What do you do?
- How do you roll back a retrieval index without restoring revoked document permissions?

# Pitfalls

- **Inferring request rate from headcount.** 5,000 employees says nothing until you know daily actives, questions per user, and synchronized bursts.
- **Adding input and output tokens/s into one number.** Prefill is limited by compute and decode by memory bandwidth; a single sum hides which one is your bottleneck.
- **Multiplying peak rate by p95 latency and calling it p95 concurrency.** Little's law gives an average, for a boundary you have to name.
- **Treating "local vs API" and "RAG vs fine-tuning" as one choice among four architectures.** They're independent axes; a self-hosted fine-tuned model can also use retrieval.
- **Presenting napkin math as a benchmark.** Napkin math gives a ceiling; size the fleet with a measured, labeled load-test number.
- **Sizing for the peak with every replica healthy.** Size for the peak with your failure unit (a replica or a whole host) removed.
- **Using closed-loop load tests only.** They slow their own arrivals when you slow down, so they hide overload.
- **Letting cache affinity carry correctness,** or silently retrying after tokens are visible. Both lead to contradictory answers after a failover.
- **Counting HTTP 200s as success.** A fast, well-formed answer citing the wrong policy is a failure.

# Checklist

- Write the user task, exclusions, data boundary, and three separate contracts before naming a model.
- Convert headcount into busy and burst request rates, then into input and output tokens/s.
- Apply Little's law at a named boundary and derive per-stream decode speed from the latency budget.
- Size one replica: weights, KV cache per token and per request, decode step time, prefill time.
- Compute utilization after losing your failure unit, and explain why queueing the burst doesn't work.
- Compare designs by cost per successful task, including idle capacity and retries.
- Describe the release bundle, the gates, how long a quality canary must run, and a rollback that keeps revocations.
- Close with the biggest uncertainty and the experiment that resolves it.

# Sources

- [Google SRE Workbook: Implementing SLOs](https://sre.google/workbook/implementing-slos/) — Customer-centred SLOs, rolling measurement windows, and error budgets as 100% minus the SLO; the numeric targets in this article are hypothetical.
- [vLLM documentation: Metrics](https://docs.vllm.ai/en/latest/design/metrics/) — Definitions of TTFT, queue time, end-to-end latency, per-request time per output token, and per-chunk inter-token latency, and why they aggregate differently.
- [Little's law (Wikipedia)](https://en.wikipedia.org/wiki/Little%27s_law) — L = λW for stable systems, independent of arrival and service distributions.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — Combines parametric (in-weights) memory with a non-parametric retrieved index; no result from it is transferred to this design.
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — Frozen pretrained weights plus trainable low-rank matrices in each layer.
- [Exponential Backoff and Jitter (Marc Brooker, AWS Architecture Blog, 2015)](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) — Jittered backoff spreads retry spikes into a roughly constant rate and cuts total calls versus un-jittered backoff.
- [NIST AI 600-1: Generative Artificial Intelligence Profile (2024)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) — Companion to the NIST AI Risk Management Framework for managing generative-AI risks across the lifecycle.

# Flashcards

## design-contract

**Q:** What should come before choosing a model in an LLM system design interview?

Three separate contracts: product quality (task success on a held-out set, including citations, permissions, and abstention), service reliability (TTFT, completion latency, availability under declared load), and data policy (where data may be processed, retention, and who sees traces). They decide which models are even eligible and what evidence can compare them. Passing one says little about the others: a bigger model can raise quality and break latency, and an API can pass both and violate data policy.

## design-rates

**Q:** Why estimate input and output tokens/second separately?

Because they load different hardware limits. Prefill processes the whole prompt in one parallel pass and is limited by compute (FLOPs). Decode generates one token at a time and re-reads the weights and KV cache from memory at every step, so it's limited by memory bandwidth. At 8 req/s with 4,000 input and 400 output tokens, that's 32,000 input tokens/s and 3,200 output tokens/s. Adding them into 35,200 hides which phase is your bottleneck.

## design-little

**Q:** At 8 requests/second and 10 seconds average time in the system, what does Little's law give you, and what doesn't it give you?

L = λW = 8 × 10 = 80 requests in the system on average, provided the system is stable. It does not say 80 GPU slots are needed (some requests are queued or in retrieval; the GPU boundary here is about 8 × 8.2 ≈ 66), and it isn't a p95 figure. Always name the boundary you're applying it to.

## design-context

**Q:** Which tokens belong in the input budget, and why reserve output space?

Everything serialized into the prompt: system instructions, tool schemas, the question, conversation history, and retrieved passages (500 + 300 + 1,200 + 2,000 = 4,000 in the example). Averages size throughput. Caps protect the system: an input cap (e.g. 12,000) plus reserved output (600) must fit the context window, and it bounds worst-case KV memory and prefill time.

## design-axes

**Q:** Why aren't "local model", "API", "RAG", and "fine-tuning" four alternative architectures?

They answer different questions. Hosting decides where computation runs. Retrieval decides where facts come from. Fine-tuning shapes behavior. You pick on each axis independently: a self-hosted, LoRA-tuned model can also use retrieval, and a hybrid can send public-data questions to an API while restricted ones stay local.

## design-rag

**Q:** Why choose retrieval for an assistant over frequently changing company policies?

Facts live in a versioned external index, so updating a policy means re-indexing a document, not retraining. Answers can cite a specific version, and permissions can be enforced per user at retrieval time. Retrieval still has to be evaluated separately for recall, freshness, ranking, and injection-laden documents, and retrieved text is evidence, never instructions.

## design-lora

**Q:** When is LoRA fine-tuning a defensible next step?

When labeled examples show a stable, recurring behavior gap (such as a wrong citation format) that better prompts and retrieval didn't fix. LoRA freezes the base weights and trains small low-rank adapters, which is cheap in parameters. But data preparation, regression evals, and serving compatibility still cost effort, and fine-tuning never supplies fresh facts or enforces permissions.

## design-kv-fit

**Q:** For a model with L = 32, h_kv = 8, d_head = 128 in bf16, how big is the KV cache for a 4,400-token request, and how many fit in 56 GB?

Per token: 2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes (128 KiB). Per request: 4,400 × 131,072 ≈ 0.58 GB. 56 ÷ 0.58 ≈ 97 average sequences, or about 33 worst-case 12,600-token sequences. If you only need ~13 per replica, memory isn't the bottleneck and decode bandwidth sets speed.

## design-headroom

**Q:** Six replicas, each validated at 2 req/s, serve an 8 req/s peak. What's utilization after one fails, and why not use five?

8 ÷ (5 × 2) = 80%. With five replicas, one failure leaves exactly 8 req/s of capacity: 100% utilization, where queueing delay explodes on any random spike. The arithmetic assumes identical replicas, a measured per-replica rate, and no shared bottleneck like retrieval or the database.

## design-burst-queue

**Q:** Why can't you just queue a 10-minute 8 req/s burst on 6 req/s of capacity?

The backlog grows at 8 − 6 = 2 req/s, reaching 2 × 600 = 1,200 requests by the end. Draining at 6 req/s, the last request waits about 1,200 ÷ 6 = 200 s for its first token, 100× over a 2 s TTFT target. So either buy headroom or shed load explicitly (429 with retry-after, shorter outputs, an approved overflow provider).

## design-open-loop

**Q:** Why must load tests be open-loop?

Open-loop tests send requests on a fixed schedule regardless of response time, just like real users who don't know your system is slow. Closed-loop testers wait for each answer before sending the next, so they automatically slow down when the system slows, hiding the overload you're trying to measure.

## design-cost

**Q:** Why compare cost per successful task instead of price per token?

Cheap attempts that fail, retry, or escalate aren't cheap. System A: 240 units ÷ 800 successes = 0.30. System B: 200 ÷ 500 = 0.40, so the "cheaper" system costs a third more per useful answer. Count all attributable costs (idle GPUs, retrieval, retries, review) over the same window, and use the same success rubric for every candidate.

## design-state

**Q:** Why isn't sticky routing to a warm inference cache a form of conversation consistency?

Caches are disposable performance state. Correctness (stored turns, revisions, idempotency keys, authorization) lives in the application and database, so when a host dies the next one rebuilds from durable history and still answers correctly. It's just slower, because the cache is cold.

## design-partial

**Q:** What changes once generated tokens have reached the user's screen?

You can no longer retry silently. A new generation would produce a different continuation, so the visible answer would contradict itself or be duplicated. Mark the turn partial/error, show that to the user, and reconcile against stored history. Retrying before any token is shown is fine within the deadline.

## design-bundle

**Q:** What belongs in an LLM release bundle, and what must rollback not undo?

Model weights, tokenizer, prompt template, retrieval index and embedding model, permission logic, and decoding settings, versioned together so a regression can be attributed and reversed as one unit. Rollback must not restore revoked permissions: access is enforced from the live source of truth, not from an old index snapshot.

## design-defense

**Q:** What makes a tradeoff defense senior-level?

It states the constraint, the decision, the evidence, the downside, and the observation that would reverse it, e.g. "self-host because restricted documents can't leave; if security approves an API, I'd rerun the same eval slices and compare cost per success." It ends with the biggest remaining uncertainty and the experiment that resolves it. That shows the design is testable, not a loyalty to a framework.
