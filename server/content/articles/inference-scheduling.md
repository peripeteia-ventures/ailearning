---
{
  "slug": "inference-scheduling",
  "title": "Serving many users: batching, scheduling, and speculative decoding",
  "category": "inference",
  "summary": "How an inference server shares a GPU between hundreds of users: continuous batching, chunked prefill, admission control, queueing arithmetic, speculative decoding, and splitting prefill and decode onto separate machines.",
  "difficulty": "Systems",
  "minutes": 30,
  "prerequisites": ["transformer-foundations", "serving-kv-cache"],
  "learningObjectives": [
    "Explain why static batching wastes a GPU and how continuous (iteration-level) batching fixes it.",
    "Describe how a long prompt stalls other users' streams, and how chunked prefill and a per-iteration token budget bound that damage.",
    "Use Little's law and a utilization argument to size a serving fleet from arrival rate, output length, and latency targets.",
    "Calculate the expected speedup of speculative decoding from the acceptance rate, and explain why the exact acceptance rule keeps outputs unbiased.",
    "Decide when prefill/decode disaggregation pays for its KV-transfer cost, and defend the decision with the right load test."
  ]
}
---

# Sections

## The scheduler is the head chef {#the-job}

Picture a restaurant kitchen with one enormous stove. Orders arrive at random. Some are a single espresso, some are a banquet for forty. The head chef doesn't cook; the head chef decides **what goes on the stove next**. Get that wrong and one banquet order makes every espresso customer wait twenty minutes, even though the stove never stops being busy.

An LLM inference server has exactly this problem. One GPU (or a group of GPUs) holds the model weights, and many users' requests share it. The component that decides which requests' tokens get computed on the next pass through the model is the **scheduler**. This article is about how good schedulers work and how you reason about them in an interview.

Two quick recaps first, because everything here depends on them. Generating a reply has two phases:

- **Prefill.** The model reads the whole prompt in one go. Every prompt token is known up front, so all of them are processed in parallel. This is big, dense matrix math, and it tends to keep the GPU's arithmetic units busy (people say it's **compute-bound**).
- **Decode.** The model then produces the reply one token at a time, because each new token depends on the one before it. Each step does very little arithmetic per sequence but has to stream all the weights through the chip, so it tends to be limited by memory bandwidth (**memory-bound**).

During prefill the model saves, for every token and every layer, the attention **keys and values** it computed. That store is the **KV cache**, and decode reuses it so it never has to recompute the past. The KV cache lives in GPU memory, grows by one token per sequence per decode step, and is usually what limits how many users you can hold at once. We go through both phases and the cache properly in **What happens at inference: prefill, decode, and the KV cache**; here we just need the picture.

```text
 request:  "Summarize this 2,000-token doc..."
            │
            ▼
 ┌────────────────────┐    ┌──┐┌──┐┌──┐┌──┐┌──┐
 │ PREFILL            │ →  │t1││t2││t3││t4││t5│ ...  DECODE
 │ all 2,000 tokens   │    └──┘└──┘└──┘└──┘└──┘      one token per step,
 │ in one big pass    │     each step reads weights   each step appends to
 │ writes KV cache    │     + whole KV cache          the KV cache
 └────────────────────┘
   compute-heavy             bandwidth-heavy, repeated hundreds of times
```

### What "good" means: the latency vocabulary

You can't design a scheduler until you know what you're protecting. The standard numbers:

| Metric | Plain meaning | What the user feels |
|---|---|---|
| **TTFT** (time to first token) | arrival → first generated token appears | "Is it thinking or broken?" |
| **TPOT** (time per output token) | (time of last token − time of first token) ÷ (N − 1), for N output tokens | how fast the text streams |
| **ITL** (inter-token latency) | each individual gap between consecutive tokens | stutters and freezes |
| **Throughput** | total output tokens per second across all users | your cost per token |
| **Goodput** | requests per second that finished **within** their latency targets | what you actually sell |

TTFT includes time spent **waiting in a queue**, not just prefill compute, plus sampling and network. TPOT is an average over the whole reply, which is exactly why you also look at individual gaps: a reply that streams at 20 ms per token but freezes for one full second in the middle has a fine TPOT and a terrible user experience.

A **service-level objective (SLO)** is a latency target you commit to for a class of traffic, for example "p95 TTFT under 500 ms and p95 TPOT under 50 ms for chat". **p95** means the 95th percentile: 95% of requests do at least this well. We care about tails because the slow 5% are the users who complain.

Two things interviewers like to poke at. First, **percentiles don't add**: p95 queue time plus p95 prefill time is not p95 TTFT, because the request that waited longest is usually not the one with the longest prompt. Measure the end-to-end distribution directly. Second, **throughput alone is a trap**: a server can post higher tokens per second by rejecting hard requests or quietly timing them out. Always report throughput next to rejections, timeouts, and goodput.

> 🎬 **Animation — the kitchen and the stove:** a single wide "GPU stove" in the middle, a queue of order tickets on the left (three small espresso tickets labelled 50-token prompts, one large banquet ticket labelled 8,000-token prompt). Step 1: the chef puts the banquet on the stove alone; a timer over each espresso ticket counts up in red while the stove shows "100% busy". Step 2: rewind; the chef slices the banquet into four pieces and interleaves an espresso between each slice; the espresso timers stay green while the stove is still 100% busy. Caption at the end: "Same utilization, very different waiting."

## Why the obvious batching wastes most of the GPU {#static-vs-continuous}

GPUs are fast when they do many things at once. During decode, one sequence alone barely uses the arithmetic units: the chip spends its time reading weights from memory, and then does only a sliver of math with them. If you process 32 sequences in the same step, you read the weights **once** and use them 32 times. That's why serving is all about **batching**: stacking several sequences' work into one forward pass. A **batch** of `B` sequences shares one pass; one **iteration** (or step) is one forward pass that advances every sequence in the batch by one token (or by a chunk of prompt, as we'll see).

### Static batching: the charter bus

The naive approach, borrowed from how you'd batch images, is **static batching**: collect `B` requests, run them together until **all** of them finish, then collect the next `B`. It's a charter bus that won't let anyone off, or on, until the last passenger reaches their stop.

The problem is that output lengths vary wildly. Tiny example: a batch of 4 slots, where the replies turn out to be 10, 50, 20, and 100 tokens long.

```text
 step:        0        10       20                50                       100
 slot A  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   (10 useful)
 slot B  ██████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   (50 useful)
 slot C  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   (20 useful)
 slot D  ██████████████████████████████████████████████████████████████████   (100 useful)
         █ = generating a real token   ░ = slot held but idle
```

The batch runs for 100 steps × 4 slots = 400 slot-steps, but only 10 + 50 + 20 + 100 = 180 of them produce a token. That's **45% useful work**. Worse, request A finished at step 10 but can't return to the user until step 100 if the system only releases results per batch, and new arrivals sit in the queue the whole time.

### Continuous batching: the city bus

The fix, introduced as **iteration-level scheduling** in the Orca paper (OSDI 2022), is to let the scheduler change the batch's membership **between every iteration**. When a sequence emits its end token, it leaves immediately and its result streams back. A waiting request can take its slot at the very next step. Sequences that are still running keep their KV cache and carry on as if nothing happened. This is now usually called **continuous batching**, and it's the default in modern engines like vLLM, SGLang, and TensorRT-LLM.

It's a city bus: people get off and on at every stop, and the bus never waits for the whole group. In the example above, when A finishes at step 10, a new request E moves into slot A at step 11. The idle `░` cells fill with real work.

One subtlety: the batch is rebuilt **at iteration boundaries**, not by interrupting a GPU kernel halfway. Also, sequences of different lengths can't simply be stacked into one rectangular tensor for attention (each has its own history). Orca handles this with **selective batching**: batch the operations that don't care about sequence length (the big weight matmuls, where every token is just a row) across all sequences, and run attention per sequence. Paged KV memory (covered in **What happens at inference: prefill, decode, and the KV cache**) made this practical at scale because a new sequence doesn't need one big contiguous memory region.

The Orca authors report up to 36.9× throughput over NVIDIA FasterTransformer at the same latency on their workloads. Treat that as "this was a big deal", not as a number to quote for your system; the gain depends entirely on how variable the output lengths are.

> 🎬 **Animation — charter bus vs city bus:** two 4-row grids of 100 columns, one labelled "static", one "continuous". Use the reply lengths A=10, B=50, C=20, D=100. Step 1: fill the static grid column by column; finished rows turn grey and stay grey until column 100; a counter shows "useful 180 / 400 = 45%". Step 2: fill the continuous grid; when A ends at column 10, a new coloured request E slides into row A at column 11; when C ends at 20, F slides in; and so on. Step 3: show both grids side by side with the grey area highlighted, and a queue icon on the left that stays full for static and drains for continuous.

## What the scheduler actually juggles every step {#scheduler-state}

Once requests can join and leave every iteration, the scheduler turns into a little state machine. Every iteration it asks: who is running, who's waiting, what fits?

```text
            admit                 prompt done              end token / max length
 WAITING ──────────► PREFILLING ─────────────► DECODING ─────────────────────────► DONE
    ▲                 (maybe in chunks)          │   │
    │    preempt (free KV, recompute later)      │   │ client disconnects / deadline passes
    └────────────────────────────────────────────┘   └──────────────────────────► CANCELLED
```

Each running request owns KV cache blocks. Every transition has to agree with that ownership: when a request finishes or is cancelled, its blocks are freed; when memory runs out mid-generation, the scheduler **preempts** someone, which means pausing a running request and freeing (or swapping out) its KV cache so others can continue. The preempted request later has its KV cache recomputed or swapped back in. The details of preemption and eviction belong to **What happens at inference: prefill, decode, and the KV cache**; what matters here is that preemption is expensive, so a good scheduler avoids getting into that corner.

### Three separate limits, not one

A new request fits only if it passes **all** of these, and passing one says nothing about the others:

| Limit | What it counts | Typical knob (vLLM names) |
|---|---|---|
| **Sequence slots** | how many sequences run concurrently | `max_num_seqs` |
| **Per-iteration token budget** | how many **new** token positions (prompt chunks + one per decoding sequence) are processed in one forward pass | `max_num_batched_tokens` |
| **KV cache memory** | total tokens of **history** held for all resident sequences | set by GPU memory left after weights |

People mix up the second and third all the time. The token budget is about **this step's work** (and therefore this step's duration). KV memory is about **accumulated history**. A request with a 30,000-token conversation and one new token to generate uses 1 unit of token budget but 30,001 tokens of KV memory.

### Fairness: don't starve anyone

The scheduler also picks **which** eligible request goes next. First-come-first-served (FCFS) is the default and is simple to reason about. Priority scheduling (paying customers first, interactive before batch jobs) is common, but strict priority can **starve** low-priority traffic: under steady load, their requests wait forever. The standard fixes are **aging** (priority rises the longer a request waits) and **per-tenant quotas** (no tenant gets more than its share of slots or tokens). In an interview, name your fairness rule explicitly; "we use priorities" without an anti-starvation story is a red flag.

## One big prompt stalls everyone: chunked prefill {#chunked-prefill}

Continuous batching decides **who** is in the batch. The next question is **what kind of work** shares an iteration, and this is where prefill and decode start fighting.

Here's the failure. You have 64 users mid-reply, each getting a token every ~20 ms. Someone uploads an 8,192-token document. If the engine runs that whole prefill as one iteration, every one of those 64 streams has to wait for it before its next token.

Let's put illustrative numbers on it. Suppose (from profiling, in our toy model) the GPU chews through prompt tokens at 25,600 tokens per second when they're mixed into iterations. Then:

```formula
prefill time = prompt tokens ÷ prefill rate = 8,192 ÷ 25,600 tokens/s = 0.32 s = 320 ms
```

So all 64 users see a **320 ms freeze** in their stream, while the GPU dashboard shows a beautiful 100% utilization. This is the **prefill/decode interference** problem, and it's the classic "p95 token gaps spike when document uploads arrive" interview scenario. It's the coffee-shop version of our kitchen: one person orders forty drinks, the barista is flat out, and everyone waiting for a single espresso is furious.

### The fix: slice the prompt

**Chunked prefill** splits a long prompt into pieces and processes one piece per iteration, alongside the ongoing decodes. Nothing about the math changes: the second chunk's tokens attend to the first chunk's tokens through the KV cache the first chunk already wrote, exactly as decode tokens attend to earlier tokens. So the final result is identical; only the timing changes.

With a **per-iteration token budget** of, say, 576 positions and 64 decoding sequences, each iteration carries 64 decode tokens plus up to 576 − 64 = 512 prompt tokens:

```text
 unchunked:
 iter k   [ 64 decodes | 8,192 prompt tokens ....................... ]  ~320 ms  ← every stream freezes
 iter k+1 [ 64 decodes ]                                                ~20 ms

 chunked (budget 576 = 64 decode + 512 prefill):
 iter k    [ 64 decodes | 512 prompt ]  ~20 ms
 iter k+1  [ 64 decodes | 512 prompt ]  ~20 ms
   ... 16 chunks in total (8,192 ÷ 512 = 16) ...
 iter k+15 [ 64 decodes | 512 prompt ]  ~20 ms  → first token for the new user
```

Now the 64 existing streams keep their ~20 ms cadence. The new user's prefill takes 16 × 20 ms = 320 ms, about the same total as before (a little more in practice, since each chunk re-reads the weights and the earlier chunks' KV). We traded almost nothing on the newcomer's TTFT for a massive win on everyone else's inter-token latency.

This idea is the heart of **Sarathi-Serve** (OSDI 2024), which pairs chunked prefills with **stall-free scheduling**: new requests are added to a batch without pausing ongoing decodes. vLLM exposes the same idea: with chunked prefill on, prefills are chunked to fit whatever is left of `max_num_batched_tokens` after the decodes. In the vLLM docs I checked, chunked prefill is on by default; defaults change between versions, so check yours.

### The chunk-size tradeoff

| Chunk / budget | Existing streams | New request's TTFT | Efficiency |
|---|---|---|---|
| Very small (e.g. 64) | smoothest gaps | slower: many iterations, each with fixed overhead | poor: the GPU is underfed and pays per-iteration overhead often |
| Medium (a few hundred to a couple thousand) | gaps stay near the SLO | close to unchunked | good |
| Unbounded (no chunking) | long freezes during big prefills | fastest for that one request | best for prefill alone |

The token budget is a **packing** limit, not a latency guarantee. An iteration's duration also depends on how much KV history the decodes attend over, so 576 tokens against short contexts is faster than 576 tokens against 100K-token contexts. The honest way to choose the budget: profile iteration time across chunk sizes, batch sizes, and context lengths, then pick the largest budget whose iteration time stays inside your TPOT target.

> 🎬 **Animation — chunked prefill in the timeline:** top lane shows 64 small blue decode tokens per tick for three sample streams (label them S1, S2, S3), bottom lane is a new orange 8,192-token prompt. Part 1 (unchunked): one wide orange block of 320 ms lands; the three streams draw a visible gap with a red "320 ms freeze" bracket. Part 2 (chunked): the orange prompt is cut into 16 slices of 512; each tick carries one slice plus the blue decodes; the streams keep a steady 20 ms rhythm; at the end the orange request emits its first token, labelled "TTFT ≈ 320 ms". Finish with a caption: "Total prefill work unchanged. Who waits changed."

## Queues, Little's law, and why 80% busy already hurts {#queueing}

A server's latency isn't only compute time. It's compute time **plus waiting**, and waiting is governed by queueing math. You need two tools.

### Little's law: how many requests are in flight

**Little's law** says that for any stable system, averaged over a long time:

```formula
L = λ · W
```

- `L` is the average number of requests inside the system (in the queue or running).
- `λ` (lambda) is the average arrival rate, in requests per second.
- `W` is the average time a request spends inside the system, in seconds.

It's startlingly general: it doesn't care about the arrival pattern, the service time distribution, or the scheduling order. The only requirement is that the system is **stable**, meaning in the long run requests leave as fast as they arrive.

Why this is gold for LLM serving: `L` is the **concurrency** you have to hold, and concurrency is what eats KV memory and batch slots. Tiny example: chat traffic arrives at λ = 10 requests/s. Each reply averages 300 tokens at 20 ms per token (6.0 s), plus 0.5 s of TTFT, so W = 6.5 s. Then:

```formula
L = 10 req/s × 6.5 s = 65 requests in flight, on average
```

Your fleet has to hold ~65 concurrent sequences on average, plus headroom for bursts. If one GPU's batch tops out at 32 sequences, one GPU cannot serve this, no matter how fast its kernels are.

### Utilization: the hockey stick

The second tool is intuition about **utilization** (ρ, rho): the fraction of time the server is busy, i.e. arrival rate ÷ maximum service rate. The classic textbook single-server queue with random arrivals and random service times (called M/M/1) has a mean queue wait of:

```formula
average queue wait = (ρ ÷ (1 − ρ)) × average service time
```

| Utilization ρ | Wait, in multiples of service time |
|---|---|
| 50% | 1× |
| 80% | 4× |
| 90% | 9× |
| 95% | 19× |

Real LLM servers aren't M/M/1 (they batch, and service times depend on length), so don't quote these numbers as predictions. The **shape** is what's universal: waiting grows gently, then explodes as you approach 100%. That's why "our average utilization is only 85%" is not reassurance, and why bursts hurt even when the average looks fine. It's also why capacity plans target something like 60–80% of measured peak capacity rather than 100%.

> 🎬 **Animation — the hockey stick:** x-axis utilization from 0% to 100%, y-axis mean wait in multiples of service time. Plot ρ/(1−ρ) and drop labelled dots at 50% (1×), 80% (4×), 90% (9×), 95% (19×). Alongside, a small queue of request icons that grows slowly while a slider moves from 50% to 80%, then balloons between 90% and 95%. Final caption: "Averages below 100% don't bound your tail latency."

## A worked capacity example {#capacity-example}

Let's size one worker end to end. All numbers are **illustrative assumptions** for a clean model, not a benchmark.

**Assumptions.**
- A model with 32 layers, 8 KV heads, head width 128, stored in 16-bit (2 bytes). This is the shape of a typical 8B-class model with grouped-query attention.
- Every iteration takes 20 ms, carries up to 32 decoding sequences, and has room for 512 prompt tokens (chunked).
- Prompts are 2,048 tokens; replies average 300 tokens.
- Ignore network time and assume the final prefill chunk also produces the first output token.

**Step 1: prefill capacity.**

```formula
prompt service time = ⌈2,048 ÷ 512⌉ × 20 ms = 4 × 20 ms = 80 ms
prefill rate = 512 tokens ÷ 0.020 s = 25,600 tokens/s = 25,600 ÷ 2,048 = 12.5 prompts/s
```

(`⌈x⌉` means round up to a whole number of iterations.)

**Step 2: decode capacity, via Little's law.** Each reply occupies a decode slot for 300 tokens × 20 ms = 6.0 s. With 32 slots:

```formula
max completion rate = L ÷ W = 32 slots ÷ 6.0 s ≈ 5.33 requests/s
```

So the binding constraint is **decode slots**, not prefill: prefill could handle 12.5 prompts/s, but decode can only finish about 5.3 requests/s. If you only looked at the prefill number, you'd over-promise by more than 2×.

**Step 3: memory check.** KV cache bytes per token (the formula is derived in the KV cache article):

```formula
KV per token = 2 (K and V) × L layers × h_kv × d_head × bytes
             = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB
```

A request at its longest holds 2,048 + 300 = 2,348 tokens, so 2,348 × 128 KiB = 300,544 KiB ≈ 293.5 MiB. Thirty-two of them at worst: 32 × 293.5 MiB ≈ 9,392 MiB ≈ 9.2 GiB. On an 80 GB GPU with ~16 GB of weights, memory isn't what caps us at 32; the 20 ms iteration target is. You could try 64 slots (≈18.3 GiB of KV), but the iteration would get slower, so you'd have to re-profile to see whether TPOT still holds.

**Step 4: size the fleet.** Target traffic is 10 requests/s. At 5.33 req/s per worker, 2 workers would run at 10 ÷ (2 × 5.33) ≈ 94% of decode capacity: right on the steep part of the hockey stick. 3 workers give 10 ÷ 16 ≈ 63%, leaving room for bursts. Plan for 3.

**Step 5: TTFT under a burst.** Four prompts arrive at once on one worker and are chunked first-come-first-served:

| Prompt | Queue wait | Prefill | TTFT |
|---|---|---|---|
| 1 | 0 ms | 80 ms | 80 ms |
| 2 | 80 ms | 80 ms | 160 ms |
| 3 | 160 ms | 80 ms | 240 ms |
| 4 | 240 ms | 80 ms | 320 ms |

With a 250 ms TTFT target, a prompt can wait at most 250 − 80 = 170 ms. Prompt 4 misses. The scheduler's options: route it to a less loaded worker, reject it quickly with a retry hint, or accept the miss. Round-robin chunking (one chunk from each prompt in turn) spreads the pain: all four would finish around 260–320 ms, which is fairer but makes prompt 1 much slower. Which is "better" depends on the metric users experience, which is the point: scheduling policy is a product decision.

Note that four samples can't give you a real p95; this table only shows how the accounting works.

> 🎬 **Animation — sizing a worker:** three gauges side by side labelled "prefill (12.5 prompts/s)", "decode slots (5.33 req/s)", and "KV memory (9.2 of ~60 GiB)". A slider for arrival rate moves from 0 to 10 req/s. The decode gauge goes red first at 5.33 req/s while the other two are still green. Then the view splits into 3 workers; each gauge drops to about 63% and turns green. Final frame: the four-prompt burst as stacked bars (grey = wait, orange = prefill) with a dashed line at 250 ms that bar 4 crosses.

## Saying no on purpose: SLOs, admission control, and backpressure {#admission}

**Admission** is the decision to let a request into the engine at all. Think of it as a promise: once admitted, that request will need memory and compute until it finishes, and its KV cache will keep growing as it generates. A good admission policy protects everyone already inside.

### Reserve or overbook?

The awkward part is that you don't know how long the reply will be. Options:

- **Reserve for the worst case**: charge each request its prompt plus its full `max_tokens` of KV. Safe, but if replies average 300 tokens and you reserve 4,000, you're holding more than 90% of that memory empty.
- **Overbook**: admit based on expected length, like an airline selling a few more seats than the plane has because some passengers don't show. Much higher utilization, but you need an explicit plan for when growth exceeds memory: preempt the newest or lowest-priority request, recompute later, and stop admitting until pressure drops.

Modern engines with paged KV memory mostly overbook and rely on preemption; the important senior-level point is that **preemption is the price of overbooking**, and you should watch its rate as a health metric.

### Bound the queue, in work rather than count

A queue limit of "100 requests" treats 100 short questions and 100 book-length prompts as the same backlog. Bound the queue by **estimated work** (prompt tokens, expected output tokens). When the queue is full, reject fast (HTTP 429 or 503) with a retry-after hint. Clients should retry with **exponential backoff and jitter** (a small random delay, so a thousand clients don't all retry in the same millisecond) and a retry cap; synchronized instant retries turn a brief overload into a sustained one.

### Deadlines and cancellation

If a request's deadline can't be met anymore, the kindest thing is to drop it before spending GPU on it. Before admitting a long prompt, estimate `queue wait + prefill time` and compare it with the remaining TTFT budget, exactly as we did for prompt 4.

**Cancellation** must actually reach the GPU. When a user closes the tab, the HTTP connection closes, but the engine will happily keep generating to `max_tokens` unless the router passes the cancellation to the scheduler, which removes the sequence and frees its KV blocks (after any in-flight GPU work that references them has completed). "The socket closed" does not prove "the GPU work stopped." Slow clients are the mirror image: if a client can't read tokens as fast as you generate them, bound the outbound buffer and pick a policy (pause generation, or disconnect).

### Backpressure

**Backpressure** is the signal that tells upstream senders to slow down. The failure mode is moving the unbounded queue somewhere else: the engine's queue is bounded, so the gateway buffers instead, and now you have a gateway that eats memory and latency. Backpressure has to reach the source of excess work: the load balancer, the client, or an upstream batch job.

```text
 client ──► gateway ──► router ──► [ admission gates ] ──► engine queue ──► scheduler ──► GPU
                                     1. queue work bound?   (bounded)
                                     2. deadline reachable?
                                     3. KV headroom?
                                     4. tenant quota?
           ◄──── 429 + retry-after (fast reject) ◄────┘
```

> 🎬 **Animation — admission gates:** a request token labelled "prompt 6,000 tok, deadline 400 ms" travels left to right through four gates: (1) "queue work: 38k / 50k tokens" passes green, (2) "deadline: est. wait 250 + prefill 240 = 490 ms > 400 ms" turns red. The request bounces back to the client with a "429, retry after 1 s" label. A second request "prompt 500 tok, deadline 400 ms" passes all four gates green and joins the engine queue. Last frame: a user closes a tab; a red cancel signal travels through the router to the scheduler, and that sequence's KV blocks visibly free up in a memory bar.

## Speculative decoding: guess ahead, verify in one pass {#speculative-decoding}

Everything so far shuffles work **between** requests. Speculative decoding attacks a different bottleneck: each request's decode is sequential, one token per full pass of a big model, and at small batch sizes that pass mostly waits on memory.

The trick: at small batch, checking 5 tokens costs almost the same as generating 1, because the expensive part is streaming the weights, not the math. So let a cheap **draft** propose several tokens, then have the big **target** model check all of them in **one** forward pass. It's a junior engineer writing five lines and a senior reviewing the whole diff at once: reading is faster than writing, and anything wrong gets fixed on the spot.

```text
 prefix: "The cat"
 draft (small, fast):   sat   near   me    ← γ = 3 guesses, cheap
 target (one pass):      ✓     ✗     (discarded)
                         │     └─► target samples its own token instead: "on"
 committed this round:  "sat on"   ← 2 tokens for one target pass
```

Each round always commits at least one token (the correction, or a bonus token if every guess is accepted), so it never makes **less** progress per target pass than normal decoding.

### The acceptance rule that keeps it honest

If you're sampling (not always taking the top token; sampling and temperature are covered in **Text in, next token out: tokens, embeddings, and sampling**), you can't just accept draft tokens that "look plausible", or the output would drift toward the draft model's taste. Leviathan, Kalman, and Matias (ICML 2023) give an exact rule. Let `p(x)` be the target's probability for token `x` at this position and `q(x)` the draft's:

```formula
accept draft token x with probability min(1, p(x) ÷ q(x))
if rejected: sample from p′(x) = norm( max(0, p(x) − q(x)) ), then discard the rest of the draft
```

`norm(...)` means rescale so the probabilities sum to 1. In words: if the target likes the token at least as much as the draft did, always keep it. If the draft was over-enthusiastic, keep it only partially, and when you reject, sample from exactly the probability mass the draft under-covered.

**Tiny check with a 3-token vocabulary.** Target `p = {on: 0.5, near: 0.1, by: 0.4}`, draft `q = {on: 0.3, near: 0.4, by: 0.3}`.

- Draft proposes "on" (prob 0.3): p ≥ q, always accept.
- Draft proposes "by" (prob 0.3): p ≥ q, always accept.
- Draft proposes "near" (prob 0.4): accept with 0.1 ÷ 0.4 = 0.25. So total rejection probability is 0.4 × 0.75 = 0.3.
- Residual: max(0, p − q) = {on: 0.2, near: 0, by: 0.1}, sum 0.3, normalized {on: 2/3, by: 1/3}.

Final probabilities: P(on) = 0.3 + 0.3 × 2/3 = 0.5. P(by) = 0.3 + 0.3 × 1/3 = 0.4. P(near) = 0.4 × 0.25 = 0.1. That's exactly `p`. The draft only changes speed, never the output distribution. (Caveat for interviews: "same distribution" is not "same text for the same random seed", and real engines can differ slightly due to floating-point and batch-size effects; vLLM's docs say as much.)

### How much faster?

The paper defines the **acceptance rate** α as the expected probability that a draft token is accepted (it equals the sum over tokens of min(p, q); 0.7 in our toy example). With γ draft tokens per round, the expected tokens committed per round is:

```formula
E[tokens per round] = (1 − α^(γ+1)) ÷ (1 − α)
speedup ≈ (1 − α^(γ+1)) ÷ ((1 − α) × (γ·c + 1))
```

`c` is the draft's cost per token relative to one target pass. The speedup formula assumes the target can verify γ+1 tokens in about the time of one normal step, which is true when decode is memory-bound.

With γ = 4 and a draft that costs c = 0.05 (so γ·c + 1 = 1.2):

| α | α^5 | tokens/round | speedup |
|---|---|---|---|
| 0.8 | 0.328 | 0.672 ÷ 0.2 = 3.36 | 3.36 ÷ 1.2 ≈ 2.8× |
| 0.5 | 0.031 | 0.969 ÷ 0.5 = 1.94 | 1.94 ÷ 1.2 ≈ 1.6× |
| 0.3 | 0.002 | 0.998 ÷ 0.7 = 1.43 | 1.43 ÷ 1.2 ≈ 1.2× |

The paper reports 2–3× on T5-XXL. Acceptance depends heavily on the task: code and templated text are very predictable, creative writing at high temperature is not.

### Why it's a scheduling problem, not a free win

From the scheduler's view, the unit of work stops being "one token per step" and becomes "a round that commits a variable number of tokens." And the core assumption, that verifying extra tokens is nearly free, **breaks at high load**. When the batch is already large, decode is closer to compute-bound, so the γ extra positions per sequence cost real compute, and the draft's own KV cache and weights take memory that could have held more users. A round that commits 3 tokens in 45 ms beats normal decoding at 20 ms/token (60 ms ÷ 45 ms ≈ 1.33×); one that commits 1 token in 45 ms is 2.25× worse. That's why engines turn speculation down or off as batch size grows, and why you measure accepted tokens per round **and** goodput at your real concurrency before switching it on globally.

The draft doesn't have to be a separate small model. vLLM, for example, supports separate draft models, n-gram lookup (copying likely continuations from the prompt, great for editing and RAG), and methods like EAGLE and multi-token-prediction heads that attach lightweight predictors to the target model itself.

> 🎬 **Animation — one speculative round:** prefix "The cat" on the left. Step 1: a small grey draft box emits "sat", "near", "me" one after another quickly. Step 2: a large blue target box processes all three at once in a single pulse; above each token show p and q (sat: p 0.6, q 0.5; near: p 0.1, q 0.4). Step 3: "sat" gets a green tick (p ≥ q). "near" shows a coin flip with 25% acceptance that lands on reject; "near" and "me" fade out. Step 4: the target samples "on" from the residual distribution (bar chart: max(0, p − q)). Step 5: the committed text "The cat sat on" glows, with a counter "2 tokens / 1 target pass".

## Splitting prefill and decode onto different GPUs {#disaggregation}

Chunked prefill makes prefill and decode share one GPU politely. **Prefill/decode disaggregation** goes further: put prefill on one pool of GPUs and decode on another. A request is prefilled on a prefill worker, its KV cache is shipped to a decode worker, and it streams from there.

```text
                ┌──────────── prefill pool ────────────┐        ┌──────── decode pool ─────────┐
 new request ─► │ GPU P1: big prompt passes, compute   │ ─KV──► │ GPU D1: large decode batch,   │ ─► tokens
                │ heavy, tuned for TTFT                │ cache  │ bandwidth heavy, tuned for    │
                │ GPU P2: ...                          │ xfer   │ TPOT                          │
                └──────────────────────────────────────┘        └───────────────────────────────┘
```

Why bother? Three reasons. **No interference**: a giant prompt can't freeze decode streams because they're on different hardware. **Independent tuning**: the two phases want different things (DistServe's point is that you can choose different parallelism and GPU counts for each). **Independent scaling**: a workload of long prompts and short replies needs more prefill capacity; chat with short prompts and long replies needs more decode.

DistServe (OSDI 2024) frames this around goodput, the maximum request rate served within both TTFT and TPOT targets, and reports serving up to 7.4× more requests, or meeting 12.6× tighter SLOs, than the systems it compared against on its workloads.

### The price: moving the KV cache

The cost is shipping every prompt's KV cache between machines. Using our 128 KiB/token model:

| Prompt | KV size | at 25 GiB/s | at 250 GiB/s |
|---|---|---|---|
| 2,048 tokens | 256 MiB | 10 ms | 1 ms |
| 32,768 tokens | 4 GiB | 160 ms | 16 ms |

(2,048 × 128 KiB = 262,144 KiB = 256 MiB; 256 MiB ÷ 25 GiB/s = 0.25 ÷ 25 s = 10 ms. The link speeds are round illustrative numbers for "a network link" and "a fast link inside one server".) These are lower bounds: add queueing on both sides and synchronization. Transfer can be overlapped layer by layer, but on a slow link a long prompt can lose more to transfer than it gained from avoiding interference. That's why DistServe treats placement and bandwidth as part of the design.

Operational gotchas that interviewers like:

- **Reserve decode capacity before starting expensive prefill.** Otherwise finished prefills pile up holding KV memory, waiting for a decode slot.
- **Handoff is a state** with its own failure modes: transfer failure, cancellation mid-transfer, and who frees the memory. Model these as ordinary states in the state machine, not as afterthoughts.
- **Compare fairly.** Disaggregated vs co-located (with chunked prefill) must be compared at equal hardware cost and equal SLOs, measuring goodput.

### How to prove any of this helped

Use an **open-loop** load test: requests are sent on a schedule (e.g. Poisson arrivals at 10 req/s) regardless of whether earlier ones have come back, the way real users keep arriving. A **closed-loop** tester (each virtual user waits for its response before sending the next) automatically slows down when the server does, hiding exactly the queue build-up you're trying to see. Sweep realistic prompt/output length distributions, burstiness, chunk budgets, and concurrency, and record client-side TTFT and gap percentiles, rejection and timeout rates, KV occupancy, preemptions, and goodput. More on turning these into release decisions in **Evaluation and observability: knowing whether it actually works**.

> 🎬 **Animation — co-located vs disaggregated:** left panel: one GPU box with a timeline where orange prefill chunks interleave with blue decode ticks. Right panel: two boxes, "prefill pool" and "decode pool", joined by a pipe. A 2,048-token request enters the prefill box, becomes a 256 MiB KV block, travels down the pipe (a label shows "10 ms at 25 GiB/s"), and starts emitting blue tokens in the decode box. Then replay with a 32,768-token prompt: a 4 GiB block crawls through the pipe with "160 ms" in red. Final frame: a scale weighing "interference removed" against "transfer + extra queue".

# Interview

## Question

You run a chat service. Median TTFT is within target, but p95 inter-token gaps spike whenever users upload long documents, even though GPU utilization is high. Walk me through what's happening, what you'd change, and how you'd prove it worked.

## Answer

**Diagnose first.** High utilization with bad streaming latency is the signature of prefill/decode interference: a long prompt's prefill runs as a big iteration, and every active stream waits for it. I'd confirm it by correlating long-prefill iterations with the gap spikes in engine traces, and separate engine time from client-side buffering (timestamps at the engine vs the client), because network buffering can fake stalls.

**Fix at the scheduler.** Make sure we're on continuous batching with chunked prefill and a per-iteration token budget. I'd profile iteration duration against chunk size, batch size, and context length, then pick the largest budget that keeps iteration time under the TPOT target. Decodes get their slot in every iteration first (stall-free), and prefill chunks fill the remaining budget. For fairness, FCFS or priority with aging, so long prompts still make progress and can't be starved.

**Protect it at admission.** Bound the queue by estimated work, not request count; check KV headroom and whether a request's deadline is still reachable; reject fast with retry-after and have clients back off with jitter. Make sure cancellations propagate to the scheduler and free KV blocks.

**Prove it.** Replay a realistic open-loop workload (same arrival rate and burstiness, real length distributions, including document uploads) before and after. Compare p95/p99 TTFT, p95 individual token gaps (not just average TPOT), rejection and timeout rates, preemptions, and goodput at equal hardware. Throughput going up alone doesn't count.

**If it's still not enough**, evaluate prefill/decode disaggregation, including KV transfer time for our real prompt lengths, the decode reservation policy, and the cost of the extra queue, compared at equal hardware cost and equal SLOs. I'd also check whether speculative decoding helps TPOT at our actual batch sizes, but I'd expect little gain if the decode batch is already large.

## Follow-ups

- Outputs vary from 10 to 4,000 tokens. How do you decide how much KV memory to reserve per request, and what happens when you guess wrong?
- Why can lowering p95 queue wait coincide with worse p95 TTFT?
- When would speculative decoding reduce goodput instead of improving it?
- How would per-tenant quotas interact with deadline-aware scheduling?
- Using Little's law, how many concurrent sequences do you need for 20 req/s with 8 s average request lifetime, and what does that imply for KV memory?

# Pitfalls

- **Adding percentiles.** p95 queue time + p95 prefill time ≠ p95 TTFT; the components come from different requests. Measure the end-to-end distribution.
- **Treating the per-iteration token budget as a memory limit.** It counts new positions processed this step; KV memory holds everyone's history. They're separate constraints.
- **Believing "high GPU utilization" means healthy serving.** A giant prefill keeps the GPU at 100% while every stream stalls. Utilization measures busyness, not deadline satisfaction.
- **Sizing only on prefill throughput.** In the worked example prefill handled 12.5 prompts/s, but decode slots capped the worker at ~5.3 req/s. Use Little's law on both phases.
- **Running near 100% utilization on average.** Queue wait explodes as utilization approaches 1; bursts break tail latency long before the average hits 100%.
- **Strict priority with no aging.** Low-priority tenants starve under sustained load.
- **Assuming a closed socket stops GPU work.** Without cancellation propagation, the engine keeps generating and holding KV memory.
- **Saying speculative decoding changes outputs, or always speeds things up.** With the exact acceptance rule the distribution is unchanged; the speed gain depends on acceptance rate and shrinks or reverses at large batch sizes.
- **Comparing disaggregated and co-located serving without transfer costs or at unequal hardware.**
- **Load testing closed-loop.** It slows its own arrival rate when the server slows, hiding overload.

# Checklist

- Define TTFT, TPOT, inter-token latency, throughput, and goodput, and say which the user actually feels.
- Compute the useful-work fraction of a static batch from a list of output lengths, and explain how continuous batching recovers it.
- Explain why a long unchunked prefill freezes every active stream, and how chunked prefill with a token budget fixes it.
- Name the three separate admission limits: sequence slots, per-iteration token budget, and KV memory.
- Apply Little's law to turn arrival rate and request lifetime into required concurrency and KV memory.
- Size a worker and a fleet end to end, identifying which constraint binds first, and leave burst headroom.
- Design admission control: work-bounded queue, deadline check, fast rejection with backoff and jitter, cancellation propagation.
- Walk through the speculative acceptance rule on a toy distribution and compute expected tokens per round from α and γ.
- Estimate KV transfer time for disaggregation and argue when it pays.
- Describe an open-loop load test and the metrics you'd report.

# Sources

- [Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — Iteration-level scheduling, selective batching, and the reported 36.9× throughput gain over FasterTransformer at equal latency.
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (Agrawal et al., OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/agrawal) — Chunked prefills and stall-free scheduling that adds new requests without pausing ongoing decodes.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (Zhong et al., OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) — Prefill/decode disaggregation, goodput under TTFT and TPOT constraints, and the reported 7.4× / 12.6× results.
- [Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman, Matias, ICML 2023)](https://arxiv.org/abs/2211.17192) — The exact acceptance rule, the residual distribution, the expected-tokens and walltime-improvement formulas, and the 2–3× T5-XXL result.
- [vLLM scheduler configuration](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/) — `max_num_batched_tokens`, `max_num_seqs`, chunked prefill, and FCFS vs priority policies; defaults are version-dependent.
- [vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) — Supported draft methods (draft model, n-gram, EAGLE, MTP and others) and the caveats on losslessness under floating-point and batching effects.
- [Little's law (Wikipedia)](https://en.wikipedia.org/wiki/Little%27s_law) — L = λW for stable systems, independent of arrival distribution, service distribution, and service order; cites Little's 1961 proof.

# Flashcards

## ttft

**Q:** What does client-visible TTFT include, and why can't faster kernels alone fix it?

Time to first token runs from the moment the request arrives to the moment the first generated token is visible. It includes **queue wait**, prefill compute, sampling, and network/buffering. Under load, queue wait often dominates, so a faster prefill kernel does little if requests sit behind a burst. Fixes live in scheduling, admission, and capacity as much as in kernels.

## tpot

**Q:** Why look at individual inter-token gaps as well as average TPOT?

TPOT is (last token time − first token time) ÷ (N − 1), an average over the whole reply. A single 1-second freeze in a 500-token reply barely moves the average but is obvious to a person watching the text stream. Tail percentiles of individual gaps catch the stalls caused by things like unchunked prefills.

## quantiles

**Q:** Can you add p95 queue time and p95 prefill time to get p95 TTFT?

No. Percentiles don't add, because the request at the 95th percentile of queue time is usually not the one at the 95th percentile of prefill time. The sum is typically a pessimistic overestimate and in any case isn't the real number. Measure the end-to-end TTFT distribution directly.

## static-waste

**Q:** A static batch of 4 has replies of 10, 50, 20, and 100 tokens. What fraction of slot-steps is useful, and what fixes it?

The batch runs 100 steps × 4 slots = 400 slot-steps, but only 10 + 50 + 20 + 100 = 180 produce tokens: 45%. Finished sequences sit idle until the longest one ends, and new requests wait. Continuous (iteration-level) batching lets finished sequences leave and new ones join at every iteration, filling those idle slots.

## continuous

**Q:** What is the scheduling boundary in continuous batching?

The iteration: one forward pass. Between iterations the scheduler removes finished sequences and admits new ones, while surviving sequences keep their KV cache. It doesn't interrupt a GPU kernel mid-flight. Orca introduced this as iteration-level scheduling, together with selective batching so that sequences of different lengths can share the big matmuls.

## contention

**Q:** Why can high GPU utilization coexist with terrible streaming latency?

A large prefill uses the GPU very efficiently, so utilization looks great, but if it runs as one long iteration every active decode stream waits for it (e.g. an 8,192-token prompt at 25,600 tokens/s freezes all streams for ~320 ms). Utilization measures busyness, not whether anyone's deadline was met.

## chunking

**Q:** What's the central tradeoff in choosing the chunked-prefill size?

Smaller chunks give decodes a turn more often, keeping inter-token gaps smooth, but add per-iteration overhead and underfeed the GPU, slightly slowing the new request's TTFT. Larger chunks are more efficient for prefill but make each iteration longer, stretching everyone's token gaps. Choose by profiling iteration time and keeping it under the TPOT target.

## budgets

**Q:** Why are the per-iteration token budget and the KV memory budget separate constraints?

The token budget counts **new** positions processed in this forward pass (prompt-chunk tokens plus one per decoding sequence), which determines how long the step takes. KV memory holds the **history** of every resident sequence and grows as they generate. A long conversation generating one token uses 1 unit of token budget but its entire history in KV memory.

## littles-law

**Q:** State Little's law and use it: 10 requests/s, each lives 6.5 s in the system. What concurrency do you need?

L = λ · W: average number in the system equals arrival rate times average time in the system, for any stable system regardless of arrival pattern or scheduling order. Here L = 10 × 6.5 = 65 concurrent requests on average, before burst headroom. That concurrency drives batch slots and KV memory, so it's the fastest way to size a fleet.

## reservation

**Q:** Why not reserve every request's maximum possible output in KV memory?

It's safe but wasteful: if replies average 300 tokens and you reserve 4,000, most of that memory sits empty and you serve far fewer users. The alternative is overbooking on expected length, which raises utilization but requires a recovery plan (preempt and recompute, stop admitting) when sequences grow more than expected. Watch the preemption rate.

## cancel

**Q:** What makes cancellation actually effective in an inference server?

The cancel has to travel from the client through the router to the scheduler, which removes the sequence from future iterations and frees its KV blocks once any in-flight GPU work referencing them has completed. A closed HTTP socket alone doesn't stop generation; without propagation, the engine keeps generating to max_tokens and holding memory.

## example

**Q:** With 512 prompt tokens per 20 ms iteration, how long is a 2,048-token prefill, and what's the TTFT of the 4th of four simultaneous prompts?

⌈2,048 ÷ 512⌉ = 4 iterations × 20 ms = 80 ms of prefill. First-come-first-served, the 4th waits 3 × 80 = 240 ms, so its TTFT is 240 + 80 = 320 ms (ignoring network). Against a 250 ms target it misses: the maximum tolerable wait is 170 ms, so route it elsewhere, reject it fast, or add capacity.

## exact-speculation

**Q:** What rule makes speculative sampling preserve the target model's output distribution?

Accept a draft token x with probability min(1, p(x)/q(x)), where p is the target's distribution and q the draft's. On rejection, sample from norm(max(0, p − q)) and discard the rest of the draft. Tokens the draft over-proposed get thinned out, and the residual adds back exactly the mass the draft under-covered, so the final distribution equals p. Accepting "plausible" tokens instead would bias output toward the draft.

## spec-speed

**Q:** How do acceptance rate and batch size determine whether speculative decoding helps?

Expected tokens per round is (1 − α^(γ+1)) ÷ (1 − α); with α = 0.8 and γ = 4 that's about 3.36, and with a cheap draft (c = 0.05) about a 2.8× speedup. This assumes verifying γ+1 tokens costs about one normal step, which holds when decode is memory-bound at small batch. At large batch the extra verification positions cost real compute and the draft takes memory, so low acceptance or high load can make it a net loss for goodput.

## disaggregate

**Q:** What does prefill/decode disaggregation buy, and what new cost does it introduce?

It removes interference (long prefills can't freeze decode streams), lets each phase use different parallelism and GPU counts, and lets them scale independently. The price is shipping each prompt's KV cache to a decode worker (e.g. a 32,768-token prompt at 128 KiB/token is 4 GiB, which is 160 ms at 25 GiB/s) plus an extra queue and new failure states. It pays when interference is the bottleneck and the link is fast.

## loadtest

**Q:** Why use open-loop load generation to evaluate a serving system under overload?

Open-loop sends requests on a fixed schedule regardless of responses, like real users who keep arriving. Closed-loop clients wait for each response before sending the next, so when the server slows down they automatically send less, hiding the queue growth and tail latency you're trying to measure.
