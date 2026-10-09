---
{
  "slug": "farm-blueprint",
  "title": "AI farm, part 1: designing a two-server Qwen deployment",
  "category": "ai-farm",
  "summary": "Part 1 of 3: the design behind a small private chat service. Two GPU hosts run the same Qwen model, and a stateless Express tier owns identity, history and policy. This part covers sizing, routing, the auth boundary, the life of a turn, and what happens when things break.",
  "difficulty": "Systems",
  "minutes": 35,
  "prerequisites": ["serving-kv-cache", "inference-scheduling", "llm-security"],
  "learningObjectives": [
    "Separate durable application state (Postgres, owned by Express) from disposable inference state (KV caches on the GPUs)",
    "Estimate GPU memory for weights and KV cache and turn measured token throughput into an admission budget, with every assumption stated",
    "Defend Nginx least_conn for stateless API replicas and explain why rendezvous hashing to a GPU is only an optional locality optimization",
    "Design the authentication boundary and a local-by-default provider policy that never silently sends a transcript to the cloud",
    "Walk through a durable chat turn and a failure matrix, and justify leaving Kafka out of the first interactive version"
  ]
}
---

# Sections

## The picture: what we're building and who owns what {#the-picture}

This is part 1 of a three-part guide to building a small, private, ChatGPT-style chat service for a company, running on hardware the company owns. Part 1 is the design: the boxes, the arrows, the numbers, and the reasons behind them. Part 2 turns the design into Docker Compose files and shell commands. Part 3 walks through the application code that routes requests and keeps conversations consistent. Everything here matches the reference example in `examples/ai-farm/`. That example is **educational**: nobody has deployed or benchmarked it for this article, so any number labelled *illustrative* is one I made up to show the arithmetic.

First, the vocabulary. You'll see these words constantly, so let's pin them down:

| Term | Plain meaning |
|---|---|
| **Inference** | Running a trained model to produce output. No learning happens. |
| **llama.cpp / `llama-server`** | An open-source C/C++ inference engine. Its `llama-server` program loads a model file and serves an HTTP API that looks like OpenAI's (`/v1/chat/completions`). |
| **Qwen2.5-7B-Instruct** | An open-weight chat model from Alibaba's Qwen team, with about 7.6 billion parameters (learned numbers) and an Apache-2.0 licence. |
| **GGUF** | The single-file format llama.cpp loads. It holds the weights plus the tokenizer, hyperparameters and chat template. |
| **Quantization** | Storing each weight in fewer bits (for example about 3–4 instead of 16) so the model is smaller and faster to read. |
| **Express** | A small Node.js web framework. Our application logic lives here. |
| **Nginx** | A web server used here as the **gateway**: the single front door that forwards browser requests to Express. |
| **Postgres** | The relational database, and the one place where truth lives. |
| **SSE** | Server-Sent Events: one long HTTP response that the server keeps writing small `data:` chunks into. It's how tokens stream to the browser. |

Here's the whole system on a napkin. The reference layout puts the control tier on host A, next to one GPU:

```text
                         Host A (10.20.0.11, example)                 Host B (10.20.0.12)
  browser                ┌───────────────────────────────────┐        ┌─────────────────┐
     │  SSH tunnel /     │  control project                  │        │ inference       │
     │  corporate edge   │  ┌───────┐   ┌──────┐             │        │ ┌─────────────┐ │
     └──────────────────►│  │ Nginx │──►│ api1 │──┐          │        │ │ llama-server│ │
          :8088          │  │gateway│   └──────┘  │  ┌────┐  │  LAN   │ │ Qwen GGUF   │ │
                         │  │least_ │   ┌──────┐  ├─►│ db │  │ :8080  │ │ GPU B       │ │
                         │  │conn   │──►│ api2 │──┘  │(PG)│  │◄──────►│ └─────────────┘ │
                         │  └───────┘   └──┬───┘     └────┘  │        └─────────────────┘
                         │                 │                 │
                         │  inference      ▼                 │
                         │  ┌─────────────────┐              │
                         │  │ llama-server    │  GPU A       │
                         │  │ same Qwen GGUF  │              │
                         │  └─────────────────┘              │
                         └───────────────────────────────────┘
   Only api1/api2 talk to Postgres. Only api1/api2 talk to inference A or B.
```

The two GPUs are **independent replicas**. Each one holds a complete copy of the same model and can answer any request by itself. They are *not* one model split across two cards (that would be tensor parallelism, a different design). A CPU-only third machine could host the control tier instead. The README mentions that option; you would just update the firewall rules.

Now the most important idea in the whole design, as an analogy. **Postgres is the filing cabinet; the GPUs are whiteboards.** The filing cabinet holds the official record: who you are, which conversations you own, every message, and each turn's status. A whiteboard holds scratch work that makes the next answer faster, namely the **KV cache**. That's the attention keys and values the model already computed for text it has read, so it doesn't have to recompute them. (We'll go through this in detail in **What happens at inference: prefill, decode, and the KV cache**.) If someone wipes a whiteboard, you lose some speed. You never lose a conversation.

| Component | Owns | May it hold state? |
|---|---|---|
| Nginx gateway | TLS/edge transport (outside the lab), spreading requests over the API replicas | No |
| Express (api1, api2) | Login, sessions, conversation ownership checks, turns, revisions, request IDs, provider policy, logs, **all Postgres writes** | Nothing it can't lose. Each process keeps small counters, like the admission counter. |
| Postgres | Authoritative sessions, conversations and turns | Yes. It's the source of truth. |
| llama-server A/B | Math: prompt in, tokens out | Only ephemeral KV/prefix caches |

"Stateless" Express means any replica can serve any request, because everything that matters is fetched from Postgres on each call. That single decision is what makes the rest of the design simple. Load balancing needs no stickiness, a GPU restart is only a performance event, and a crashed API process loses at most the one stream it was serving.

> 🎬 **Animation — filing cabinet vs whiteboards:** show the topology above. Step 1: a request travels browser → Nginx → api1 → Postgres (a "read history" arrow) → GPU A, and tokens flow back. Step 2: GPU A's whiteboard (labelled "KV cache, 448 MiB") fills with scribbles. Step 3: GPU A reboots and the whiteboard is wiped clean; the Postgres cabinet is untouched. Step 4: the next turn arrives through api2 (not api1), reads the same history from Postgres, lands on GPU A or B, and succeeds. Caption: "Cache lost = slower. History lost = never."

One thing that surprises people: llama-server ships with its own built-in chat web page. Don't let users use it. It talks straight to the GPU and **bypasses** login, ownership checks and history. Treat it as an operator diagnostic tool, reachable only from the control network.

## Write the service contract before you touch hardware {#service-contract}

A junior engineer's first question is "which GPU should we buy?" A senior engineer's first question is "what exactly are we promising?" You can't size anything until the promise is written down, the same way you can't size a restaurant kitchen from the number of people who live in the town. You need to know how many walk in at lunch and what they order.

Here's the contract the reference example is built around, split into fixed choices (from the example files) and targets you'd negotiate (illustrative):

| Item | Value | Where it comes from |
|---|---|---|
| Model | Qwen2.5-7B-Instruct, Q3_K_M GGUF, one pinned revision and SHA-256 hash | `.env.example` |
| Public model name | `corp-qwen` on both hosts (`--alias`) | `inference/compose.yml` |
| Slots per GPU host | 2 (`--parallel 2`) | `inference/compose.yml` |
| Context per host | 8,192 tokens total, split by `--no-kv-unified` into roughly 4,096 per slot | `inference/compose.yml` and README |
| Output cap | 512 tokens (`--n-predict 512`, and the API also sends `max_tokens: 512`) | compose and `api/server.js` |
| API bounds | 2,000-character user message; 6,000 characters of history plus the new message; 16,000 characters of output; 120 s timeout; 4 active chats per API replica | `api/server.js` and README |
| Time to first token, p95 | ≤ 4 s *(illustrative target)* | negotiated |
| Peak arrival rate | 0.2 requests/s *(illustrative: 20 active people each sending a message about every 100 s)* | measured or estimated |
| Degraded mode | With one GPU down, reject excess load with an explicit 429/503 rather than queue forever | design decision |

Two terms in that table deserve a definition. A **slot** is llama-server's name for one sequence it can work on at a time, with its own reserved slice of KV cache. **p95** means the 95th percentile: 95 % of requests do at least this well. Tail percentiles matter more than averages for chat, because the person who waited 30 s is the one who files the ticket.

Notice what's *not* in the contract: "100 employees". Registered users are nearly useless for sizing. What matters is the **arrival rate** (requests per second at peak) and the **token distribution** (how long the prompts and answers are). A hundred people who send three messages a day is a different system from twenty analysts hammering it all afternoon.

Also notice the **model identity** rule. Both hosts must run the *same* artifact: the same file hash, tokenizer, chat template (the exact text formatting wrapped around each chat message) and sampling settings. The `corp-qwen` alias is just a label; two servers can share a label and serve different weights. If A and B ever diverge, a conversation that bounces between them gets answers from two different models, and nothing will tell you. So the operator records a manifest of the image digest, model revision, file hash and flags, and upgrades are done as a deliberate, one-host-at-a-time procedure. **AI farm, part 2: deploying it with Docker, step by step** covers that procedure.

## Will it fit? Sizing GPU memory {#memory}

GPU memory is the first hard wall. A model that doesn't fit serves nobody. Think of VRAM (the GPU's own memory) as a suitcase holding three things: the **weights**, the **KV cache**, and **working space** (runtime buffers, the CUDA context and scratch space for intermediate results).

### Weights

Start with the file. The Qwen model card lists the Q3_K_M file at about 3.81 GB. Here's a sanity check that teaches something:

```formula
effective bits/param = file bytes × 8 ÷ parameters = 3.81×10⁹ × 8 ÷ 7.61×10⁹ ≈ 4.0 bits
```

"Q3" does not mean three bits per weight across the board. K-quant schemes store a scale per group of weights and keep some sensitive tensors (like the embedding and output layers) at higher precision, so the *effective* rate lands around 4 bits. The lesson: size from the real file, not from the name. In binary units, 3.81×10⁹ bytes ÷ 2³⁰ ≈ **3.55 GiB** of weights. (Formats, group scales and quality tradeoffs are covered properly in **Quantization: spending fewer bits per weight**. Choosing a 3-bit-class file here simplifies the lab. It is not a recommendation over Q4 or Q5, and lower precision needs its own task-quality evaluation.)

### KV cache

For every token a slot holds, every layer stores one key vector and one value vector per KV head. From the model's `config.json`, Qwen2.5-7B has `L` = 28 layers, `h` = 28 query heads, `h_kv` = 4 key/value heads, and `d = 3584`, so `d_head = 3584 / 28 = 128`. It uses **grouped-query attention (GQA)**: 7 query heads share each K/V head, which shrinks the cache by 7×. (Why that works is in **Making attention cheaper: GQA, FlashAttention, and long context**.) llama.cpp's default cache type is f16, which is 2 bytes per element.

```formula
KV bytes per token = 2 × L × h_kv × d_head × bytes = 2 × 28 × 4 × 128 × 2 = 57,344 B = 56 KiB
```

The leading 2 is for K *and* V. `L` is the layer count, `h_kv` the number of KV heads, `d_head` the per-head width, and `bytes` the size of one element (2 for f16).

```text
 per host: --ctx-size 8192, --parallel 2, --no-kv-unified

   slot 0: 4,096 tokens × 56 KiB = 224 MiB   ████████
   slot 1: 4,096 tokens × 56 KiB = 224 MiB   ████████
                                   -------
   total KV                        448 MiB
```

Check: 57,344 × 4,096 = 234,881,024 B = 224 MiB, and twice that is 448 MiB. So the whole suitcase looks roughly like this:

| Item | Size | Status |
|---|---|---|
| Weights (Q3_K_M) | ≈ 3.55 GiB | from the model card |
| KV cache, 2 slots × 4,096 | 0.44 GiB | computed |
| Runtime and compute buffers | ~1–2 GiB | *illustrative guess; read the startup log* |
| **Total** | **≈ 5–6 GiB** | plausibility estimate only |

Two lessons come out of this table.

First, at this configuration **memory is not the binding constraint**. Almost any modern data-centre or workstation GPU has room to spare. The limit you'll hit is speed, which is the next section.

Second, KV cache is where memory goes when you get ambitious. Suppose someone asks for 8 slots of 32,768 tokens each: 57,344 × 262,144 = 15,032,385,536 B = **14 GiB** of KV alone. Without GQA (28 KV heads instead of 4) the original 2 × 4,096 tokens would take 7 × 448 MiB ≈ 3.06 GiB. KV scales with *tokens held concurrently*, so long contexts and many slots multiply fast.

> 🎬 **Animation — packing the VRAM suitcase:** a vertical bar for one GPU. Step 1: drop in a 3.55 GiB "weights" block. Step 2: add two 224 MiB "slot" blocks labelled 4,096 tokens each. Step 3: add a hatched "runtime buffers (measure me)" block. Step 4: a slider changes slots from 2 to 8 and per-slot context from 4,096 to 32,768; the KV blocks grow to 14 GiB and overflow a dashed line labelled "your card". Step 5: toggle "no GQA" and watch the original 448 MiB swell to 3.06 GiB.

Why `--no-kv-unified`? With a **unified** KV buffer, all slots share one pool, and a single long request can use most of the context. With it disabled, each slot gets a fixed share (8,192 ÷ 2 ≈ 4,096 per slot on a matching build; the README says to confirm the actual per-slot figure in the startup logs). Fixed shares make the per-request limit predictable, which is what the API's bounds are designed around. The companion flag `--no-context-shift` means that when a slot's context is full, llama.cpp won't quietly discard the oldest tokens to make room. Silent truncation would mean the model forgot part of the conversation without telling anyone.

Now the per-slot token budget:

```text
  4,096 tokens per slot
  ├── chat template overhead (role markers etc.)       ~tens of tokens
  ├── history + new user message (≤ 6,000 characters)
  └── reserved for output                             512 tokens
  => prompt side must stay ≤ 4,096 − 512 = 3,584 tokens
```

The API caps history at 6,000 *characters*, not tokens, because counting characters is cheap and the API doesn't run Qwen's tokenizer. For English prose at an illustrative ~4 characters per token, that's about 1,500 tokens, well inside 3,584. But dense code or some non-Latin scripts can approach one token per character, and 6,000 tokens would overflow. The character cap is a guardrail, not a guarantee. The README says exactly that, and a senior candidate should say it before the interviewer does. When the cap is hit, the reference API returns `413 history_limit_start_new_conversation` instead of quietly dropping old turns.

## From tokens per second to how many people can chat {#capacity}

Fitting in memory tells you the GPU *can hold* two conversations. It doesn't tell you how fast it answers them. Here's the intuition. Generating each new token (**decode**) requires streaming essentially all the weights from VRAM through the compute units. It's like a librarian who has to walk the entire library to write each word. Reading the prompt (**prefill**) is different: the model processes the whole prompt in parallel, one walk for many tokens, so prefill is compute-heavy but fast per token.

A back-of-envelope ceiling, *illustrative*: on a GPU with 500 GB/s of memory bandwidth, one stream can't decode faster than about 500 ÷ 3.81 ≈ **131 tokens/s**, because each token re-reads ~3.81 GB of weights. Real numbers come in lower. The useful insight is that when two slots decode together, llama.cpp batches them into the same pass. The weights are read once for both, so aggregate throughput rises while each stream gets somewhat slower. That's why the README warns that "two slots are not a promise of two requests at full single-user speed." (Continuous batching and its tradeoffs are covered in **Serving many users: batching, scheduling, and speculative decoding**.)

So you **measure**, then do arithmetic on the measurements. Here's a worked example with *illustrative* load-test results:

| Assumption (illustrative) | Value |
|---|---|
| Per-stream decode speed with both slots busy | 30 tokens/s |
| Mean output length | 300 tokens |
| Mean prompt processing (prefill) time | 1 s |
| Slots in the farm | 2 hosts × 2 = 4 |

```formula
service time per request  S ≈ prefill + output ÷ speed = 1 + 300 ÷ 30 = 11 s
farm throughput ceiling   μ = slots ÷ S = 4 ÷ 11 ≈ 0.36 requests/s ≈ 22 per minute
```

Now bring in **Little's law**, the single most useful queueing fact for an interview:

```formula
L = λ × W
```

`L` is the average number of requests in the system, `λ` (lambda) is the arrival rate, and `W` is the average time each request spends inside. The coffee-shop version: if 2 people arrive per minute and each stays 10 minutes, about 20 are inside on average.

At our illustrative peak λ = 0.2 requests/s and W ≈ 11 s, L = 0.2 × 11 = **2.2 busy slots** on average, out of 4. That's 55 % utilization, which is comfortable. Averages hide bursts, though. Several people hitting Enter in the same second will briefly exceed 4, and those extras wait.

Now the scenario that separates a senior answer from a junior one: **one GPU dies**.

```text
                 slots   ceiling μ          arrival λ    verdict
  both hosts       4     4/11 ≈ 0.36 /s     0.20 /s      OK  (55% busy)
  one host         2     2/11 ≈ 0.18 /s     0.20 /s      OVERLOADED: λ > μ
```

When arrivals exceed capacity, the queue doesn't just get long; it grows *without bound* for as long as the peak lasts. Every waiting request eventually hits the 120 s API timeout, having burned a slot's worth of patience for nothing. You have exactly three honest options: buy spare capacity, keep peak load below one host's measured budget, or **reject excess load explicitly**. The reference design picks the third. It returns `429` (too many requests) or `503` (service unavailable) quickly, so the UI can say "busy, try again shortly", which beats a spinner that dies after two minutes.

> 🎬 **Animation — the queue that never drains:** four slot boxes (A0, A1, B0, B1) with arrivals as dots dropping in at 0.2/s, each dot occupying a slot for 11 s. Step 1: normal state, with about 2 slots lit on average and a short queue that empties. Step 2: host B goes grey; only A0/A1 remain. Step 3: dots arrive faster than they leave; the queue line grows steadily and a timer on the oldest dot climbs toward 120 s. Step 4: switch on "admission limit": extra dots bounce off with a red "429" label and the queue stays short.

Now look at how admission is actually wired in the reference, because it has a deliberate wrinkle:

```text
   api1: active < 4 ? admit : 429        api2: active < 4 ? admit : 429
          (own counter)                          (own counter)
                  \                              /
                   ▼                            ▼
          up to 8 admitted chats  ──►  only 4 GPU slots
                                       extras wait inside llama-server
                                       ("requests_deferred" metric)
```

Each API replica counts only its *own* active chats, so the two replicas together can admit up to 8 against 4 slots. The overflow waits in llama-server's queue, still inside the 120 s timeout. This is **best-effort, replica-local** admission, not a global semaphore. The README labels it exactly that way. A real cluster admission controller would need shared state (a Postgres row or Redis counter, or a token bucket per GPU) and would have to account for tokens, not just request counts. In an interview, name this limitation yourself.

## Two routing decisions, not one {#routing}

"Send the request to a server" is actually two separate choices made by two different components:

```text
  decision 1 (Nginx):   which API replica handles this HTTP request?
                        → least_conn over api1, api2
  decision 2 (Express): which GPU runs this generation?
                        → rendezvous hash(conversationId, server) over healthy nodes
```

### Decision 1: least_conn at the gateway

Nginx's docs define `least_conn` as passing each request "to the server with the least number of active connections, taking into account weights of servers". That fits chat well. An SSE stream holds its connection open for the whole generation, so "open connections" is a decent proxy for "busy with a stream". Round-robin would happily send a new request to the replica already holding three long streams.

Know its blind spots. It counts connections, not tokens remaining or GPU load, and each Nginx worker process tracks its own counts. Also note what's *missing*: there's no sticky routing (pinning a user to one replica). Stickiness is what you use to hide state kept in process memory. Our API has none, so a user's next message can land on either replica and read the same truth from Postgres.

The gateway config also makes streaming work. `proxy_buffering off` makes Nginx pass the response "to a client synchronously, immediately as it is received"; with buffering on, tokens would arrive in lumps. `proxy_read_timeout 180s` sets the allowed gap *between two reads*, not the total duration. And `proxy_next_upstream off` disables automatic retry to the other replica. Nginx's own docs point out that retrying "is only possible if nothing has been sent to a client yet", and by default it won't retry a POST once it has reached an upstream anyway. Retries belong in the application, which understands idempotency.

### Decision 2: rendezvous hashing to a GPU

Inside Express, the reference ranks the two GPUs per conversation. For each server it computes `sha256(conversationId + serverName)` and sorts highest first. Then it probes each server's `/health` in that order (with a 1.5 s timeout) and uses the first healthy one. This is **rendezvous hashing** (also called highest-random-weight hashing).

A tiny worked example with *illustrative* scores:

| Conversation | score(A) | score(B) | Preferred | If B is down |
|---|---|---|---|---|
| c-17 | 0.82 | 0.31 | A | A |
| c-42 | 0.15 | 0.67 | B | A (fallback) |
| c-99 | 0.54 | 0.91 | B | A (fallback) |

Why it's nice: every API replica computes the same ranking with no shared state and no coordination. When a server disappears, only the conversations that preferred it move. When it returns, they drift back.

Why bother at all? **Locality**. llama-server's `cache_prompt` option (on by default) will "re-use KV cache from a previous request if possible", so "the common prefix does not have to be re-processed". If turn 5 of c-17 lands on the same GPU as turn 4, and that GPU still has turn 4's context in a slot, prefill only covers the new message instead of the whole history. Think of the barista who already knows your order.

Here's the part to say twice in an interview: **locality is an optimization, never correctness**. Express sends the full, authorized history on every call anyway. Landing on the same host doesn't guarantee that llama.cpp kept, or picks, the matching slot (slot choice uses a prompt-similarity heuristic). And locality can fight load balance: three long conversations can all hash to A while B idles. A production scheduler would override the preference when a queue or token budget is exceeded, and log why. The reference scaffold doesn't do that. Never use anything secret as the hash key or put it in logs. A raw bearer token, for example, would leak into every log line.

> 🎬 **Animation — two independent dice rolls:** left panel: Nginx with two API boxes; a new request arrives and a counter over each box shows open connections (api1: 3, api2: 1); the request goes to api2. Right panel: inside api2, the conversation ID "c-42" is hashed against "a" and "b", producing bars 0.15 and 0.67; B wins. Step 3: B turns red; the ranking is walked and A is chosen; the KV cache on A shows "cold" and prefill takes visibly longer. Step 4: B returns; the next c-42 turn goes back to B. Caption: "Different decisions, different components, neither needed for correctness."

## Who are you, and may you read this? The auth boundary {#auth-boundary}

Two words people blur together: **authentication** answers "who are you?", and **authorization** answers "are you allowed to touch *this*?" The design principle: *Express owns both*, because only Express knows which conversation belongs to whom.

Why not just put a password on Nginx? A gateway password gate can let people *in*, but it can't express "Alice may read conversation c-17 and Bob may not". Run two unrelated auth systems (one at the gateway, one in the app) and you get confusing logouts and audit trails that don't line up. In a real company with single sign-on (SSO), an upstream identity proxy may do the *authentication*. Express then consumes the verified identity through a trusted channel, strips any identity headers the client supplied at the edge, and still does the *authorization* on every call.

Here's what the reference implements, step by step:

```text
 login:   POST /api/login {Admin, password}
            ├─ throttle: 20 attempts / minute / process   → 429
            ├─ constant-time compare of password hashes   → 401 if wrong
            ├─ token = 32 random bytes (hex)
            ├─ INSERT sessions(sha256(token), owner, now()+8h)
            └─ Set-Cookie: farm_session=token; HttpOnly; SameSite=Strict; (Secure when enabled)

 every /api request:
            ├─ non-GET must carry Origin == APP_ORIGIN    → 403  (CSRF defence)
            ├─ look up sha256(cookie token), not expired  → 401
            └─ conversation queries always include  WHERE id=$1 AND owner_id=$2
```

The session is **opaque**: the cookie is just a random handle. The browser can't forge "I am Bob" because the identity lives in a Postgres row the browser never sees. Storing only a *hash* of the token means a leaked database backup doesn't hand out live sessions. The `Origin` check blocks cross-site request forgery (CSRF: another website making your browser send a request with your cookie attached). And knowing a conversation's UUID gets you nothing unless your session owns it.

The honest caveat: the reference has **one shared Admin account**. It demonstrates the *mechanism* of server-side sessions and ownership checks. It does not demonstrate multi-tenant isolation. The README says so, and so should you.

The other half of the boundary is the network:

| Boundary | Reference setting | Why |
|---|---|---|
| Gateway | binds `127.0.0.1:8088`; reach it via `ssh -N -L 8088:127.0.0.1:8088 …` | Nothing public in the lab. Real deployments add TLS at the edge. |
| Postgres and API | no published host ports | Only reachable inside the Compose network |
| Inference :8080 | bound to the host's private IP; firewall allows only control hosts | Stops users bypassing Express |
| Inference auth | `LLAMA_API_KEY` bearer key shared by Express and both GPUs | Defence in depth. `/health` stays public by design. |
| Transport | plain HTTP on an isolated lab LAN | Use TLS or mTLS before any shared or untrusted network |

A classic trap the README calls out: Docker-published ports can bypass a host firewall's usual INPUT rules. Verify the filtering on your distribution instead of assuming it. The broader threat model (prompt injection, model output as untrusted input, exfiltration) belongs to **LLM security: treating model output as untrusted**.

## Local by default: choosing a provider without leaking data {#provider-privacy}

The farm can optionally forward a turn to a cloud model (OpenAI or Anthropic). That's a **data-routing** decision: the transcript leaves the building. So the design treats it like opening a door in a secure facility, which takes three separate keys:

```text
 provider = "local" ─────────────────────────────────────────────► local GPUs
 provider = "openai" | "anthropic":
   key 1  server policy:   ALLOW_CLOUD == "true" ?        no → 403 cloud_not_approved
   key 2  per-request:     cloudConsent === true ?         no → 403 cloud_not_approved
   key 3  configuration:   provider API key AND model set? no → 503 <provider>_unconfigured
                                                          all yes → cloud call
```

`ALLOW_CLOUD` defaults to `false`, so a fresh install is local-only. Cloud keys live only on the control host and never reach the browser; the example tells you to remove them from host B's `.env` entirely. The consent flag is part of the request's idempotency fingerprint, so a replay can't quietly flip it.

The iron rule: **a local failure never silently falls back to the cloud**. If both GPUs are down, the user gets a 503, not a surprise export to a third party. Fallback changes where the data goes, and that decision belongs to policy and to the user, not to an error handler.

In a real deployment, "a toggle plus a checkbox" is only the start. Policy should decide per user, data class, destination and purpose. And cloud terms differ by provider and change over time. As one example, OpenAI's current data-controls page says API data is not used for training unless you opt in, and abuse-monitoring logs are kept for up to 30 days, with Zero Data Retention available only with prior approval. Read each provider's current documentation and your contract; don't rely on a blog summary (including this one).

One more thing: local isn't automatically private. Prompts can leak through application logs, backups, browser storage, telemetry and admin access. Define retention and deletion for *every* copy, and log request IDs and outcomes rather than prompt text.

> 🎬 **Animation — three keys to open the cloud door:** a turn marked "provider: openai" walks toward a door labelled "leaves the building". Three padlocks: "ALLOW_CLOUD", "user consent", "key + model configured". Scene A: ALLOW_CLOUD is false, so the first lock stays shut and a 403 bounces back. Scene B: all three unlock and the door opens. Scene C: a "local" turn finds both GPUs red; a dashed arrow toward the cloud door is struck through, and a 503 is returned instead.

## The life of one chat turn {#turn-lifecycle}

A **turn** is one user message plus the model's reply. Following one from click to saved history is where most correctness bugs hide. Three ideas carry the design:

- **Request ID (idempotency key).** The browser generates a UUID per send. If the same request arrives twice (a double-click, or a network retry), the server recognizes it and returns the existing result instead of creating a second turn. Think of a numbered deli ticket: showing it twice doesn't get you two sandwiches.
- **Revision (optimistic concurrency).** Each conversation has a counter. The client says "I last saw revision 6". If the server is at 7, another tab changed things, so the server rejects the send with `409 revision_conflict` and the client reloads.
- **Lease.** A running turn carries an expiry time (150 s). If the process handling it vanishes, the lease lets someone else later declare it dead instead of it being "running" forever.

Here's the flow as the reference implements it:

```text
 browser                 Express                          Postgres                 GPU
   │ POST /chat {requestId, expectedRevision, text, provider}
   │───────────────────────►│ validate, policy, active<4?
   │                        │── BEGIN; SELECT conv … FOR UPDATE ─►│  (short transaction)
   │                        │   expire stale leases               │
   │                        │   same requestId? → replay snapshot │
   │                        │   revision ≠ expected? → 409        │
   │                        │   a turn already running? → 409     │
   │                        │   history+text > 6000 chars? → 413  │
   │                        │   INSERT turn 'running', lease 150s │
   │                        │   revision += 1; COMMIT ───────────►│
   │◄── event: accepted ────│                                     │
   │                        │── POST /v1/chat/completions (stream) ─────────────────►│
   │                        │◄──────────────────────────── data: token ─────────────│
   │                        │── UPDATE assistant_text (lease valid?) ►│             │
   │◄── event: delta ───────│   (saved BEFORE shown)                │             │
   │        …repeat…        │                                       │             │
   │                        │── BEGIN; UPDATE status=complete|partial|error|canceled
   │                        │   WHERE status='running' AND lease valid; revision+=1; COMMIT
   │◄── event: snapshot ────│  (authoritative history)
```

Some details are worth dwelling on.

**Commit before contacting inference.** The acceptance transaction takes a row lock (`SELECT … FOR UPDATE`, which per the Postgres docs blocks other transactions from modifying or locking that row "until the current transaction ends"). That's exactly what you want for a few milliseconds while deciding whether to accept. Held for a 30-second generation, though, it's poison: slow GPUs would pin database connections and locks, and your database's availability would depend on your GPU queue. So the lock is released *before* any network call to a GPU.

**Durable before visible.** The reference writes each chunk of assistant text to Postgres *before* sending it to the browser. That's intentionally expensive, but it means anything a user saw survives a crash. It's a teaching choice; a production system might batch those writes and accept a small loss window.

**One running turn per conversation** is enforced twice: by the check in the transaction, and by a partial unique index (`UNIQUE … WHERE status='running'`), so even a race between two API replicas can't start two generations.

**Conditional terminal update.** The final `UPDATE … WHERE status='running' AND lease_until > now()` only succeeds if this worker still holds a valid lease. If the lease expired and something else already marked the turn `partial` or `error`, the stale worker's write matches zero rows and changes nothing. That's a simple form of **fencing**: stopping a worker that lost ownership from overwriting the newer outcome.

**The stream is not the record.** The final `snapshot` event, or a fresh `GET /api/conversations/:id`, is the truth. After any disconnect, conflict or error, the client re-reads the snapshot.

> 🎬 **Animation — deli ticket and version stamp:** a conversation card showing "rev 6". Step 1: tab 1 sends ticket #A with rev 6; the card flips to "rev 7, turn running". Step 2: the same ticket #A arrives again (double-click); the server returns the existing snapshot and no new turn appears. Step 3: tab 2 sends ticket #B still claiming rev 6; a red "409 conflict" stamp appears and tab 2 reloads to rev 7. Step 4: tokens stream in, each written into the card before appearing in the chat bubble; at the end the status flips to "complete" and the card shows "rev 8".

Now the honest part: the **crash windows**. A process can die after committing "running" but before calling the GPU, or halfway through the stream. The reference recovers these *lazily*: the next request that touches the conversation expires stale leases, marking the turn `error` if no text was saved or `partial` if some was. There's no background worker sweeping for abandoned turns. Stronger recovery needs a background worker, proper lease fencing and a policy for partial results. The README lists this among its "exact limits", and **AI farm, part 3: routing and keeping conversations consistent** walks through the code.

## Why there's no Kafka in version one {#no-kafka}

Sooner or later someone suggests putting Kafka between the API and the GPUs. Kafka is a distributed, durable event log commonly used as a message queue. The instinct is understandable: queues sound like reliability. For *interactive* chat, the first version deliberately says no, and uses plain synchronous HTTP with SSE from Express to llama-server.

The core reason fits in one line: **a queue can make callers wait; it cannot make the GPU faster.** It's a longer line at the coffee shop. People have somewhere to stand, but the barista isn't any quicker. Meanwhile an interactive user needs things a queue makes harder: a first token within seconds, a visible "busy" when overloaded, and cancellation that actually stops work. Put Kafka in the middle and you add brokers, consumers, correlation IDs, lag monitoring, and a *second* channel to stream tokens back to the right browser, with zero extra decode capacity.

| | Interactive chat (v1) | Durable async job (later) |
|---|---|---|
| Example | "Explain this error message" | "Summarize these 40 documents overnight" |
| User is | watching, right now | gone; comes back later |
| Transport | HTTP + SSE, bounded admission | DB outbox + worker, or a queue |
| Overload behaviour | fast 429/503 | wait in the queue (acceptable) |
| Needs | timeouts, cancellation, snapshot reconcile | leases, idempotency, retry limits, dead-letter handling, per-conversation ordering |

When a real asynchronous requirement shows up, the standard pattern is the **transactional outbox**. In the *same* database transaction that records the job, you insert a "please process this" row. A worker claims rows under a lease, does the work, and records the result. Because job and outbox row commit together, you can never have one without the other. Kafka becomes one possible transport once event volume, replay, or multiple independent consumers justify it.

The subtle trap: "Kafka has exactly-once semantics, so our LLM calls happen exactly once." No. Kafka's own design docs explain that when a consumer writes to an external system, you must coordinate the consumer's position with what's stored as output. Classically that takes a two-phase commit, or more simply storing the offset "in the same place as its output". An LLM call is an external side effect that you can't roll back. You get *effectively once* only by building idempotency around the call yourself, which the request ID and conditional updates already do.

> 🎬 **Animation — the longer line:** two panels side by side. Left, "HTTP + admission": 4 slots, arrivals above capacity bounce off with 429 within 50 ms, and admitted requests stream tokens immediately. Right, "Kafka in front": the same 4 slots, but arrivals pile into a growing log; a clock above the oldest message ticks past the 4 s TTFT target, then past 120 s; the user's browser tab closes, yet the message is still consumed and generated for nobody. Caption: "Same GPU, same throughput, worse experience."

## When things break, and how you'd know {#failures-and-signals}

Retries look like free reliability until you've decided what each failure should *look like*. A useful failure matrix answers three questions per row: what does the user see, what does Postgres keep, and who, if anyone, may retry?

| Failure | User sees | Postgres keeps | Retry? |
|---|---|---|---|
| GPU A unhealthy **before dispatch** | normal reply from B (maybe a slower first token, since B's cache is cold) | one accepted turn | Rendezvous order falls through to the next healthy node. Both down → 503. |
| GPU fails **after tokens were shown** | stream ends with `partial` (or `error` if nothing arrived) | text saved so far, status `partial` | **Never silently.** User sends a new turn. |
| API replica dies mid-stream | broken stream; reload shows the snapshot | `running` until the lease expires, then `partial`/`error` when the conversation is next touched | New request via the other replica. Nginx won't auto-retry. |
| Postgres down | login and sends fail (503) | n/a | Fail closed. Never keep improvised local history. |
| Gateway or control host down | whole service unavailable | untouched (on disk) | Two GPUs don't help here. |
| Client disconnects or cancels | n/a | `canceled` with whatever text was saved | Upstream request aborted best-effort |
| Over capacity | fast 429 | nothing | Client may retry later, with backoff |

Why never retry after tokens were shown? Sampling is random, so a second generation starts differently. Glue it onto the first half and you get a Frankenstein answer: "The capital of France is Par— Berlin is a city in Germany…". Even before any token is visible, an upstream timeout is ambiguous, because the GPU may already be computing. So any retry must be bounded and must keep **one** logical generation identity.

Notice the second-to-last row, because it's the classic interview gotcha: **two GPU replicas are not high availability**. The gateway, the API tier's host, and the single Postgres instance are each a separate failure domain. The reference has one control host and no database replication. It relies on backups and restore drills, which are *recovery*, not *failover*. Say so explicitly.

> 🎬 **Animation — failure matrix walk-through:** the topology diagram with a "fault" cursor. Click GPU A before dispatch: the arrow re-routes to B and the turn completes. Click GPU A mid-stream: the chat bubble freezes with an amber "partial" badge, and Postgres shows the saved text. Click api1 mid-stream: the bubble shows "connection lost, reloading…", then the snapshot appears. Click Postgres: every request turns red with 503. Click the control host: everything left of the LAN goes dark while both GPUs stay green, with the caption "healthy GPUs, dead service".

### Signals to watch

You can't operate what you can't see. Track these, correlated by request ID across gateway, API and GPU logs, *without* logging cookies or prompt text by default:

| Signal | Why it matters | Where |
|---|---|---|
| Time to first token (p50/p95) | The user-perceived "is it alive?" metric | API timing |
| Inter-token latency | Slows as slots contend | API timing |
| Prompt vs generation tokens/s | Separates prefill cost from decode cost | llama-server `/metrics`: `prompt_tokens_seconds`, `predicted_tokens_seconds` |
| Requests processing / deferred | Deferred > 0 means requests are queued waiting for a slot | `requests_processing`, `requests_deferred` |
| 429/503 rate | Your admission policy biting | API logs |
| Terminal outcomes | Share of `complete` / `partial` / `error` / `canceled` | Postgres `turns.status` |
| GPU memory, `/health` | Readiness. `/health` returns 503 while loading and 200 when ready; readiness is **not** spare capacity | `nvidia-smi`, `/health` |

llama-server only exposes `/metrics` when started with `--metrics`. The reference compose file doesn't enable it, so adding it is a monitoring decision for you to make. Also remember that a `complete` status means the stream ended normally, not that the answer is good, and the scaffold doesn't store the upstream `finish_reason` (whether the model stopped naturally or hit the 512-token cap). Measuring answer quality is its own discipline, covered in **Evaluation and observability: knowing whether it actually works**.

Finally, validate outward in three layers: (1) GPU and model, by loading it, calling `/health`, making one authenticated completion from the control host, and confirming an unauthenticated one is rejected; (2) the full path, meaning login, new conversation, streamed reply, reload, duplicate request, stale revision, and reading another owner's conversation; (3) injected failures from the matrix above, comparing what the user saw with what Postgres recorded each time. These are *proposed* operator checks. Part 2 gives the exact commands, and none of them were run while writing this guide.

# Interview

## Question

A department of about 100 people wants a private chat assistant on two GPU servers. Someone proposes sticky sessions, Kafka between the API and the GPUs, and automatic fallback to a cloud model when the local GPUs are busy. Walk me through your first design, how you'd size it, and which of those three proposals you'd accept.

## Answer

I'd start by pinning down the contract rather than the headcount: peak arrival rate, prompt and output length distributions, a TTFT and total-latency target, the data classes involved, and what should happen with one server down. "100 people" doesn't size anything; requests per second and tokens per request do.

The architecture: an Nginx gateway in front of two stateless Express replicas, which own login, sessions, conversation ownership, turns, revisions, provider policy and every Postgres write. Behind them sit two llama-server replicas running the *same* pinned Qwen GGUF, with identical hash, template and flags under one alias. Postgres is the source of truth. GPU KV caches are disposable warmth.

Sizing: for Qwen2.5-7B at Q3_K_M the weights are about 3.8 GB. KV with GQA is 2 × 28 layers × 4 KV heads × 128 × 2 bytes = 56 KiB per token, so two 4,096-token slots cost about 448 MiB. Memory isn't the binding limit; decode speed is. I'd load-test per-stream speed with both slots busy, compute service time, and apply Little's law. With illustrative numbers (30 tok/s, 300-token answers, 1 s prefill), each request takes about 11 s, so 4 slots give about 0.36 req/s. That's fine at 0.2 req/s, but one host alone gives about 0.18 req/s, which is overloaded. So either I buy headroom or I return fast 429/503s in degraded mode.

On the three proposals. **Sticky sessions: no.** They exist to hide process-local state, and our API has none. I'd use least_conn, which suits long-lived SSE connections, and optionally rendezvous-hash each conversation to a preferred GPU for prompt-cache locality, always sending full history, so locality is never needed for correctness. **Kafka: not for v1.** A queue doesn't add decode capacity. It adds latency, lag and a second channel for streaming tokens back to the browser. Interactive chat gets synchronous HTTP/SSE with bounded admission and short DB transactions committed *before* inference. If durable async jobs appear later, I'd use a transactional outbox with leases and idempotency, and I'd point out that Kafka's exactly-once doesn't extend to an external LLM call. **Automatic cloud fallback: no.** It silently changes where the data goes. Cloud needs a server-side policy switch, per-request consent and configured credentials, and a local outage returns 503.

Last, I'd be explicit about limits. Two GPUs aren't HA while the gateway, API host and Postgres are single instances. Admission in the reference is replica-local, not global. After tokens have been shown there's no silent retry: the turn ends `partial`, and the client reconciles from the authoritative snapshot.

## Follow-ups

- How would you build a *global* admission controller across both API replicas, and would you count requests or tokens?
- A worker's lease expired but it's still streaming. Exactly which writes must be fenced, and how does the conditional UPDATE do it?
- How would you upgrade the model on both hosts without ever serving one conversation from two different model versions?
- Locality has sent three long conversations to GPU A while B idles. What signal would trigger overriding the hash, and what do you log?
- Which privacy controls apply to backups, logs and the llama-server diagnostic UI, not just to the cloud path?

# Pitfalls

- **Sizing by headcount.** "100 users" says nothing about load. Size from peak arrival rate and token distributions, and check the degraded single-host case.
- **Trusting the alias.** Two servers both answering as `corp-qwen` doesn't mean identical weights, template or settings. Compare hashes and manifests.
- **Sticky sessions to paper over state.** If you need stickiness for correctness, state is hiding in process memory; move it to Postgres.
- **Treating cache locality as consistency.** Rendezvous hashing improves prefill reuse at best. History must be sent from Postgres every time.
- **Holding a DB transaction or row lock across generation.** It turns slow GPUs into a database outage. Commit, *then* call inference.
- **Silent cloud fallback.** A local failure that exports the transcript is a privacy incident, not resilience.
- **Retrying after tokens were shown.** It produces spliced, contradictory answers. End as `partial` and let the user resend.
- **"Kafka exactly-once covers it."** Broker semantics don't make an external, non-transactional LLM call happen exactly once.
- **Calling two GPUs "high availability"** while the gateway, API host and database are single instances.
- **Treating a character cap as a token guarantee.** 6,000 characters can exceed a 3,584-token prompt budget for dense code or some scripts.

# Checklist

- Write the service contract (arrival rate, token distributions, TTFT/latency targets, degraded-mode behaviour) before choosing hardware.
- Compute weight and KV memory for your model from its config (`L`, `h_kv`, `d_head`, bytes) and the real file size.
- Turn measured per-stream decode speed into service time, slot throughput and Little's law occupancy, for both the normal and the one-host case.
- Explain why least_conn with no stickiness is correct for stateless Express, and why rendezvous hashing to a GPU is optional.
- Draw the auth boundary: opaque server-side sessions, Origin checks, owner-scoped queries, and inference ports reachable only from control hosts.
- State the three conditions that must all hold before a turn may go to a cloud provider, and why local failure never falls back.
- Walk through a turn: idempotency key, expected revision, short locked transaction, commit, stream with durable-before-visible writes, conditional terminal update, snapshot reconcile.
- Argue for synchronous HTTP/SSE over Kafka in v1, and describe the outbox pattern for real async jobs.
- Fill in a failure matrix (user view, stored state, retry policy) and name the monitoring signals, including llama-server's deferred-requests metric.

# Sources

- [llama.cpp server README (ggml-org)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — Flags (`--parallel`, `--ctx-size`, `--kv-unified`, `--context-shift`, `--n-predict`, `--alias`, `--api-key`/`LLAMA_API_KEY`, `--metrics`), public `/health` behaviour, `cache_prompt` prefix reuse, `--slot-prompt-similarity`, f16 default KV type, and metric names. The master branch changes; pin and re-check your tested build.
- [Qwen2.5-7B-Instruct-GGUF model card (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) — 7.61B parameters, 28 layers, 28 Q / 4 KV heads, available quantizations, the Q3_K_M size (~3.81 GB), and the Apache-2.0 licence.
- [Qwen2.5-7B-Instruct config.json (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) — hidden size 3584, 28 attention heads, 4 key/value heads and 28 layers, used for the KV arithmetic.
- [Qwen docs: llama.cpp](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html) — What GGUF contains and llama-server's OpenAI-compatible API (the page now focuses on Qwen3).
- [NGINX upstream module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html) — Definition of `least_conn` (fewest active connections, weighted) and consistent hashing.
- [NGINX proxy module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html) — `proxy_buffering off` streaming, `proxy_read_timeout` measured between reads, and `proxy_next_upstream` retry limits for non-idempotent requests.
- [PostgreSQL 18: explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html) — `SELECT … FOR UPDATE` row-lock semantics, which serialize the short acceptance transaction.
- [Apache Kafka design: message delivery semantics](https://kafka.apache.org/41/design/design/) — Exactly-once to external systems requires coordinating consumer position with output, via two-phase commit or offsets stored alongside output.
- [OpenAI API: data controls](https://developers.openai.com/api/docs/guides/your-data) — Current example of provider data handling: no training on API data by default, abuse-monitoring retention, and Zero Data Retention by approval.
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — Driver prerequisite and `nvidia-ctk runtime configure --runtime=docker` for GPU access from Docker.

# Flashcards

## farm-owner

**Q:** In the farm design, which component owns durable conversation state, and what state do the GPUs hold?

Express owns every Postgres write: sessions, ownership, turns, revisions and statuses. Postgres is the source of truth. The llama-server replicas hold only ephemeral KV and prefix caches, which are scratch work that speeds up the next prefill. Filing cabinet vs whiteboard: wiping a GPU's cache costs speed, never history, because Express rebuilds the prompt from Postgres on every turn.

## farm-api-affinity

**Q:** Why don't the Express replicas need sticky sessions?

Stickiness exists to hide state that lives in one process's memory. Our replicas are stateless: the session, ownership and history are all read from Postgres on each request, so either replica gives the same answer. Needing stickiness for *correctness* is a smell that state is in the wrong place.

## farm-least-conn

**Q:** What does Nginx least_conn measure, why does it suit chat, and what does it miss?

It sends each request to the upstream with the fewest active connections (weighted). SSE streams keep connections open for a whole generation, so open connections roughly track "busy streaming", which is better than round-robin. It does *not* see tokens remaining, prompt size or GPU load, and each Nginx worker counts separately. It's a coarse balancer, not a GPU scheduler.

## farm-locality

**Q:** What does rendezvous hashing of (conversationId, server) buy you, and what doesn't it guarantee?

Every API replica independently ranks the GPUs the same way for a given conversation, and only the affected conversations move when a server leaves. Sending a conversation back to the same GPU raises the chance that llama.cpp's prompt cache can skip re-reading the shared prefix. It guarantees nothing: the slot may have been reused, health failover moves the conversation, and it can unbalance load. Full history is always sent, so it's a performance hint, never correctness.

## farm-memory

**Q:** Why isn't the model file size enough to size a GPU?

VRAM holds weights, plus KV cache for every token held across all slots, plus runtime and compute buffers. KV grows with slots × context per slot and can dominate. Qwen2.5-7B at 8 slots × 32,768 tokens needs about 14 GiB of KV against about 3.55 GiB of Q3_K_M weights. Also, nominal quant names mislead: Q3_K_M works out to about 4 effective bits/param, so use the real file size and read the runtime's allocation log.

## farm-kv

**Q:** Estimate KV-cache bytes per token for Qwen2.5-7B and for one 4,096-token slot.

2 (K and V) × L × h_kv × d_head × bytes = 2 × 28 × 4 × 128 × 2 = 57,344 B = 56 KiB per token at f16. One 4,096-token slot is 224 MiB, and two slots are 448 MiB. GQA (4 KV heads instead of 28) is why it's 7× smaller than it would otherwise be. Confirm against the actual runtime allocation.

## farm-throughput

**Q:** How do you turn measured decode speed into a request-rate ceiling?

Service time S ≈ prefill time + output tokens ÷ per-stream speed (measured with all slots busy). The ceiling is slots ÷ S. Example (illustrative): 1 s + 300/30 = 11 s, and 4 slots ÷ 11 s ≈ 0.36 req/s. Little's law, L = λW, then gives average occupancy: 0.2 req/s × 11 s = 2.2 slots. Averages don't prove p95, so load-test bursts and long prompts too.

## farm-failure-capacity

**Q:** What happens to capacity when one of the two GPU hosts fails, and what are your options?

Slots halve, so the ceiling halves (4/11 → 2/11 ≈ 0.18 req/s in the example). If arrivals (0.2 req/s) now exceed it, the queue grows without bound until timeouts fire. The options are spare capacity, keeping peak load below one host's measured budget, or fast explicit 429/503 rejection. Pretending the queue will drain is not an option.

## farm-auth

**Q:** Why isn't a password gate at Nginx enough, and how does the reference authenticate and authorize?

A gateway gate lets people in but can't say who owns which conversation. The reference logs in at Express. A random 32-byte token goes in an HttpOnly, SameSite=Strict cookie, and only its SHA-256 is stored in Postgres with an 8-hour expiry. Non-GET requests must carry the expected Origin (CSRF defence). Every conversation query is scoped `WHERE id = $1 AND owner_id = $2`. It's a single shared Admin account, so it shows the mechanism, not multi-tenant isolation.

## farm-cloud

**Q:** Why must a local failure never fall back to a cloud model, and what gates a cloud call?

Fallback changes where the transcript goes, which is a privacy decision, not an error-handling one. A cloud call requires all three of: `ALLOW_CLOUD=true` on the server, `cloudConsent: true` on that request, and a configured provider key and model. Otherwise the call gets 403 or 503. If both GPUs are down, the answer is 503.

## farm-transaction

**Q:** Why does Express commit the accepted turn *before* calling inference?

The acceptance step needs a row lock (`SELECT … FOR UPDATE`) to check the revision, idempotency and "one running turn" safely, but only for milliseconds. Holding that lock and connection for a 30 s generation would tie database availability to GPU speed. Committing first makes the request durable. The cost is a crash window between commit and completion, which leases and stale-turn recovery handle.

## farm-stream-retry

**Q:** Why never automatically retry a generation after tokens have reached the user?

A new sample diverges from the old one, so splicing produces a contradictory answer, and the user already saw the first half. Mark the turn `partial` (or `error` if nothing was saved) and let the client reconcile from the snapshot and resend. Even before visible output a timeout is ambiguous, so retries must be bounded and keep one logical generation identity. That's why Nginx `proxy_next_upstream` is off.

## farm-kafka

**Q:** Why leave Kafka out of the first interactive version, and when would a durable queue be justified?

A queue makes callers wait but adds no decode capacity. It also adds brokers, lag, correlation and a second channel for streaming tokens back. Interactive chat wants fast feedback, cancellation and explicit 429/503. A durable queue (or a DB outbox plus worker) is justified for work that must survive disconnects and wait: batch summaries, replay, multiple consumers. Even then, Kafka's exactly-once doesn't cover an external LLM call; you need idempotency around it.

## farm-ha

**Q:** Why are two inference replicas not "high availability" for the service?

The gateway, API host and single Postgres instance are separate failure domains. In the reference, one control host carries all three. Two healthy GPUs behind a dead control host is still an outage. Backups and restore drills give recovery, not automatic failover, and each tier needs its own redundancy design.

## farm-model-identity

**Q:** What must match across both inference hosts, and why isn't the shared alias enough?

The GGUF file hash and revision, tokenizer and chat template, inference flags and runtime image digest. `corp-qwen` is only a label either host could claim. If the hosts diverge, a conversation bouncing between them gets answers from two different models without anyone noticing. Keep a redacted manifest, and upgrade one host at a time with routing drained.

## farm-admission

**Q:** How does the reference API bound concurrency, and what's the limitation?

Each Express replica admits at most 4 active chats and returns 429 beyond that, but the counter is in-process. Two replicas can admit 8 against 4 GPU slots, and the overflow waits inside llama-server (visible as `requests_deferred`) under a 120 s timeout. It's best-effort and replica-local, not a global semaphore. Real cluster admission needs shared state and ideally token-aware budgets.
