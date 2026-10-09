---
{
  "slug": "farm-routing-consistency",
  "title": "AI farm, part 3: routing and keeping conversations consistent",
  "category": "ai-farm",
  "summary": "Walk through the reference Express gateway line by line: how any request can land on any server, retries and crashes never produce a double or silently-spliced answer, and PostgreSQL stays the single source of truth for every conversation.",
  "difficulty": "Systems",
  "minutes": 35,
  "prerequisites": ["farm-blueprint", "farm-deployment", "inference-scheduling", "llm-security"],
  "learningObjectives": [
    "Separate the four identities in a chat request (session, conversation, request ID, inference node) and explain what each one is allowed to decide.",
    "Trace a chat turn through reservation, streaming, and finalization, including how the revision number and the lease change at each step.",
    "Explain why idempotency is checked before the revision, and what a retry can and cannot recover after a dropped stream or a crashed process.",
    "Calculate rendezvous-hash placement and replica-local admission limits, and defend why neither one is needed for correctness.",
    "Defend explicit cloud consent and honest terminal statuses (complete, partial, error, canceled) against 'just retry it' or 'just fail over' designs."
  ]
}
---

# Sections

## The one question this article answers: who decides what a conversation says? {#who-owns-truth}

In **AI farm, part 1: designing a two-server Qwen deployment** we sketched the farm, and in **AI farm, part 2: deploying it with Docker, step by step** we stood it up. This part zooms in on the code that sits in the middle, the tiny Express app in `examples/ai-farm/api/`, and on one question that sounds boring until it bites you in production: **when any request can land on any server, and any server can die at any moment, who decides what a conversation actually contains?**

Here's the cast. A browser talks to **Nginx**, a reverse proxy (a server that accepts HTTP connections and forwards them to backend servers). Nginx spreads requests across two copies of our API, `api1` and `api2`. Each copy runs **Express**, the Node.js web framework holding all the application logic. Behind them sit two **inference servers**, A and B. Each runs **llama.cpp** (an open-source program that loads a model file and serves it over HTTP) with the *same* Qwen model file, the same tokenizer, and the same chat template. Finally, one **PostgreSQL** database (Postgres for short) stores everything that matters.

```text
                        ┌──────────── control host ─────────────┐
 browser ──HTTPS*──►  Nginx :8088 ──least_conn──┬─► api1 :3000 ─┐ │
                        │                        └─► api2 :3000 ─┤ │
                        │                                        ▼ │
                        │                                  Postgres│  ◄── the ledger
                        └────────────────────────────────────────┬─┘
                                  api1/api2 also call ────────────┤
                                                                  ▼
                         inference A :8080  (llama.cpp, Qwen GGUF)  ◄─ same model
                         inference B :8080  (llama.cpp, Qwen GGUF)  ◄─ same model

 * the lab publishes plain HTTP on loopback; real deployments terminate TLS
```

The whole design rests on one decision: **Postgres is the ledger, and everyone else is a clerk.** Think of an old bank branch. Tellers (the Express replicas) can come and go, and any teller can serve you, because none of them keeps your balance in their head. They read the ledger, propose an entry, and only what gets written into the ledger counts. The vault staff (the inference servers) do heavy work on request, but they don't decide anything about your account.

Concretely, Express is the only thing that writes to Postgres, and it owns authentication, conversation ownership checks, turn records, revisions, request IDs, provider choice, and logs. The inference servers are **stateless compute**: they get a fully built prompt, and they send back tokens. The only state they keep is an ephemeral cache. The main one is the **KV cache**, the saved attention keys and values for text the model has already read, which lets a server skip re-reading a prompt prefix it's seen before. (We'll go through this in detail in **What happens at inference: prefill, decode, and the KV cache**.) Losing that cache makes the next request slower. It never makes a conversation wrong.

| Component | Owns | May lose on restart without harm |
|---|---|---|
| Nginx | which API replica gets a connection | in-flight connections |
| Express (`api1`, `api2`) | rules: auth, ownership, turn lifecycle, provider policy | nothing durable; an in-flight stream can be interrupted |
| Postgres | **truth**: sessions, conversations, turns, revisions | nothing (it's the thing we back up) |
| llama.cpp A/B | compute; ephemeral KV/prefix cache | cache warmth only |

> 🎬 **Animation — the ledger and the clerks:** start with the topology above. Step 1: a chat message enters the browser and travels to Nginx, which lights up api2 (fewest connections). Step 2: api2 writes a "running" row into a big ledger icon labelled Postgres *before* anything else moves. Step 3: api2 sends the prompt to inference A; tokens flow back, and each token first drops into the ledger and only then continues to the browser. Step 4: api2 explodes (crash icon). Step 5: the browser's next request goes to api1, which reads the ledger and shows the exact saved text. Caption: "the clerk died; the ledger didn't."

One reminder before we start. Everything in `examples/ai-farm/` is an **educational Linux Docker scaffold**, not this learning app's infrastructure. It passed syntax checks and a small SSE-parser test (`api/test-stream.mjs`), but nobody ran it against a live GPU, Docker, or paid provider while writing this. Where it's a teaching simplification, I'll say so.

## Four IDs you must never mix up, and how the caller proves who they are {#identities-and-auth}

Most bugs in multi-server chat systems come from treating one identifier as if it were another. There are four, and each one answers exactly one question:

| Identity | Answers | Created by | Lives in |
|---|---|---|---|
| **Session** | "Who is calling?" | server at login (random token) | cookie in browser, hash in `sessions` |
| **Conversation ID** | "Which ordered history?" | server (`randomUUID()`) | `conversations` table |
| **Request ID** | "Which attempt at one user turn?" | browser (`crypto.randomUUID()`), once per message | `turns.request_id` |
| **Inference node** | "Where does compute run this time?" | config (`a`, `b`) | nowhere durable; chosen per turn |

Three classic mistakes fall straight out of this table. Routing by a cookie doesn't prove ownership of anything. An HTTP connection isn't a durable turn, because it can vanish mid-answer. And a warm prefix cache isn't a history database, because it can be evicted at any time. One more rule from the farm contract: **never hash a raw bearer token (or session cookie) into a routing key or a log line.** A hash of a secret is still a stable fingerprint of it that you've now scattered across logs.

### Authentication: who are you?

The scaffold has exactly one account, `Admin`, whose password comes from the `ADMIN_PASSWORD` environment variable (the server refuses to start if it's shorter than 16 characters). Login works like this:

1. The server hashes the supplied password and the real one with SHA-256 and compares them with `timingSafeEqual`, a comparison that takes the same time whether the first byte or the last byte differs. That way response timing can't leak how close a guess was.
2. On success it creates a **session token**: 32 random bytes (256 bits) as hex. It's **opaque**, meaning the string carries no information itself and only means something when the server looks it up.
3. Postgres stores only the **SHA-256 hash** of the token, plus the owner (`'admin'`) and an expiry eight hours out. A database leak therefore doesn't hand out live cookies.
4. The browser gets the token in a cookie marked `HttpOnly` (page JavaScript can't read it), `SameSite=Strict` (the browser won't attach it to requests that start on another site), `Max-Age=28800` (8 h × 3,600 s = 28,800 s), and `Secure` when `COOKIE_SECURE=true` (only sent over HTTPS).

Every `/api/*` request then hashes the cookie, looks the hash up with `expires_at > now()`, and attaches `req.owner`. Because the lookup is in Postgres, **either replica can authenticate any request**. There's no sticky session and no in-memory login state.

There's also a blanket rule: every non-GET request must carry an `Origin` header exactly equal to `APP_ORIGIN`. Browsers set `Origin` automatically on cross-origin and POST requests, so this blocks **cross-site request forgery** (CSRF: another website making your browser send a cookie-bearing request to our app). It applies to login too. A crude login throttle allows 20 attempts per minute *per process*, which is a lab-only control.

### Authorization: is this conversation yours?

Authentication says who you are. **Authorization** says whether you may touch *this* thing. Every conversation query includes both the ID and the owner:

```sql
SELECT * FROM conversations WHERE id = $1 AND owner_id = $2 FOR UPDATE
```

An unknown ID and someone else's ID both return the same `404 not_found`, so an attacker can't probe which IDs exist. The browser never sends an owner field, and if it did, nobody would read it. `$1` and `$2` are **parameterized SQL**: values travel separately from the query text, so user input can't turn into SQL.

```text
 request ──► [ Origin == APP_ORIGIN? ] ──no──► 403 origin_required
               │ yes (or GET)
               ▼
            [ cookie → sha256 → sessions row, not expired? ] ──no──► 401 login_required
               │ yes: req.owner = 'admin'
               ▼
            [ conversations WHERE id AND owner_id ] ──none──► 404 not_found
               │ found
               ▼
            only now: read history, reserve a turn, pick a node
```

Be honest about the limits: this is **one shared demo account**. Two people logged in as Admin have the same owner and can read each other's conversations if they know the IDs. It shows the *shape* of authorization, not corporate identity isolation. Production needs SSO, distinct immutable user IDs, revocation, audit, and a shared rate limiter. The injection side of the security story (why model output is untrusted, and why the client renders it with `textContent` rather than as HTML) is covered in **LLM security: treating model output as untrusted**.

## Writing the user's message down before any model runs {#reserve-the-turn}

Here's the heart of the design. **Before a single token is generated, the user's message is committed to Postgres as a turn with status `running`.** If everything after that point burns down, we still know the user asked something, and we can say honestly what happened to it.

Picture a restaurant. The waiter writes your order on a ticket and pins it to the rail *before* the kitchen starts cooking. If the cook quits halfway through, the ticket is still on the rail, and the manager can tell you "your soup was half made, sorry" instead of "what soup?"

A chat request body looks like this:

```json
{
  "requestId": "a030f1ce-b5bf-49ec-b07d-638057abf913",
  "expectedRevision": 4,
  "text": "Explain our deployment boundaries",
  "provider": "local",
  "cloudConsent": false
}
```

The server validates the shape first: `requestId` must be a UUID, `expectedRevision` an integer, and `text` non-empty and at most 2,000 characters. Then it opens a **short transaction** that does everything in this order:

```text
BEGIN; SET LOCAL lock_timeout = '3s'
 1. SELECT conversation ... FOR UPDATE           ← take the conversation's row lock
 2. recover any expired 'running' turn           ← (lease section, below)
 3. request_id already here?  same fingerprint → return snapshot (replay)
                              different        → 409 idempotency_key_reused
 4. revision ≠ expectedRevision?                → 409 revision_conflict
 5. any turn still 'running'?                    → 409 generation_active
 6. build prompt from COMPLETE turns + new text; > 6,000 chars? → 413
 7. INSERT turn (status 'running', lease_until = now() + 150 s)
 8. UPDATE conversations SET revision = revision + 1
COMMIT                                           ← lock released, connection returned
 ...only now: contact inference
```

### The row lock: a single-occupancy meeting room

`SELECT ... FOR UPDATE` takes a **row-level lock**. The PostgreSQL docs say it blocks other transactions that try to `UPDATE`, `DELETE`, or `SELECT ... FOR UPDATE` the same row until the current transaction ends. Think of a meeting room with one key: whoever holds it makes the decision, and everyone else waits in the hallway (for up to 3 s here, thanks to `lock_timeout`). That gives each conversation a **serialization point**: decisions about one conversation happen one at a time, even across two API replicas, because both replicas lock the same Postgres row.

A second guard sits in the schema:

```sql
CREATE UNIQUE INDEX one_running_turn ON turns(conversation_id) WHERE status='running';
```

That's a **partial unique index**, a uniqueness rule that applies only to rows matching the `WHERE`. So even if the code had a bug, the database would refuse a second running turn in the same conversation.

### Why the transaction closes before inference

It's tempting to hold the transaction open while the model generates, since it's "one operation". Don't. Generation takes seconds to minutes, and the external call can hang. Holding the transaction would pin a database connection (the pool has only 12) and hold the lock the whole time, so a simple GET of that conversation would stall behind it. The rule from the contract is **never hold a DB transaction or connection across inference.** Later writes during streaming are small, independent statements.

### The revision: a wiki-style edit guard

The `revision` column is an **optimistic concurrency** counter. "Optimistic" means we don't lock anything while the user is typing. We just check at submit time that nothing changed since they last looked, the same way a wiki says "someone else saved this page after you opened it." The browser sends the revision it last saw as `expectedRevision`.

The revision advances on every **committed turn-state transition**: reservation, terminal completion, and stale-lease recovery. It does *not* advance per token. Here's the arithmetic for a fresh conversation where everything goes well (the new turn's `ordinal`, its sort key, is set to the revision plus one at reservation):

| Event | Revision after | Turn ordinal |
|---|---|---|
| conversation created | 0 | — |
| turn 1 reserved | 1 | 1 |
| turn 1 finished (`complete`) | 2 | — |
| turn 2 reserved (expected 2) | 3 | 3 |
| turn 2 finished | 4 | — |

So each happy turn moves the revision by exactly 2, and ordinals come out as 1, 3, 5…, which is sparse but strictly ordered and unique. Now open two tabs, both showing revision 4, and send different messages from each. Whichever transaction gets the lock first sees 4 = 4, reserves, and moves the revision to 5. The second sees 5 ≠ 4 and gets `409 revision_conflict`. It must reload the authoritative snapshot and let the human decide what to send. Nobody's message silently lands in a history they didn't see.

> 🎬 **Animation — two tabs racing for revision 4:** two browser tabs side by side, both showing "rev 4". Step 1: both send at once; two arrows reach a Postgres row with a padlock. Step 2: Tab 1's arrow grabs the padlock; a counter flips 4→5 and a new row "ordinal 5, running" appears. Step 3: the padlock releases and Tab 2's arrow enters, compares "expected 4" with "actual 5", and bounces back with a red "409 revision_conflict". Step 4: Tab 2 fetches the snapshot, now showing Tab 1's message, and the user retypes.

One subtlety: because the revision doesn't move per token, **a revision is not a stream offset.** It can't tell you how many characters of a running answer you have. For that you fetch the snapshot and read `assistant_text`.

## Retrying without getting two answers: request IDs and fingerprints {#idempotent-retries}

Networks lie by omission. If your `fetch` throws, you don't know whether the request never arrived, arrived and was reserved, or is streaming happily to a connection you just lost. **Idempotency** is how we make "try again" safe: doing the same operation twice has the same effect as doing it once. (The general pattern of idempotent retries for side-effecting actions comes up again in **Agents and tools: from model decisions to safe actions**.)

The mechanism has two parts:

- **Request ID.** The browser generates one UUID *per user message* and keeps it (in `localStorage`, as `farmPending`) until it learns the outcome. A retry reuses it. It never mints a fresh ID just because `fetch` threw.
- **Fingerprint.** The server stores `sha256(JSON.stringify([text, provider, expectedRevision, cloudConsent]))` with the turn. That pins down *what* was asked under that ID.

When a request arrives with an ID that already exists, there are three cases:

| Situation | Server response | Why |
|---|---|---|
| Same ID, same fingerprint | `200` JSON snapshot with `replay: true` | It's a retry. Show the truth, and don't run the model again |
| Same ID, different fingerprint | `409 idempotency_key_reused` | An ID names one immutable attempt; it's not a license to overwrite |
| New ID, stale revision | `409 revision_conflict` | Someone (maybe you, in another tab) moved the conversation on |

### Why idempotency is checked *before* the revision

This ordering is the classic interview probe. Walk through it: the browser sends turn X with `expectedRevision: 4`. The server reserves it, so the revision becomes 5, and then the connection drops. The browser retries X, correctly carrying the *original* `expectedRevision: 4`, because that's part of the fingerprint. If the server checked the revision first, it would see 5 ≠ 4 and reject a perfectly legitimate retry with a conflict. Checking the request ID first recognizes "oh, that's X, I already have it" and returns the snapshot.

```text
 retry of X (expected 4)   ──►  [ X exists? ] ──yes, same fp──► snapshot (no new inference)
                                    │ no
                                    ▼
                               [ rev == 4? ] ──no──► 409 revision_conflict
                                    │ yes
                                    ▼
                                reserve X
```

### What a replay gives you, and what it doesn't

If the retried turn is still `running`, the snapshot says so, and the bundled client polls `GET /api/conversations/:id` every 3 seconds until the status becomes terminal. This is **snapshot reconciliation**, not event replay: you get the saved user text, the saved assistant text so far, and the status, but not the exact sequence of stream events you missed. If you need every token event in order (for audit, or truly resumable streams), you need a durable event table with sequence numbers and a retention policy. An HTTP stream doesn't give you that on its own.

And be precise about what idempotency *doesn't* buy you. It guarantees that **one request ID never starts a second inference once a turn is reserved.** It does *not* give you exactly-once model execution. There's no transaction spanning Postgres and a remote GPU. A process can commit the reservation and crash before calling the model, or call the model and crash while the GPU keeps computing. The system records the ambiguity as an interrupted outcome. It never guesses that re-running is safe.

> 🎬 **Animation — the three repeat cases:** three horizontal lanes. Lane 1: an envelope labelled "X / fp 7a3…" arrives at a table that already holds "X / fp 7a3…"; a green check appears and a snapshot card slides back, while the GPU icon stays grey. Lane 2: an envelope "X / fp 91c…" arrives at the same row; the fingerprints are highlighted in red and a "409 idempotency_key_reused" stamp appears. Lane 3: an envelope "Y, expected rev 4" arrives while a counter shows 5; red "409 revision_conflict". End with the caption "same ID + same content = same answer, never a new generation".

## Streaming the answer: save first, show second {#durable-streaming}

Now the fun part: getting tokens to the browser while they're still being generated. The gateway uses **server-sent events** (SSE), a dead-simple text format for pushing a series of messages down one open HTTP response. Each event is a few `field: value` lines ending in a blank line:

```text
event: delta
data: {"text":"Hel"}
                              ◄── blank line = end of event
event: delta
data: {"text":"lo!"}

```

The browser's built-in `EventSource` only does GET requests with no body, and we need to POST a JSON body, so the client uses `fetch` and parses the stream itself. The trap is that **one network read is not one event.** A read can hold half an event, three events, or split a multi-byte UTF-8 character down the middle. Both the server's upstream parser and the browser's parser therefore:

1. decode bytes with `TextDecoder` in streaming mode (so a split `☕` waits for its remaining bytes),
2. append to a buffer and normalize `\r\n` to `\n` (many providers send CRLF),
3. cut complete events at each `\n\n`, keeping the leftover tail for the next read,
4. join multiple `data:` lines in one event with `\n`, as the SSE rules specify,
5. (server side) refuse a buffer over 131,072 bytes (128 KiB) as `upstream_frame_limit`, so a broken upstream can't grow memory forever.

`test-stream.mjs` feeds a sample stream in 1-, 2-, 3-, 7-, and 128-byte chunks and checks that the parser yields the same three events every time. Those are the only tests the scaffold ships.

### The three events and the persist-then-display rule

The API emits three event types to the browser:

| Event | When | Payload |
|---|---|---|
| `accepted` | right after the reservation commits | `{requestId, revision}` |
| `delta` | each text fragment, **after it's saved** | `{text}` |
| `snapshot` | after the terminal status commits | the full authoritative conversation |

(A fourth, `reconcile`, appears only if finalization itself fails after headers were sent. It tells the client to fetch the snapshot later.)

The key invariant is **durable before visible**. For every fragment, the server first runs

```sql
UPDATE turns SET assistant_text = $3
 WHERE conversation_id = $1 AND request_id = $2
   AND status = 'running' AND lease_until > now()
```

with the *whole accumulated* answer, and only if that updated a row does it write the `delta` to the browser. So anything a user has seen on screen is already in Postgres, and a reload through the other replica will show it. The `WHERE` clause also acts as a **fence** (more on that in the next section): if the turn was already recovered or finalized, zero rows update, and the stream aborts with `lease_lost`.

```text
 upstream token ──► accumulate ──► UPDATE ... WHERE running AND lease valid
                                        │ 1 row                 │ 0 rows
                                        ▼                       ▼
                                  send 'delta' to browser   abort: lease_lost
```

This is deliberately expensive. Suppose an answer arrives as 400 fragments averaging 5 characters, for 2,000 characters total (illustrative numbers). Rewriting the whole accumulated text each time writes 5 + 10 + … + 2,000 characters, which is 5 × (1 + 2 + … + 400) = 5 × 80,200 = 401,000 characters of `assistant_text`, about 200× the final answer, across 400 statements. That's **write amplification**: many writes of mostly the same data. A production version would batch (say, flush every 250 ms or every 200 characters) and *document the loss window*, meaning how much displayed text a crash can lose. Or it would append fragments to an event table. The scaffold picks the simplest invariant to reason about, not the fastest.

> 🎬 **Animation — persist, then display:** a token stream flows from a GPU box towards a browser. Between them sits a Postgres cylinder. Step 1: token "Hel" arrives, drops into the cylinder (the row's text becomes "Hel"), and only then a copy continues to the browser. Step 2: "lo!" does the same; the row shows "Hello!". Step 3: the API process crashes between the cylinder and the browser on token " How"; the row shows "Hello! How" but the browser shows "Hello!". Step 4: the browser syncs and sees "Hello! How" labelled partial. Caption: "the screen can lag the ledger, never lead it".

### Ending honestly: complete, partial, error, canceled

A stream has to **end explicitly**. OpenAI-style streams (and the scaffold's llama.cpp path) finish with a final `data: [DONE]` event. Claude's Messages API finishes with an `event: message_stop`. The scaffold treats those as the only successful ends. If the TCP connection just closes (EOF) without them, that's `truncated_stream`, an interruption, even if lots of text arrived. An in-stream `error` event (Claude documents events like `overloaded_error`) also counts as a failure.

The terminal status is decided like this:

```text
                 terminator seen? ──yes──► complete
                       │ no (error, timeout, EOF, lease lost, output limit)
                       ▼
          browser disconnected? ──yes──► canceled   (code client_disconnect)
                       │ no
                       ▼
          any text saved? ──yes──► partial
                       │ no
                       ▼
                     error
```

Then the server takes the conversation lock again, updates the turn to that status (again fenced with `status='running' AND lease_until > now()`), bumps the revision, and sends the final `snapshot`.

Two rules keep this honest. First, **non-complete turns stay visible** with their label (`[partial / upstream_failed]`) but are **excluded from future prompts**. Only `complete` turns get rebuilt into `messages`. You don't want a half-answer silently treated as a finished exchange next time. (It also keeps the history cleanly alternating user/assistant, which some provider APIs require.) Second, `complete` means "the transport ended properly", **not** "the answer is correct", and not even "the model finished its thought". A model can hit the 512-token output cap and still end the stream normally. OpenAI reports this as `finish_reason: "length"` and Claude as a `stop_reason`, but the scaffold doesn't store either, so a length-truncated answer shows as `complete`. A production adapter should persist the stop reason and show it.

## When a process dies mid-answer: leases and fencing {#leases-and-fencing}

Streaming covers the happy path and a clean cancel. But what if `api2` gets OOM-killed halfway through an answer? Its turn is stuck at `running` forever, and the partial unique index would block every future message in that conversation. We need a way to say "whoever owned this has had long enough."

That's a **lease**: a claim on a piece of work that expires at a fixed time, like a parking meter. The spot is yours until the meter runs out, whether or not you come back. At reservation, the turn gets `lease_until = now() + 150 seconds`, using the **database's clock**, so the two API hosts' clocks don't matter. Separately, the request handler aborts generation after **120 seconds**.

```formula
lease (150 s) = generation timeout (120 s) + finalization margin (30 s)
```

Here the **lease** is how long a `running` turn is protected, the **generation timeout** is the handler's `AbortController` deadline, and the **margin** is the room left for a normal timeout to still write its terminal status before the lease expires. The margin is a design allowance. It isn't a guarantee: a slow or down database can eat it.

**Recovery is lazy.** There's no background sweeper. Whenever *anything* takes the conversation lock (a GET snapshot, a new chat reservation, or a finalization), it first runs:

```sql
UPDATE turns
   SET status = CASE WHEN assistant_text = '' THEN 'error' ELSE 'partial' END,
       error_code = 'lease_expired'
 WHERE conversation_id = $1 AND status = 'running' AND lease_until <= now()
```

and bumps the revision if anything changed.

### A worked crash timeline

```text
 12:00:00  turn reserved; lease_until = 12:02:30; revision 4 → 5
 12:00:20  api2 has saved 400 chars ... and crashes (kill -9)
 12:00:30  browser retries same requestId via api1
           → replay snapshot: status 'running', 400 chars shown, polling every 3 s
 12:02:00  (api2 would have timed out here, but it's dead)
 12:02:30  lease expires
 12:02:33  next poll (GET) takes the lock, finds running + expired
           → status 'partial', error_code 'lease_expired', revision 5 → 6
           → browser shows the 400 chars labelled [partial / lease_expired]
```

No second generation is started automatically. The user can read the partial answer and send a **new** turn (new request ID) if they want to try again.

### Fencing: stopping a zombie from writing

Now the nasty case: `api2` didn't die, it just **paused**. Maybe a long GC pause, or the VM was frozen for a migration. At 12:03:00 it wakes up, gets a token from the model, and tries to save it. Its `UPDATE ... WHERE status='running' AND lease_until > now()` matches **zero rows**, because the turn is already `partial` and the lease is in the past. The write is refused, the zombie aborts with `lease_lost`, and its finalization is fenced the same way. The conditional `WHERE` is the **fencing** check: a guard that stops a worker whose claim has been taken away from writing anything. Without it, the zombie could overwrite the recovered record with text the user never saw arrive.

> 🎬 **Animation — lease timeline with a zombie:** a horizontal time axis from 12:00:00 to 12:03:30. A green bar "lease" spans 12:00:00–12:02:30; a shorter orange bar "timeout 120 s" ends at 12:02:00. Step 1: a crash icon on api2 at 12:00:20; the turn row stays "running, 400 chars". Step 2: the green bar reaches its end and turns grey. Step 3: at 12:02:33 a GET arrow from api1 flips the row to "partial / lease_expired", and the revision goes 5→6. Step 4: at 12:03:00 a ghost api2 fires an UPDATE arrow at the row; it hits a shield labelled "WHERE running AND lease_until > now()" and bounces with "0 rows → lease_lost".

### What this design can't do

Be clear-eyed about the limits. An abandoned conversation can sit at `running` in storage until someone next reads it, even though its expired lease already blocks writes. A scheduled sweeper would make dashboards converge without user activity. The 150-second lease is fixed, so a legitimately long generation can't renew it. Longer jobs need **renewable leases** with an owner token (an "epoch" number that increments on each claim), heartbeats, and fencing on every side effect. And if work must survive the requester disconnecting and **resume automatically**, you've left interactive chat. You need a durable job design: an outbox or work table, workers that claim jobs with leases, idempotency keys, and per-conversation ordering. The farm contract is explicit that putting Kafka (a distributed message log) in the middle doesn't give you any of that for free, and **event delivery alone never gives exactly-once LLM execution.** The first version deliberately stays synchronous: HTTP plus SSE, no queue between Express and inference.

## Picking an inference server: rendezvous hashing as a nice-to-have {#rendezvous-placement}

With correctness nailed down in the database, routing becomes pleasantly low-stakes. There are **two independent routing decisions**, made by different components for different reasons:

```text
 decision 1 (Nginx):    which API replica?      least_conn  → balances open connections
 decision 2 (Express):  which inference server? rendezvous  → prefers the same node per conversation
                                   │                               │
                                   └──── neither affects correctness ┘
                                          Postgres does
```

### Decision 1: Nginx `least_conn`

Nginx's `least_conn` sends each request to the upstream server with the fewest **active connections**. That fits SSE nicely, since a streaming chat holds a connection open for its whole duration. There's no sticky "user X always goes to api1" hashing, because nothing about a user lives in an API process. Note what `least_conn` *doesn't* measure: tokens, GPU load, or prompt length. Two long streams and two instant errors look the same to it.

The gateway config also sets `proxy_buffering off` (so each SSE event is forwarded immediately instead of held in a buffer), `proxy_read_timeout 180s` (above the 120 s generation timeout), and `proxy_next_upstream off`. That last one matters here: Nginx will **not** automatically resend a failed request to the other API replica. We don't want the proxy guessing that a chat POST is safe to repeat. Retries are the client's job, with the same request ID. The flip side is that if `api1` is down, a request routed to it fails, and you drain it from the config or restore it (part 2 covers the procedure).

### Decision 2: rendezvous hashing for inference placement

Why prefer the same inference node for a conversation? Because of **prefix caching**. Turn 5's prompt is turns 1–4 plus the new message, so if the same llama.cpp server handled turn 4 and still has those tokens' KV cache, it can skip re-processing most of the prompt. That's a nice cut in time-to-first-token. (We'll go through prefix caching in detail in **What happens at inference: prefill, decode, and the KV cache**.)

The scaffold uses **rendezvous hashing**, also called *highest random weight* hashing. The picture: every conversation holds a little election. Each node gets a score from hashing the conversation ID together with the node's stable name, and the highest score wins. Here's the code, trimmed:

```javascript
const nodes = [['a', env.INFERENCE_A_URL], ['b', env.INFERENCE_B_URL]].filter(n => n[1]);
nodes.sort((x, y) => hash(id + y[0]).localeCompare(hash(id + x[0])));   // highest score first
for (const [name, url] of nodes) {
  // GET {url}/health with the bearer key, 1.5 s timeout; first healthy one wins
}
throw fail(503, 'no_healthy_local_model');
```

```formula
preferred(c) = argmax over healthy nodes n of  SHA256(c ‖ n)
```

Here `c` is the conversation ID, `n` is the node's stable name (`a` or `b`, deliberately *not* its URL, so changing an IP doesn't reshuffle placement), `‖` is string concatenation, and the 64-hex-digit hashes are compared as strings. For lowercase hex that's the same as comparing them as 256-bit numbers.

**Worked example** (real SHA-256 values, first 12 hex digits shown):

| Conversation ID | score `a` | score `b` | winner | (hypothetical `c`) | winner with c |
|---|---|---|---|---|---|
| `3f2b8c1e-5a7d-4e19-9c0a-1b2c3d4e5f60` | `b53554b8d07c` | `9f7bf81b64bc` | **a** | — | — |
| `1a2b3c4d-0000-4000-8000-000000000001` | `6ba2b5bf2c05` | `dd3a32590f94` | **b** | `02d83c9d7f00` | b |
| `1a2b3c4d-0000-4000-8000-000000000002` | `5d5e1f1582ee` | `e199fa5f3811` | **b** | `8505e6431524` | b |

Now kill B. Conversations 2 and 3 fall back to their runner-up, A. Conversation 1 doesn't move at all, since it never preferred B. That's the defining property: **when a node leaves, only the keys it was winning get reassigned**, and when a node joins, it only steals keys where it now has the top score (none of these three, as it happens). Compare Nginx's plain `hash` method, whose docs warn that adding or removing a server "may result in remapping most of the keys". Nginx offers a `consistent` (ketama ring) option to fix that. Rendezvous gets the same property with no ring and no shared placement table: any replica computes the same answer from the conversation ID alone.

> 🎬 **Animation — the per-conversation election:** three conversation cards on the left, two node boxes A and B on the right. Step 1: for each card, two score bars grow using the table's hex prefixes (b535 vs 9f7b; 6ba2 vs dd3a; 5d5e vs e199); the taller bar's node lights up and an arrow connects the card to it (1→A, 2→B, 3→B). Step 2: node B turns red ("health failed"). Cards 2 and 3 re-point to A (their runner-up) with a "cache cold" snowflake; card 1's arrow doesn't flicker. Step 3: B comes back green and cards 2 and 3 return to it, again cold.

### Why locality is only an optimization

Three reasons this can never be load-bearing:

1. **A cache hit isn't guaranteed.** Even on the "right" node, the prefix can be gone: other requests evicted it, the conversation landed in a different slot, the server restarted, or the prompt formatting changed. So **the full prompt is rebuilt from Postgres on every turn**, and a hit is a bonus.
2. **Health checks are snapshots.** `/health` in llama.cpp is public and says whether the model is loaded (it returns 503 while loading). It doesn't say whether a slot is free, and A can die one millisecond after answering. Failover to B **loses warmth**, not correctness.
3. **"Same model" means more than the same name.** Both nodes answer to `model: "corp-qwen"`, but identity really means the same GGUF file (checked by SHA-256 in part 2), the same quantization, the same tokenizer, and the same chat template. If one host drifts, the transport keeps working and answers quietly change depending on placement.

And one firm rule: **no silent retry on the other node, ever**, whether before or after text appears. Before any text arrives, an error might hide a request that actually ran, and GPU time or cloud money is already spent. After text is visible, retrying elsewhere would splice two *different* random continuations into one answer: sampling is stochastic, so B won't continue A's sentence the way A would have. The scaffold records `partial` or `error`, and the user makes an explicit new turn.

## How much work to let in: admission, timeouts, and context budgets {#admission-and-limits}

Routing decides *where* work goes. **Admission control** decides *whether* it gets in at all. Without it, the GPU becomes the queue: requests pile up invisibly inside llama.cpp, everyone's latency climbs, and nobody gets a clear "busy, try later."

The scaffold keeps a simple in-process counter: **at most 4 active chat handlers per Express process.** The 5th gets `429 replica_busy` immediately, and nothing gets reserved, which is why the client treats 429 as "safe to forget this pending request." Here's how that compares with the farm's real capacity:

```text
 API admission:   api1 [■■■■]  +  api2 [■■■■]            = up to 8 handlers
 GPU slots:       A    [■■]    +  B    [■■]              = 4 concurrent generations
                  (--parallel 2 each; 8,192 ctx / 2 = 4,096 tokens per slot)

 8 admitted  −  4 running  =  up to 4 waiting inside llama.cpp
```

Be precise about what this counter is. It's **replica-local**, not a cluster-wide limit. Because `least_conn` counts connections rather than work, `api1` can be full and rejecting while `api2` sits at 1 of 4. And the combined 8 is more than the 4 GPU slots, so the counter protects the API processes more than it schedules the GPU. A real global limiter, with per-tenant fairness and awareness of slots, needs shared state (Redis, Postgres, or a dedicated scheduler). The farm contract asks guides to state this limitation rather than hide it.

### A back-of-envelope with Little's law

**Little's law** says the average number of requests in flight equals arrival rate × average time in the system:

```formula
L = λ × W
```

Here `L` is the average number of requests in the system, `λ` (lambda) is the arrival rate in requests per second, and `W` is the average time each request spends in the system, in seconds. Picture a coffee shop: 2 customers a minute who each stay 10 minutes means about 20 people inside. We'll go through it properly, with queueing and SLOs, in **Serving many users: batching, scheduling, and speculative decoding**.

Illustrative numbers, not a benchmark: 0.3 requests/s × 10 s average = **3 in flight**, which fits under 4 GPU slots. If answers get twice as long and decode dominates, `W` roughly doubles to 20 s, so L ≈ 6. That's above the 4 slots, so requests start queuing inside llama.cpp, which stretches `W` further. The only way to know your real `W` is to measure time to first token, decode speed, queue time, and total duration on your actual GPU and quantization.

### Timeouts and size bounds

| Limit | Value | What it protects |
|---|---|---|
| JSON body | 16 KiB (Nginx body cap 32 KiB) | parser memory |
| user text | 2,000 characters | one message's size |
| prompt history (complete turns + new message) | 6,000 characters → `413 history_limit_start_new_conversation` | context window |
| output | 512 tokens requested; 16,000 characters hard stop | runaway generation |
| generation timeout | 120 s | stuck upstreams |
| lease | 150 s | stuck `running` turns |
| slow client | abort if Node's write buffer fills (`slow_client`) | server memory |

Notice the mix of **characters** and **tokens**. Characters are cheap to count in JavaScript, but the model's context is measured in tokens (sub-word pieces; see **Text in, next token out: tokens, embeddings, and sampling**). Under a rough English heuristic of about 4 characters per token, 6,000 characters is about 1,500 tokens. Add 512 output tokens and some chat-template overhead, and that fits a 4,096-token slot comfortably. But code, non-English text, or emoji can use far more tokens per character, so **a character cap is not a token guarantee.** A production gateway counts tokens with the real tokenizer and reserves room for output. If the scaffold ever overflows, llama.cpp returns an error, and that becomes a visible failed turn. It never silently trims history.

## Going to the cloud is a data decision, not a failover {#cloud-consent}

The last routing choice is the sensitive one: sending a conversation to OpenAI or Anthropic instead of the local farm. The whole reason a company runs a local farm is usually data handling, so the scaffold treats "use the cloud" as a **policy decision about data leaving the building**, not as a load-balancing trick.

For a non-local provider, **every** gate must pass:

```text
 provider ∈ {local, openai, anthropic}?            no → 400 invalid_provider
   │ openai / anthropic
   ▼
 operator: ALLOW_CLOUD == "true"  AND  request: cloudConsent === true ?
                                                    no → 403 cloud_not_approved
   ▼
 server has <PROVIDER>_API_KEY and <PROVIDER>_MODEL? no → 503 openai_unconfigured / anthropic_unconfigured
   ▼
 send completed history + new message to that provider; record provider on the turn
```

`ALLOW_CLOUD` defaults to `false` in the Compose file. The consent checkbox in the UI says in plain words that the completed history plus the new message will leave the farm, and it **unticks itself after every send**, so consent is per request. And critically: **a local outage never flips the provider.** If A and B are both down you get `503 no_healthy_local_model`, not a silent trip to someone else's datacenter. Automatic cloud fallback would turn an ops incident into a data incident.

The adapters are small and text-only:

| Provider | Endpoint | Auth | Output cap field | Text lives in | Stream ends with |
|---|---|---|---|---|---|
| local llama.cpp | `{INFERENCE_x_URL}/v1/chat/completions` | `Authorization: Bearer $INFERENCE_API_KEY` | `max_tokens: 512` | `choices[0].delta.content` | `data: [DONE]` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `Authorization: Bearer $OPENAI_API_KEY` | `max_completion_tokens: 512` | `choices[0].delta.content` | `data: [DONE]` |
| Anthropic | `https://api.anthropic.com/v1/messages` | `x-api-key` + `anthropic-version: 2023-06-01` | `max_tokens: 512` | `content_block_delta` → `delta.text_delta` | `message_stop` |

Everything upstream is decided **server-side**: the browser picks a provider *name*, never a URL, and API keys never reach it. Accepting an upstream URL from the browser would hand anyone a way to make your server call arbitrary hosts with your keys (server-side request forgery). Model IDs come from environment variables, not hard-coded "latest" names, because those change. Tools, images, structured output, usage accounting, and reasoning events are deliberately outside these adapters.

Two more honest caveats. First, the providers don't produce equivalent answers or share context limits. The gateway only makes their *failures* land in the same durable states. Second, when the browser disconnects, the `AbortController` cancels the upstream `fetch`, but that's a request, not proof: the remote side may already have computed (and billed) more tokens. Keep prompts out of logs by default. The error handler logs only a status code and path.

> 🎬 **Animation — the cloud gate chain:** a message card moves left to right through four turnstiles: "valid provider?", "ALLOW_CLOUD=true?", "cloudConsent ticked?", "key + model configured?". Run 1: consent unticked; the card stops at turnstile 3 with a red "403" and never reaches the cloud icon. Run 2: all green; the card, now showing "history + new message", passes through a boundary labelled "leaves the farm" to a cloud icon, and the turn row records "provider: anthropic". Run 3: both local GPU boxes go red; the card is handed "503 no_healthy_local_model" and a crossed-out dotted arrow to the cloud reads "never automatic".

## Break it on purpose: a test plan and the honest limits {#break-it-on-purpose}

The happy path proves almost nothing. Consistency designs are proven by **adversarial sequences**. Once the farm is running (part 2 walks through `docker compose ... up` and the curl-based login and turn commands), work through this list. If you'd rather not run anything yet, `node --check server.js` and `node test-stream.mjs` in `examples/ai-farm/api/` syntax-check the server and run the parser tests without Docker or a GPU.

| # | Do this | Expect |
|---|---|---|
| 1 | Send the exact same `turn.json` twice | second call returns `replay: true`; still one user turn |
| 2 | Same `requestId`, edit the text | `409 idempotency_key_reused` |
| 3 | Two tabs at the same revision, different messages | one accepted, one `409 revision_conflict` |
| 4 | Click Cancel mid-stream | saved text kept, labelled `canceled / client_disconnect` |
| 5 | `docker compose stop api1` mid-stream, wait > 150 s, sync via api2 | `partial` (or `error` if empty) with `lease_expired`; no auto re-run |
| 6 | Stop inference B mid-stream | `partial / upstream_failed`; the next turn lands on A |
| 7 | Stop both inference servers, send | `503 no_healthy_local_model`; nothing sent to the cloud |
| 8 | GET a conversation without the cookie | `401 login_required` |
| 9 | Paste > 6,000 characters of history over several turns | `413 history_limit_start_new_conversation` |

A caveat on test 5: an interrupted request may still be sitting in an old replica's handler, and (as part 2 notes) Nginx resolves upstream names at startup, so recreating an API container can require recreating the gateway too.

The schema loads only when the Postgres volume is first created. Editing `schema.sql` later doesn't migrate an existing database, and part 2 covers migrations and backups.

### What the scaffold deliberately doesn't do

- One shared Admin account, so no multi-tenant isolation. Sessions last 8 hours, and the login throttle is 20 attempts per minute *per process*.
- Replica-local admission (4 per process), with no global or per-tenant fairness.
- A fixed, non-renewable 150 s lease and lazy recovery, with no sweeper or durable job worker.
- Persistence on every fragment: simple invariant, heavy writes.
- No stored `finish_reason`/`stop_reason`, so `complete` can hide a length cut-off.
- Character limits, not token counting.
- No TLS configuration (plain HTTP on a loopback-bound lab port), no schema migration system, no automatic drain.

None of that is an accident. Each item is a place where you'd say in an interview, "here's the simple version, here's exactly where it breaks, and here's what I'd add."

# Interview

## Question

You run two stateless API replicas behind a load balancer, calling two identical LLM inference servers. A user watched half an answer stream in, their connection dropped, and their client retried, landing on the *other* API replica. Walk me through exactly what happens to the conversation's state, and explain why neither sticky sessions nor putting Kafka in the middle would solve this by themselves.

## Answer

I'd start with where truth lives. Postgres is the only authority. The API replicas are clerks that enforce rules, and the inference servers are stateless compute with a disposable KV cache. Each conversation row has a `revision` counter, and each turn has a client-generated request ID, a fingerprint of its content, a status, and a lease deadline.

When the original request arrived, the first replica took a row lock on the conversation (`SELECT … FOR UPDATE`), recovered any expired running turn, checked the request ID was new and the expected revision matched, confirmed no other turn was running, then inserted the user text as a `running` turn with `lease_until = now() + 150 s` and bumped the revision. It committed *before* calling inference, so no DB transaction or connection is held across generation. While streaming, it saved the accumulated answer with a conditional `UPDATE … WHERE status='running' AND lease_until > now()` *before* sending each delta. So everything the user saw is already durable.

The retry carries the same request ID and the same original expected revision. The second replica locks the conversation and checks the request ID **before** the revision. That order matters: the revision has already advanced because of this very turn, so checking it first would reject a legitimate retry. It finds the ID with a matching fingerprint and returns a snapshot. No second inference starts. The client renders the saved prefix and polls. Now there are two branches. If the first replica is alive, it notices the disconnect, marks the turn `canceled` (or finishes it `complete` or `partial`), and bumps the revision. If it crashed, the next read after the lease expires marks the turn `partial` (or `error` if no text was saved) with `lease_expired`. If the first replica was only paused and wakes up, its conditional writes match zero rows. That's the fence, so it can't overwrite the recovered record.

Sticky sessions only improve locality. They'd pin the user to a replica that just died, and they give you no durable record, ownership check, or ordering. Kafka is a transport: you'd still need the reservation, ownership, per-conversation ordering, leases, and terminal-state rules, and event delivery doesn't give exactly-once model execution anyway. I'd also be explicit that this design doesn't claim exactly-once execution either. It guarantees one request ID never *starts* two generations, it never splices a retry onto visible text, and it reports honestly what happened.

## Follow-ups

- The scaffold rewrites the whole answer on every fragment. How would you batch persistence, and how would you state the maximum amount of displayed text a crash could lose?
- How would a renewable lease with an owner epoch let generations run longer than 150 s while still fencing an old worker?
- Where would you enforce a global inference-slot budget and per-tenant fairness, given that admission today is per-replica?
- How would you design an explicit "regenerate" action that keeps both the failed attempt and the new answer in history?
- When would you move this from synchronous HTTP/SSE to a durable job queue, and what extra machinery (outbox, leases, idempotency, ordering) comes with it?

# Pitfalls

- **Trusting a browser-supplied owner ID, or hashing a bearer token into a routing key or log.** Identity must come from a server-side session lookup, and secret-derived hashes are still stable fingerprints of the secret.
- **Holding a Postgres transaction open during inference.** It pins a pooled connection and the conversation lock for seconds or minutes, so unrelated reads stall and the pool drains.
- **Minting a new request ID whenever `fetch` throws.** The server can't tell a retry from a new message, and you get duplicate turns and duplicate GPU spend.
- **Checking the revision before the request ID.** Legitimate retries get rejected, because the turn being retried already advanced the revision.
- **Calling a stream `complete` because the socket hit EOF.** Without the provider's terminator (`[DONE]` / `message_stop`), it's an interruption: `partial` if text was saved, otherwise `error`.
- **Silently retrying on another node after tokens were shown.** Sampling is random, so you splice two unrelated continuations and may double-bill.
- **Assuming consistent node affinity guarantees a prefix-cache hit.** Eviction, slot assignment, restarts, and template changes all defeat it, so always rebuild the prompt from the database.
- **Treating a per-replica counter as a global GPU capacity controller.** Two replicas admit 8 against 4 GPU slots, and uneven routing rejects on one while the other idles.
- **Falling back to a cloud provider automatically during an outage.** That exports history without consent and turns an availability problem into a data-handling incident.
- **Equating character limits with token limits.** Token counts vary with language, code, and template, so only the real tokenizer tells you whether a prompt fits.

# Checklist

- Name the four identities (session, conversation, request ID, inference node) and say what each is allowed to decide.
- Explain why both chat and snapshot routes query by conversation ID **and** authenticated owner.
- Trace the revision through a happy turn (+1 at reservation, +1 at completion) and through a two-tab conflict.
- Commit the user's text as a `running` turn before contacting inference, and release the DB connection before the external call.
- Explain why the idempotency check precedes the revision check, using a concrete retry.
- Describe persist-then-display, its write-amplification cost, and how batching would change the loss window.
- Map a failed stream to `complete`, `partial`, `error`, or `canceled`, and say why only `complete` turns go into future prompts.
- Walk through a crash timeline with a 150 s lease and explain how conditional writes fence a paused process.
- Compute rendezvous placement for a conversation and say which conversations move when a node dies.
- Explain why replica-local admission (4 per process, 8 total) isn't a cluster-wide limit against 4 GPU slots.
- List every gate a cloud request must pass, and why an outage never opens them.

# Sources

- [PostgreSQL 18 docs: Explicit Locking](https://www.postgresql.org/docs/current/explicit-locking.html) — `SELECT FOR UPDATE` blocks competing updates and lockers of the row until transaction end; row locks are released at transaction end.
- [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) — Blank-line event framing, `field: value` lines, consecutive `data:` lines joined with a newline; EventSource is a one-way GET-style connection.
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — Public `/health` (503 while loading, 200 when ready), OpenAI-compatible streaming `/v1/chat/completions`, `--api-key`/`LLAMA_API_KEY`, `--parallel` slots with per-slot KV cache, prompt caching, and the separate built-in web UI.
- [Claude docs: Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming) — Event sequence ending in `message_stop`, `content_block_delta` with `text_delta`, in-stream `error` events such as `overloaded_error`, and the `anthropic-version: 2023-06-01` header in examples.
- [OpenAI API reference: Chat Completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) — `chat.completion.chunk` objects carrying `choices[].delta.content`, and `finish_reason` values including `length`.
- [OpenAI Cookbook: How to stream completions](https://developers.openai.com/cookbook/examples/how_to_stream_completions) — Chat Completions streams as data-only server-sent events; read text from the `delta` field.
- [nginx: ngx_http_upstream_module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html) — `least_conn` picks the server with the fewest active connections; plain `hash` may remap most keys when servers change, while `consistent` (ketama) remaps only a few.

# Flashcards

## farm-truth

**Q:** In the farm, which component owns conversation truth, and what are the inference servers allowed to keep?

Postgres owns it: sessions, conversations, turns, revisions, and statuses. Express is the only writer and enforces the rules. The inference servers are stateless compute. They may keep only ephemeral caches (the KV/prefix cache), and losing those costs speed, never correctness, because every prompt is rebuilt from Postgres.

## farm-auth

**Q:** Why does the scaffold use an opaque random session token stored (hashed) in Postgres instead of trusting an identity the browser sends?

An opaque token carries no claims, so the browser can't choose or forge an identity. It only means something when the server looks it up. Storing it in Postgres lets **either** API replica authenticate any request without sticky sessions, and storing only its SHA-256 hash means a database leak doesn't hand out live cookies. Ownership of a specific conversation is still checked separately (`WHERE id AND owner_id`).

## farm-lock

**Q:** What happens inside the reservation transaction, and what is deliberately kept outside it?

Inside, under a `FOR UPDATE` lock on the conversation row: stale-lease recovery, the request-ID/fingerprint check, the expected-revision check, the "no other running turn" check, the prompt-size check, inserting the `running` turn with its lease, and bumping the revision. Then it commits. Inference stays outside, because a long, unpredictable external call must not hold a pooled DB connection or the conversation lock.

## farm-idempotency-order

**Q:** Why must the server check the request ID before comparing the expected revision?

A legitimate retry carries the revision the client saw *before* the original attempt was reserved, and that reservation already advanced the revision. Checking the revision first would reject the retry as a conflict. Checking the request ID first recognizes it and returns the existing snapshot without starting another generation.

## farm-fingerprint

**Q:** What happens when a client reuses a request ID with different text, and why?

The server stored a fingerprint (a hash of text, provider, expected revision, and consent) with the original turn. A mismatch returns `409 idempotency_key_reused`. A request ID names one immutable attempt, so reusing it with different content is a client bug, not a way to edit a message.

## farm-durable-delta

**Q:** What does "persist before display" guarantee, and what does it cost?

Every delta is written to Postgres (a conditional update of the accumulated answer) before it's sent to the browser, so anything the user saw survives a crash and shows up on reload through any replica. The cost is write amplification: rewriting the whole answer per fragment. For example, 400 fragments of 5 characters write about 401,000 characters for a 2,000-character answer. Batching reduces that but opens a loss window you must document.

## farm-terminal

**Q:** Why isn't reaching EOF enough to mark an answer `complete`?

A connection can close without the provider's terminal signal (`data: [DONE]` for OpenAI-style streams, `message_stop` for Claude), for example because of a crash, timeout, or network cut. The scaffold treats that as an interruption: `partial` if text was saved, `error` if not (or `canceled` if the browser left). Only `complete` turns are fed into future prompts. Even `complete` only means the stream ended properly, not that the answer is right or untruncated.

## farm-lease

**Q:** How does the 150-second lease recover a turn whose API process crashed?

At reservation the turn gets `lease_until = now() + 150 s` by the database clock. Any later operation that locks the conversation (GET, a new chat, finalization) first flips expired `running` turns to `partial` (text saved) or `error` (none), with `lease_expired`, and bumps the revision. Recovery is lazy, so there's no background sweeper. It's 150 s rather than 120 s to leave margin for a normal timeout to finalize.

## farm-fencing

**Q:** A paused API process wakes up after its turn's lease was recovered. What stops it from overwriting the record?

Every text and terminal write is conditional: `WHERE status = 'running' AND lease_until > now()`. After recovery the turn is no longer `running` and the lease is in the past, so the zombie's update matches zero rows, and it aborts with `lease_lost`. That conditional write is the fence.

## farm-replay

**Q:** After a dropped stream, does the scaffold replay the SSE events the client missed?

No. A retry with the same request ID returns an authoritative **snapshot**: saved user text, saved assistant text, status, and revision. The client polls until the status is terminal. Replaying exact events would require a durable event log with sequence numbers and retention, which an HTTP stream doesn't provide.

## farm-rendezvous

**Q:** How does rendezvous hashing pick an inference node, and what happens when a node dies?

Each healthy node gets a score of SHA-256(conversation ID ‖ stable node name), and the highest score wins, with the next ones as fallbacks. It needs no shared table, so every replica computes the same answer. When a node dies, only conversations that preferred it move to their runner-up. Others don't move. It improves the odds of a prefix-cache hit but doesn't guarantee one, balance load, or authenticate anyone.

## farm-retry

**Q:** Why doesn't the gateway silently retry on the other inference node when a stream fails?

Before any text, an error may hide a request that actually ran (GPU time or money already spent). After text is visible, a second execution is a different random sample, so appending it would splice two unrelated continuations into one answer. The scaffold records `partial` or `error` and lets the user start an explicit new turn. Nginx has `proxy_next_upstream off` for the same reason.

## farm-admission

**Q:** Each API replica admits at most 4 chats. Why isn't that a farm-wide limit?

It's a process-local counter. Two replicas admit up to 8 combined, against only 4 GPU slots (2 per server), so up to 4 can wait inside llama.cpp. Because `least_conn` balances connections rather than work, one replica can reject with 429 while the other has room. A global, tenant-fair limit needs shared state.

## farm-cloud

**Q:** What has to be true for a turn to go to OpenAI or Anthropic, and what happens during a local outage?

The provider name must be valid, the operator must set `ALLOW_CLOUD=true`, the request must carry `cloudConsent: true` (the UI unticks it after each send), and the server must have that provider's key and model configured. A local outage just returns `503 no_healthy_local_model`. It never switches providers automatically, because that would export conversation history without consent.

## farm-context

**Q:** The scaffold caps history at 6,000 characters. Why doesn't that prove the prompt fits the model's context?

Context is measured in tokens, and characters per token vary a lot with language, code, and emoji, plus chat-template overhead and reserved output tokens. At about 4 characters per token for English, 6,000 characters is roughly 1,500 tokens, which fits in a 4,096-token slot, but other text can blow past that. Exact budgeting needs the model's tokenizer, and an overflow must show up as a visible failed turn, not silent truncation.
