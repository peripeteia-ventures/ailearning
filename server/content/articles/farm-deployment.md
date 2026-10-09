---
{
  "slug": "farm-deployment",
  "title": "AI farm, part 2: deploying it with Docker, step by step",
  "category": "ai-farm",
  "summary": "A step-by-step Linux lab walkthrough that brings up two llama.cpp GPU replicas and one authenticated Express/Postgres/Nginx control project with Docker Compose, with a verification gate at every layer and a practised path back when something breaks.",
  "difficulty": "Systems",
  "minutes": 35,
  "prerequisites": ["farm-blueprint", "serving-kv-cache", "llm-security"],
  "learningObjectives": [
    "Bring up two independent llama.cpp inference replicas and one control project (Postgres, two Express replicas, Nginx) from the example Compose files",
    "Verify each layer in order (network, GPU runtime, image digest, model hash, readiness, authenticated generation) before testing chat",
    "Calculate the per-slot token budget and the KV cache size for the example configuration, and explain why readiness is not free capacity",
    "Explain which Nginx settings keep Server-Sent Events streaming and why automatic retries are turned off",
    "Back up, restore-test and roll back the deployment without destroying conversation history"
  ]
}
---

# Sections

## What we're building, and what "done" actually means {#what-done-means}

In **AI farm, part 1: designing a two-server Qwen deployment** we drew the design on a whiteboard: two GPU machines that generate text, and one central web application that owns logins, chat history and every decision about who may do what. This article is the hands-on half. We take that drawing and turn it into containers, one command at a time, and at each step we check that the layer we just built really works before we stack the next one on top.

First, a boundary that matters. This is an **educational Linux lab**. It's separate from the Windows learning app you're reading right now. The files live in `examples/ai-farm/` (downloadable as `/examples/ai-farm.zip`). Nobody has run these commands on real hardware while writing this guide: no container was started, no model was downloaded and nothing was benchmarked. Every command below is an instruction for *you*, the operator. None of it is evidence that your GPU, driver or network will cooperate.

Here's the cast. **Docker** runs programs in *containers*, isolated boxes that each hold one program plus its dependencies. **Docker Compose** starts a group of related containers from one YAML file, which Compose calls a *project*. A **GGUF** is the single-file model format that **llama.cpp**, a lightweight open-source inference server, loads and serves over HTTP. **Express** is a small Node.js web framework, **Postgres** is the database, and **Nginx** is the *reverse proxy*, the front door that receives browser traffic and forwards it to the right back-end.

```text
   your workstation                     Host A  (10.20.0.11)                    Host B  (10.20.0.12)
 ┌──────────────────┐   SSH tunnel   ┌───────────────────────────────────┐    ┌────────────────────┐
 │ browser          │═══════════════►│ control project                   │    │ inference project  │
 │ localhost:8088   │                │  gateway (Nginx) 127.0.0.1:8088   │    │  llama  :8080      │
 └──────────────────┘                │     │ least_conn                  │    │  (Qwen GGUF, GPU)  │
                                     │     ├──► api1 :3000 ─┐            │    └─────────▲──────────┘
                                     │     └──► api2 :3000 ─┤            │              │
                                     │              db :5432◄┘ (Postgres)│              │
                                     │                      │            │              │
                                     │ inference project    ├────────────┼──────────────┘ LAN :8080
                                     │  llama :8080 ◄───────┘            │
                                     └───────────────────────────────────┘
```

Host A does double duty: it runs one GPU inference replica *and* the `control` Compose project (Postgres, `api1`, `api2`, `gateway`). Host B runs only a second, identical inference replica. The two GPUs are **independent replicas**, meaning two full copies of the same model. They are not two halves of one model split across the network. That one fact sets the failure story. Lose B and you lose half your generation capacity. Lose A and you lose the database and the front door, so everything stops.

An analogy that holds up well: a restaurant with two kitchens but one front desk and one order book. A second kitchen means more meals cook at once. But if the front desk closes, nobody gets seated, however many kitchens are open. Two GPUs give you capacity. They don't make the whole system highly available.

So what counts as "done"? Not "a hello-world came back". The acceptance test is an end-to-end conversation whose identity, authorization, *revision* (a version counter on each conversation, used later to catch conflicting edits) and history all survive when requests land on a different Express replica. Performance is a separate experiment that comes later. Run correctness first with short synthetic text and one user. Only then should you raise concurrency or context length.

> 🎬 **Animation — two kitchens, one front desk:** draw the topology above as a restaurant. Step 1: a browser (customer) walks through an SSH tunnel to the front desk (Nginx on A). Step 2: the desk hands the order to one of two waiters (api1/api2), who writes it into the order book (Postgres). Step 3: the waiter sends it to kitchen A or kitchen B (llama :8080). Step 4: grey out kitchen B; orders still flow, just slower. Step 5: grey out the front desk on A; every path goes dark, even though kitchen B is still lit. Caption: "Replicas add capacity; the control host is still a single point of failure."

### Which files are runnable and which are fragments

| File | Owner | What it is |
|---|---|---|
| `inference/compose.yml` | this article | Runnable Compose project, copied unchanged to both GPU hosts |
| `control/compose.yml` | this article | Runnable Compose project for host A |
| `control/nginx.conf` | this article | Complete Nginx config, mounted read-only |
| `.env.example` | this article | Template. It is **deliberately invalid** until you replace the placeholders |
| `README.md` | this article | The full ordered command list |
| `api/*` (Dockerfile, server.js, schema.sql, public/index.html) | part 3 | The tiny Express app that Compose builds |

Code blocks in this article marked "from README" are copied from the README exactly. Blocks marked "excerpt" show part of a file so we can discuss it; the file in the repo is the source of truth. The Express app's internals (routing, leases, revisions) are covered in **AI farm, part 3: routing and keeping conversations consistent**. Here we only deploy it.

## Addresses first: getting two machines to see each other {#network-first}

Before a single container starts, both machines need stable, known addresses. The example uses `10.20.0.11` for A and `10.20.0.12` for B. Those are placeholders. Get real ones reserved by your network administrator or DHCP service. Compose doesn't hand out host addresses. It only knows about the private networks it creates *inside* one host.

This is the first place people get tripped up. Inside the control project, `api1` can reach Postgres at the hostname `db`, because Compose runs a little DNS service for each project's network. But the inference project on B is a completely separate project on a separate machine. The service name `llama` means nothing from A. So the API reaches inference through plain LAN URLs (`INFERENCE_A_URL=http://10.20.0.11:8080`, `INFERENCE_B_URL=http://10.20.0.12:8080`), and it reaches A's own replica the same way, through A's LAN address, not through a service name.

```text
 Three kinds of "address" in this lab — don't mix them up

  name                    works from…                        example
  ─────────────────────   ─────────────────────────────────  ─────────────────────────
  localhost:8088          your workstation, via SSH tunnel   browser → gateway
  db / api1 / api2        inside the control project only    api1 → db:5432
  10.20.0.11 / .12:8080   anywhere on the private LAN        api → inference A / B
```

Check the plumbing from the host before you blame any application code (from README):

```bash
ip -brief address          # A must show 10.20.0.11; B must show 10.20.0.12
ip route get 10.20.0.12    # on A (and the reverse on B)
```

Next, who may connect. Each inference container publishes port 8080 **only on its host's private address** (`INFERENCE_BIND_IP`). At the network firewall, allow TCP 8080 only from the control host. Here's a subtle Linux trap: traffic to a Docker-published port is *forwarded* into the container, and it doesn't necessarily pass through the host firewall's ordinary INPUT rules. So "my host firewall drops everything by default" doesn't prove the port is closed. Test from a machine that *shouldn't* get in, and confirm it's refused.

The inference API key (`INFERENCE_API_KEY`, a shared secret the API sends with every request to llama.cpp) is *defense in depth*, an extra lock on top of the network rules. It doesn't encrypt anything. The key and the prompts travel over plain HTTP here. So run this lab on an isolated, trusted network segment, or add TLS or mTLS between the machines before prompts cross a shared network.

The browser side is different on purpose. The gateway publishes on `127.0.0.1:8088` by default, which is A's *loopback* interface, reachable only from A itself. To reach it from your laptop, you open an **SSH local forward**, a tunnel that carries a port on your workstation through your SSH login to a port on A (from README):

```bash
ssh -N -L 8088:127.0.0.1:8088 user@10.20.0.11
# keep this running, then open http://localhost:8088
```

`APP_ORIGIN` (default `http://localhost:8088`) must match what the browser's address bar shows (scheme, host and port), because the API checks the `Origin` header on state-changing requests. If you ever expose a real corporate hostname, three things change together: TLS termination at the edge, the correct `APP_ORIGIN`, and `COOKIE_SECURE=true`. Flipping `GATEWAY_BIND_IP` to `0.0.0.0` on its own isn't a production release. It's plain HTTP carrying session cookies across the LAN. The TLS setup itself is outside this lab. The threat model behind these choices is in **LLM security: treating model output as untrusted**.

> 🎬 **Animation — three kinds of address:** three horizontal lanes. Lane 1 (workstation): a packet leaves `localhost:8088`, enters an SSH tube, and pops out at `127.0.0.1:8088` on A. Lane 2 (inside the control project): `api1` looks up `db`, a small Compose DNS bubble answers with an internal IP, and the packet arrives at Postgres. Lane 3 (LAN): `api1` sends to `10.20.0.12:8080`; a firewall gate lets it through because the source is A. Then a laptop tries the same port and the gate slams shut. Final frame: `api1` tries the name `llama` on B and gets a red "name not found".

## Making the GPU visible inside Docker, and pinning the image {#gpu-runtime}

With the network sorted, the next layer is the GPU. Treat this as its own gate. If the GPU isn't visible inside a container, no amount of fiddling with the model will help.

On each GPU host, install Docker Engine and the Compose v2 plugin from Docker's instructions for your distribution, a compatible NVIDIA driver, and the **NVIDIA Container Toolkit**, the component that lets containers use the host's GPU. The lab assumes *rootful* Docker. Rootless Docker needs a different toolkit configuration (a per-user config path and a cgroups setting, per NVIDIA's guide). Then (from README):

```bash
nvidia-smi
docker version
docker compose version
ip -brief address
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`nvidia-smi` is NVIDIA's status tool. If it fails on the host, stop there, because containers can't do better than the host. `nvidia-ctk runtime configure` edits Docker's daemon configuration so Docker knows about the NVIDIA runtime. Restarting Docker interrupts every running container on that machine, so do it in a maintenance window.

Now the image. llama.cpp publishes prebuilt server images, and `ghcr.io/ggml-org/llama.cpp:server-cuda` is the CUDA build. But `:server-cuda` is a **tag**, and tags move. Think of a tag as a street address. The building at that address can be renovated overnight and the address stays the same. A **digest** (`@sha256:…`) is a fingerprint of the exact bytes. It's like a photo of the building on the day you inspected it. You approve bytes, so you pin the digest (from README):

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda --format '{{index .RepoDigests 0}}'
# Put the returned ghcr.io/...@sha256:... in LLAMA_IMAGE in .env.
# Reload your reviewed environment, then inspect the exact image:
docker run --rm "$LLAMA_IMAGE" --help
docker run --rm --gpus all --entrypoint nvidia-smi "$LLAMA_IMAGE"
```

The `.env.example` value `…:server-cuda@sha256:REPLACE_WITH_TESTED_DIGEST` is deliberately broken, so you can't run the lab until you've made this choice yourself. Two checks happen here. The `--help` output must list every flag the Compose file uses: `--no-kv-unified`, `--parallel`, `--ctx-size`, `--n-gpu-layers` and `--no-context-shift`, plus support for the `LLAMA_API_KEY` environment variable. If any of those is missing, reject that image. The second command overrides the image's *entrypoint* (the program a container runs by default) so you can run `nvidia-smi` inside it and check that the GPU really shows up in there.

A digest pins bytes. It doesn't make them trustworthy. Pull from a registry your organization approves, and pin `POSTGRES_IMAGE` and `NGINX_IMAGE` to tested digests too. Then start a **deployment manifest**, a plain record of everything that defines this deployment: image digests, the `--help` output or version, driver version, GPU model, model revision and hash, and configuration with secrets removed. Every later step adds a line to it.

```text
 Gate order on each GPU host (each gate must pass before the next is worth testing)

  [1] nvidia-smi on host ──► [2] docker + compose present ──► [3] nvidia-ctk configured, docker restarted
        │                                                              │
        ▼                                                              ▼
  [6] GPU visible INSIDE image ◄── [5] --help lists every flag ◄── [4] image pinned by digest
        │
        ▼
  next: model artifact
```

> 🎬 **Animation — tag versus digest:** show a signpost labelled `server-cuda` pointing at a box stamped `sha256:aa11…`. Step 1: the operator photographs the box and writes `sha256:aa11…` into `.env`. Step 2: a week later, the signpost swings to a new box `sha256:bb22…` (the upstream rebuilt it). Step 3: a host pulling by tag gets `bb22`, while a host pinned by digest still gets `aa11`. Step 4: both hosts are pinned; highlight that A and B now provably run the same bytes.

## Pinning the model: revision, hash, and a read-only mount {#model-identity}

Now for the model itself. The lab uses `Qwen/Qwen2.5-7B-Instruct-GGUF`, the Q3_K_M file `qwen2.5-7b-instruct-q3_k_m.gguf`. *Quantization* stores the weights with fewer bits (Q3_K_M averages roughly three to four bits per weight instead of sixteen), which shrinks memory at some cost in quality. We'll go through this properly in **Quantization: spending fewer bits per weight**. Q3_K_M was picked because it's a **single file**, which keeps this lab simple. The Qwen repo splits some larger quantizations into multiple parts that must be merged. That isn't a recommendation over Q4 or Q5, and lower precision needs its own quality evaluation on your tasks.

Two identifiers pin the model down, and they answer different questions:

| Identifier | Value in `.env.example` | What it proves |
|---|---|---|
| `MODEL_REVISION` | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` | Which commit of the Hugging Face repo you pulled from (like a git commit) |
| `MODEL_SHA256` | `a96b16179dc6cc9afdf0cf7a96a80c199cbd00b9be207c3465be21cb721cca5e` | That the bytes on disk equal the bytes you approved |

The example's author opened the repo commit, the current file metadata and the LICENSE at that revision. The file page at the pinned revision failed to render, though, so **your** check against the revision's own metadata is still required. Before downloading, open the file page at that revision, compare the SHA256 with `.env`, and read the LICENSE (the repo lists Apache-2.0) and the model card. Save that evidence. One trap: Hugging Face also shows a Xet storage hash, an internal ID from its storage backend. It is not the SHA256 of the file, so don't compare against it.

Then download exactly one file at exactly that revision, and verify it (from README, run after loading your reviewed `.env`):

```bash
# Run after loading your reviewed .env as above. Provision a writable directory.
sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "$MODEL_DIR"
hf download Qwen/Qwen2.5-7B-Instruct-GGUF "$MODEL_FILE" LICENSE README.md \
  --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_DIR/$MODEL_FILE" | sha256sum --check -
```

`hf` is Hugging Face's CLI; install it from their official instructions. If `sha256sum` prints anything but `OK`, stop. Repeat on B, or copy the approved file over securely and check the hash again there. Don't download every quantization. Don't follow `main` implicitly. And never rename a tiny Git LFS pointer file into a `.gguf`: a pointer is a text stub a few hundred bytes long, not weights, and the hash check will catch it.

Back-of-envelope, labelled illustrative: Qwen2.5-7B has about 7.6 billion parameters. At about 3.5 bits per weight that's 7.6 × 10⁹ × 3.5 / 8 ≈ 3.3 × 10⁹ bytes, a bit over 3 GB of weights before the KV cache and runtime buffers. The real number is the file size shown on the page. Use that.

```text
 Model promotion gate (per host)

  review at revision ──► download 1 file ──► sha256sum --check ──► read-only mount ──► serve as corp-qwen
  (hash, LICENSE,        (--revision,          │ mismatch?
   model card, saved)     --local-dir)         └──► STOP. Do not "just try it".
```

Why can't a matching name prove a match? The server gets `--alias corp-qwen`, the name clients put in their `model` field. An alias is a label. You could serve two different files under the same label, and nothing would complain. Two endpoints are interchangeable only when the file hash, the runtime digest, the chat template behaviour and the server flags all match. That's why the manifest records all four.

> 🎬 **Animation — the promotion gate:** a GGUF file moves left to right along a conveyor. Station 1: a clipboard checks the revision `bb5d59e…` and the LICENSE; tick. Station 2: the file is downloaded into `/srv/ai-farm/models`. Station 3: a scanner computes SHA256 and compares it with `a96b1617…`; for the good file, a green OK. Then replay with a 134-byte LFS pointer renamed `.gguf`: the scanner shows a mismatch in red and the conveyor stops. Final station: the file goes behind glass (the read-only mount) with the label `corp-qwen`. Caption: "The alias is a label; the hash is the identity."

## One .env file, two hosts, and how Compose reads it {#configuration}

Both hosts get a copy of the whole `ai-farm` directory, and every command runs from inside it. Configuration lives in one operator-owned file, `.env`, created from the template (from README):

```bash
cp .env.example .env
chmod 600 .env
openssl rand -hex 32
```

Run `openssl rand -hex 32` three times and paste the three *different* results into `POSTGRES_PASSWORD`, `ADMIN_PASSWORD` and `INFERENCE_API_KEY`. Each is 32 random bytes shown as 64 hex characters, so 256 bits. Why hex? The database password gets embedded in `DATABASE_URL=postgres://farm:<password>@db:5432/farm`, and characters like `@`, `/` or `:` would need URL escaping there. Hex has none of those. `chmod 600` means only the owner can read the file.

Then edit per host:

| Variable | Host A (control + inference) | Host B (inference only) |
|---|---|---|
| `LLAMA_IMAGE`, `MODEL_*` | same approved values | **same** approved values |
| `INFERENCE_BIND_IP` | `10.20.0.11` | `10.20.0.12` |
| `INFERENCE_API_KEY` | the shared key | **same** shared key |
| `INFERENCE_A_URL`, `INFERENCE_B_URL` | both set | not needed |
| `POSTGRES_PASSWORD`, `ADMIN_PASSWORD` | set | **remove** |
| cloud keys, `ALLOW_CLOUD` | leave empty / `false` | **remove** |

That's least privilege, meaning each machine gets only the secrets it needs. B only generates text, so a compromised B shouldn't hand over the database password. The lab uses opaque sessions stored in Postgres (random IDs the API looks up, not signed tokens), so no `SESSION_SECRET` is needed. Keep `ALLOW_CLOUD=false` for now. Letting prompts reach an external provider is a data-governance decision as well as a setting.

Here's the part that surprises people. `docker compose --env-file .env` feeds values into **interpolation**: it fills `${VAR}` placeholders inside the YAML. It doesn't automatically copy every variable into every container. Each Compose file explicitly lists what each service receives. Excerpt from `control/compose.yml`:

```yaml
    DATABASE_URL: postgres://farm:${POSTGRES_PASSWORD:?Set hex DB password}@db:5432/farm
    ADMIN_PASSWORD: ${ADMIN_PASSWORD:?Set Admin password}
    APP_ORIGIN: ${APP_ORIGIN:-http://localhost:8088}
```

`${VAR:?message}` means "use VAR if it's set and non-empty, otherwise refuse to start and print the message". `${VAR:-default}` means "otherwise use the default". So a forgotten secret fails loudly at `config` time instead of starting a database with an empty password. To check the interpolation without printing the resolved secrets:

```bash
docker compose --env-file .env -f inference/compose.yml config --quiet   # both hosts
docker compose --env-file .env -f control/compose.yml config --quiet     # host A only
```

Never paste the full, unquiet `config` output into a ticket, because it contains your secrets in the clear. For the `curl` checks later you'll want the same values in your shell. After reading your own `.env` and confirming it holds only plain `NAME=value` lines, load it with `set -a; . ./.env; set +a`. That line *executes* the file as shell code, so never do it with an environment file somebody else sent you.

```text
 How one .env reaches a container

  .env ──(--env-file)──► Compose interpolation: ${POSTGRES_PASSWORD:?…}
                               │
                               ▼
                         compose.yml environment: block (explicit list)
                               │
                 ┌─────────────┼──────────────┐
                 ▼             ▼              ▼
             db gets:      api1/api2 get:   gateway gets:
             POSTGRES_*    DATABASE_URL,    (nothing secret)
                           ADMIN_PASSWORD,
                           INFERENCE_*, ...
```

## The inference Compose file, and the slot budget it sets {#inference-compose}

One `inference/compose.yml` is copied unchanged to both GPU hosts. The only things that differ are the values in each host's `.env`. Here's what each part does:

| Line(s) | What it does | Why |
|---|---|---|
| `image: ${LLAMA_IMAGE:?…}` | Pinned server-cuda digest | Same bytes on both hosts |
| `ports: "${INFERENCE_BIND_IP}:8080:8080"` | Publish only on the private LAN IP | Not on every interface |
| bind mount `MODEL_DIR → /models`, `read_only: true`, `create_host_path: false` | Model directory, read-only | The server can't modify the weights, and a typo'd path fails instead of silently creating an empty directory |
| `LLAMA_API_KEY: ${INFERENCE_API_KEY}` | llama.cpp's API-key environment variable | Generation requires the bearer key |
| `--alias corp-qwen` | Model name clients use | Matches `LOCAL_MODEL=corp-qwen` in the API |
| `--n-gpu-layers 99` | Offload up to 99 layers to the GPU | Qwen2.5-7B has 28 layers, so this means "all of them" |
| `--parallel 2`, `--ctx-size 8192`, `--no-kv-unified` | Two slots, 8192 total context, separate per-slot KV | Explicit capacity, see below |
| `--no-context-shift` | Don't silently drop old tokens when a slot fills | A full context becomes an error, not quiet amnesia |
| `--n-predict 512` | Server-side cap on generated tokens | Backs up the API's own `max_tokens: 512` |
| `deploy.resources.reservations.devices` (`driver: nvidia`, `count: 1`, `capabilities: [gpu]`) | Ask Docker for one GPU | Compose's docs say `capabilities` must be set |

A GPU reservation hands an existing device to the container. It doesn't install drivers, and it doesn't guarantee there's enough memory.

### Slots, and splitting the whiteboard

A **slot** is one sequence the server can generate for at the same moment. Each slot needs its own **KV cache**, the stored attention keys and values for every token processed so far, so that the next token doesn't recompute the whole prompt. We'll go through this in detail in **What happens at inference: prefill, decode, and the KV cache**. Picture the total context as a whiteboard of fixed size, and the slot count as the number of people who each get a section of it.

With `--no-kv-unified`, each slot gets its own separate share: 8192 ÷ 2 = **4096 tokens per slot**. The alternative, a unified KV buffer, is one shared pool that all slots draw from, and newer llama.cpp builds also offer a per-slot limit option. Those are different configurations, and llama.cpp's defaults have changed across versions. That's exactly why the file spells every choice out instead of trusting defaults. Even so, read the startup log on your pinned image and confirm the per-slot context it actually reports.

```text
 --ctx-size 8192, --parallel 2, --no-kv-unified

  ┌──────────── slot 0: 4096 tokens ─────────────┬──────────── slot 1: 4096 tokens ─────────────┐
  │ template+system │ conversation   │ output ≤512 │ template+system │ conversation  │ output ≤512 │
  │   ~256 (illus.) │    3328        │             │                 │               │             │
  └─────────────────┴────────────────┴─────────────┴─────────────────┴───────────────┴─────────────┘
```

Inside a slot, the prompt and the answer compete for the same space:

```formula
content budget = slot context − reserved output − rendered template/system overhead
               = 4096 − 512 − 256 = 3328 tokens
```

*Slot context* is the per-slot share (4096). *Reserved output* is the most the model may generate (512, set by `--n-predict` and the API). *Template overhead* is what the chat template adds: role markers, special tokens and any system text. The 256 here is an **illustrative assumption**. Tokenize your real rendered prompt to get the true number.

Does the API's character cap fit? The API allows at most 6000 characters of selected history per prompt. English often runs around four characters per token, which gives 6000 ÷ 4 ≈ 1500 tokens, comfortably under 3328. But code, other languages and odd Unicode can use far more tokens per character, so a character cap is only a crude guard, not a guarantee. If the prompt doesn't fit, llama.cpp rejects the request and the API has to show that as an error.

### How much GPU memory do two slots cost?

A worked estimate, assuming the default 16-bit (2-byte) KV cache. From Qwen2.5-7B's `config.json`: L = 28 layers, h_kv = 4 key/value heads (it uses grouped-query attention, covered in **Making attention cheaper: GQA, FlashAttention, and long context**), d_head = 3584 / 28 = 128.

```formula
KV bytes per token = 2 (K and V) × L × h_kv × d_head × bytes
                   = 2 × 28 × 4 × 128 × 2 = 57,344 bytes ≈ 56 KiB
KV for all slots   = 57,344 × 8192 tokens = 469,762,048 bytes = 448 MiB
```

So a full 8192-token budget costs roughly 0.45 GiB of KV cache on top of the ~3 GB of weights and the runtime's compute buffers. These are estimates, not measurements. If the load fails, the log tells you which allocation failed. A weights failure means the model doesn't fit. A context-buffer failure can be fixed by lowering `--ctx-size` or `--parallel`. Neither fix tells you anything about answer quality.

One more thing to keep straight: **two slots do not mean two users at full single-user speed**. Slots let the server interleave work on one GPU. Each extra slot can raise total throughput while slowing each individual stream, and the balance depends on your hardware. Scheduling is covered in **Serving many users: batching, scheduling, and speculative decoding**. Change one capacity variable at a time and load-test on the target machine.

> 🎬 **Animation — splitting the whiteboard:** a GPU memory bar. Step 1: fill ~3.3 GB labelled "weights (Q3_K_M, illustrative)". Step 2: add a 448 MiB block labelled "KV 8192 tokens × 56 KiB", then split it into two 224 MiB halves labelled slot 0 and slot 1. Step 3: zoom into slot 0 (4096 tokens) and colour it as 256 template + 3328 conversation + 512 output. Step 4: a conversation grows past 3328 and the bar turns red with "context exceeded → explicit error (no context shift)". Step 5: flip to `--kv-unified` and show the two halves merging into one shared pool, captioned "different config, check the logs".

### Starting the replicas and testing them directly

On each GPU host (from README):

```bash
docker compose --env-file .env -f inference/compose.yml config --quiet
docker compose --env-file .env -f inference/compose.yml up -d
docker compose --env-file .env -f inference/compose.yml logs --tail=100 llama
```

In the logs, look for three things: how many layers actually landed on the GPU, the per-slot context, and that the model loaded without error. Then, **from host A** with its reviewed environment loaded (from README):

```bash
curl --fail-with-body "$INFERENCE_A_URL/health"
curl --fail-with-body "$INFERENCE_B_URL/health"
curl --fail-with-body -N "$INFERENCE_A_URL/v1/chat/completions" \
  -H "Authorization: Bearer $INFERENCE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"corp-qwen","messages":[{"role":"user","content":"Reply with hello."}],"max_tokens":32,"stream":true}'
```

Repeat the streaming call against B. `-N` turns off curl's own output buffering so you can watch chunks arrive. Probing from A matters because it tests the exact network path the API will use, and the upstream image may not ship `curl` anyway. Then run the negative test: the same request **without** the `Authorization` header must be refused. Don't save any output that contains the key.

Evidence gets stronger in steps. A running process is weak evidence. `/health` returning 200 is stronger: llama.cpp answers 503 while the model loads and 200 once it's ready. An authenticated generation that streams is stronger still. `/health` is public (it doesn't need the key), which is one more reason to keep port 8080 off the wider network. And "ready" means the model is loaded. It doesn't mean a slot is free. A restaurant can be open without having a free table. If A passes and B fails, diff the manifest (hash, digest, flags, driver, firewall) before blaming randomness. Also don't expect identical wording from the two replicas; test structure, not exact sentences.

This direct call is a diagnostic gate only. Real users go through Express, because the llama.cpp built-in web UI bypasses central authentication and history.

## The control project: Postgres, two APIs and a gateway {#control-plane}

Once both GPU hosts answer correctly, we move to the control project on A. It has four services:

| Service | Image / build | Published? | Health check |
|---|---|---|---|
| `db` | `${POSTGRES_IMAGE:-postgres:18}` | no | `pg_isready -U farm -d farm` every 5 s |
| `api1`, `api2` | built from `../api` (`node:24-bookworm-slim`) | no, internal port 3000 | Node `fetch` of `/healthz`, which runs `SELECT 1` |
| `gateway` | `${NGINX_IMAGE:-nginx:stable-alpine}` | `${GATEWAY_BIND_IP:-127.0.0.1}:${GATEWAY_PORT:-8088}` → 80 | none (waits for healthy APIs) |

A few design points, each of which answers a question an interviewer might ask.

**`api1` and `api2` are the same definition.** The file defines one YAML anchor, `x-api: &api`, and both services merge it with `<<: *api`. They're stateless: they hold no conversation state in memory that the other replica would need. Everything durable lives in Postgres. That's why the gateway doesn't need sticky sessions (pinning each user to one replica).

**Build context `../api`.** Compose resolves relative paths from the Compose file's folder, so `build: ../api` means `examples/ai-farm/api`. That Dockerfile (owned by part 3) installs `express` and `pg`, runs as the unprivileged `node` user, and listens on `0.0.0.0:3000` inside the container. `public/index.html` is served by Express itself, so it isn't mounted into Nginx.

**The Postgres volume mounts at `/var/lib/postgresql`, not `/var/lib/postgresql/data`.** From Postgres 18, the official image keeps its data in a version-specific subdirectory (`/var/lib/postgresql/18/docker`), and the image's documentation says to mount the parent. Mounting the old path is a classic upgrade trap.

**`schema.sql` is an init script, not a migration tool.** It's mounted read-only at `/docker-entrypoint-initdb.d/001-schema.sql`. The image runs those scripts **only when the data directory is empty**. After the first start, editing `schema.sql` does nothing to the existing database. Changing an existing database needs an explicit, reviewed *migration* (a versioned change script). And if the first initialization failed halfway, investigate before retrying, because on the next start the scripts won't run again. Whatever you do, don't "fix" it by deleting the volume once it holds real history.

**Health gates start-up order, not ongoing failover.** The API containers wait for `db: service_healthy`, and the gateway waits for both APIs to be healthy. That's all `depends_on` does. It orders the first start. It isn't a cluster manager: if `api2` dies later, nothing removes it from Nginx's list.

```text
 Start-up ordering in the control project (depends_on + health checks)

  t=0    db starts ──► pg_isready (every 5s, up to 12 retries)
                              │ healthy
                              ▼
         api1, api2 start ──► fetch /healthz → SELECT 1  (every 15s, start_period 20s)
                              │ both healthy
                              ▼
         gateway starts ──► listens 127.0.0.1:8088 → least_conn → api1:3000 / api2:3000

  after start-up: no one watches this chain. A later api2 crash is NOT auto-removed.
```

Bring it up and check each layer (from README):

```bash
docker compose --env-file .env -f control/compose.yml config --quiet
docker compose --env-file .env -f control/compose.yml up -d --build
docker compose --env-file .env -f control/compose.yml ps
docker compose --env-file .env -f control/compose.yml exec gateway nginx -t
curl --fail-with-body http://localhost:8088/healthz
```

`ps` shows health states, `nginx -t` checks the config syntax inside the running gateway, and the last `curl` goes through Nginx to an API and on to the database. Then open the tunnelled UI, log in as `Admin` with your generated password, and send one short message. Success means the conversation shows up in the **authoritative snapshot**, a fresh read straight from Postgres. What's painted in the browser tab doesn't count.

> 🎬 **Animation — start-up gates:** four boxes in a column (db, api1, api2, gateway), all grey. Step 1: db pulses with a heartbeat labelled `pg_isready` and turns green after a few beats. Step 2: api1 and api2 light up, each sending a small `SELECT 1` arrow to db, and turn green. Step 3: the gateway turns green and a browser arrow enters at `127.0.0.1:8088`. Step 4: later, api2 turns red; the gateway keeps its arrow pointed at api2 and a request bounces with a 502. Caption: "depends_on orders start-up; it doesn't repair failures."

## Keeping streams streaming through Nginx {#streaming-gateway}

Chat answers arrive as **Server-Sent Events (SSE)**: the server keeps an HTTP response open with content type `text/event-stream` and pushes `data: …` lines as they're produced, with a blank line after each event. That's why words appear one by one. For that to work, every hop between the GPU and the browser has to pass chunks along *immediately*, and Nginx sits right in the middle. Here's the whole config, a complete file:

```nginx
events { worker_connections 1024; }
http {
  upstream farm_api {
    least_conn;
    server api1:3000;
    server api2:3000;
    keepalive 16;
  }
  server {
    listen 80;
    client_max_body_size 32k;
    location / {
      proxy_pass http://farm_api;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host $http_host;
      proxy_set_header X-Forwarded-For $remote_addr;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_buffering off;
      proxy_cache off;
      gzip off;
      proxy_connect_timeout 5s;
      proxy_read_timeout 180s;
      proxy_send_timeout 180s;
      proxy_next_upstream off;
    }
  }
}
```

Taking it a few lines at a time:

**`least_conn`** sends each new request to the upstream with the fewest *active connections*. A streaming chat holds its connection open for its whole length, so this spreads load better than plain round-robin. But it counts connections, not tokens, GPU load or queue depth. Which GPU a request uses is a separate decision made *inside* each API. Each API ranks the healthy endpoints by a hash of (conversation ID, server) so a conversation tends to return to the same GPU and reuse its warm cache. Warmth is an optimization, never a correctness requirement, and part 3 covers it.

**`proxy_buffering off`** is the key SSE setting. With buffering on (Nginx's default), Nginx reads the upstream response into buffers and sends it along in bigger pieces. With it off, Nginx passes each chunk to the client as soon as it arrives. `proxy_cache off` and `gzip off` stop anything else from holding bytes back to cache or compress them.

**`proxy_http_version 1.1` + `Connection ""` + `keepalive 16`** reuse TCP connections to the APIs instead of opening a new one per request. Recent Nginx versions default to 1.1 upstream, and older ones used 1.0. Setting it explicitly keeps the config correct on either.

**`Host $http_host` and the `X-Forwarded-*` headers** pass through the browser's host (including the `:8088` port) and the client's address, so the API sees the real origin.

**Timeouts.** `proxy_read_timeout 180s` is the longest gap *between two reads* from the upstream. It is not a cap on the whole response. The API aborts a generation at **120 s**, so the API's explicit error always fires before Nginx's timeout does. `client_max_body_size 32k` sits above the API's 16 KiB JSON limit, so the API's own, clearer error message is the one users see.

**`proxy_next_upstream off`** turns off automatic retries completely. Even by default, Nginx won't resend a `POST` to another upstream once it has already been sent (POST isn't *idempotent*, meaning running it twice isn't safe). Setting `off` also stops retries after connection errors. We give up a bit of transparent availability to get a simple rule: **the proxy never replays a generation**. It's like a card payment that times out. You check your statement before paying again. You don't let the terminal quietly charge you twice. An ambiguous failure is resolved by the client reading the snapshot and deciding.

One operational gotcha: Nginx resolves `api1` and `api2` when it starts. (Nginx does have a dynamic `resolve` option, but this config doesn't use it.) If you recreate an API container it may get a new internal IP, so **recreate the gateway after recreating APIs**.

```text
 One streamed answer, hop by hop (buffering OFF everywhere)

  GPU ─token─► llama.cpp ─SSE chunk─► api1 ─SSE chunk─► Nginx ─SSE chunk─► browser
               (8080)                 (3000)            (buffering off,     (renders
                                        │                read gap ≤180s)     word by word)
                                        └─► Postgres: user turn committed BEFORE inference;
                                            terminal status written at the end
                                            (complete / partial / error / canceled)
```

> 🎬 **Animation — buffered vs unbuffered proxy:** two identical pipelines side by side, llama.cpp → API → Nginx → browser. Tokens `The`, `KV`, `cache`, `stores`, … leave the GPU every 50 ms. Left pipeline (`proxy_buffering on`): tokens pile up in a bucket inside Nginx and the browser sees nothing for a while, then a burst. Right pipeline (`off`): each token passes straight through and the browser text grows word by word. Then show the retry lesson: in the right pipeline, api1 dies after `cache`; the browser shows "partial" and a "reload snapshot" button, and nothing is replayed through api2.

## Breaking it on purpose: replay, conflicts and failures {#failure-tests}

So far we've checked the happy path. Now we check that failures stay safe. The README has a complete scripted session (it needs `curl`, `jq` and `uuidgen`). This excerpt shows its shape: log in with a cookie jar, create a conversation, and build one turn with a fresh `requestId`:

```bash
origin=http://localhost:8088
umask 077
jq -n --arg password "$ADMIN_PASSWORD" '{username:"Admin",password:$password}' > login.json
curl --fail-with-body -c cookies.txt -H "Origin: $origin" -H 'Content-Type: application/json' \
  --data-binary @login.json "$origin/api/login"
rm login.json
# ... create conversation → $cid, $rev; request_id=$(uuidgen); write turn.json ...
curl --fail-with-body -N -b cookies.txt -H "Origin: $origin" -H 'Content-Type: application/json' \
  --data-binary @turn.json "$origin/api/conversations/$cid/chat"
curl --fail-with-body -b cookies.txt "$origin/api/conversations/$cid"
```

`umask 077` makes the new files private to you. The password goes into a file through `jq` rather than onto the command line, where other users could see it in the process list. Each turn carries a `requestId` (a UUID that lets the server recognise a repeat, which is what makes the request *idempotent*) and an `expectedRevision` (the conversation version the client last saw). Now run the experiments:

| Test | What you do | What should happen | What it proves |
|---|---|---|---|
| Replay | Send the *same* `turn.json` again | No second user turn | Idempotency by `requestId` |
| Stale write | New `requestId`, *old* revision | A conflict response; client must reload | No silent overwrite |
| No cookie | GET the conversation without `-b cookies.txt` | Rejected | History only after the ownership check |
| Cloud off | Ask for a cloud provider while `ALLOW_CLOUD=false` | Refused | Private prompts don't leak out |
| GPU dies mid-stream | Stop the selected inference container during a synthetic generation | Turn ends `partial` or `error`; no text from another GPU gets stitched on | Honest terminal status |
| API dies | Stop `api1`, then send requests | Some may fail (no proxy retry); drain it from Nginx and reload, or restore it | Replica failure is visible, not hidden |

Delete `cookies.txt` and `turn.json` afterwards, since both are sensitive. And be honest about what the no-cookie and two-browser tests *don't* show. Every login is the same `Admin` demonstration owner, so this lab can't test multi-tenant isolation. That needs separate identities.

After any ambiguous failure the rule is the same: **GET the snapshot, read the turn's status, then decide**. The API gives each running generation a *lease* (a time-limited claim of "I'm working on this") of 150 s, longer than its 120 s generation timeout. If an API process dies mid-turn, the lease expires, and the *next* API request touching that conversation marks the turn `partial` (if some text was saved) or `error`. There's no background worker doing this. Recovery happens on the next activity. Text that was streamed but never committed can be lost when an API replica dies.

```text
 Timeline: API replica dies mid-generation

  t=0s    user turn committed, status=running, lease_until = t+150s
  t=0–40s tokens stream to the browser
  t=40s   api1 killed ──► stream breaks; browser shows "interrupted"
  t=40–150s snapshot still says running (lease not expired)
  t≥150s  next request on this conversation (via api2) ──► turn marked partial/error, revision+1
          client reloads snapshot, user decides whether to ask again (new requestId)
```

Also note that `complete` in this scaffold means the stream ended cleanly. It doesn't mean the answer is good, and it doesn't mean it wasn't cut off at 512 tokens, because the scaffold doesn't store the upstream `finish_reason`. Production observability should record why each generation stopped. See **Evaluation and observability: knowing whether it actually works**.

The capacity mismatch is also worth pointing out in an interview. Each API admits at most **4** active chat handlers, so 2 replicas × 4 = **8** can be admitted, while the GPUs offer 2 hosts × 2 slots = **4** slots. Up to four admitted requests can therefore be waiting inside llama.cpp. That's intentional evidence that per-replica admission isn't global GPU admission control.

> 🎬 **Animation — replay and conflict:** a conversation card showing `revision 3`. Step 1: turn.json (`requestId R1, expectedRevision 3`) arrives; a user turn appears and the card flips to `revision 4`. Step 2: the identical R1 arrives again; the server finds R1 already recorded, no new turn appears, and it returns the existing result. Step 3: a new `requestId R2` still carrying `expectedRevision 3` arrives; a red "conflict — reload" badge appears and the card stays at 4. Step 4: the client GETs the snapshot, sees revision 4, and resubmits R2 with `expectedRevision 4`, which succeeds.

## Backups, upgrades and rolling back {#backup-rollback}

Once it works, make sure you can get back to this point. The model file can be downloaded again from its pinned revision. Your users' conversation history can't. So the database is what you protect. This drill is from the README and runs from `ai-farm` on A:

```bash
mkdir -p backups
chmod 700 backups
docker compose --env-file .env -f control/compose.yml stop gateway api1 api2
docker compose --env-file .env -f control/compose.yml exec -T db \
  pg_dump -U farm -d farm -Fc > backups/farm-before-change.dump
test -s backups/farm-before-change.dump
docker compose --env-file .env -f control/compose.yml exec -T db \
  pg_restore --list < backups/farm-before-change.dump
docker compose --env-file .env -f control/compose.yml exec db createdb -U farm farm_restore_test
docker compose --env-file .env -f control/compose.yml exec -T db \
  pg_restore -U farm -d farm_restore_test --no-owner --exit-on-error < backups/farm-before-change.dump
docker compose --env-file .env -f control/compose.yml exec db \
  psql -U farm -d farm_restore_test -c '\dt'
docker compose --env-file .env -f control/compose.yml up -d api1 api2
docker compose --env-file .env -f control/compose.yml up -d --force-recreate gateway
```

`pg_dump -Fc` writes a *logical* backup (the database's contents, not its raw files) in Postgres's compressed custom format, which `pg_restore` can list and restore. Postgres documents that `pg_dump` makes consistent exports even while the database is in use. The lab stops the writers anyway, just to make the backup boundary obvious. `-T` turns off the pseudo-terminal so binary data can pass through the redirect cleanly. `test -s` checks the file isn't empty. The last line recreates the gateway because the APIs were restarted.

A dump that exists isn't a recovery capability. A dump you've **restored and checked** is. `\dt` only lists tables. A real drill compares row counts, a few representative conversations, ownership, request IDs and terminal statuses. Use a new restore-test database name on each drill. Encrypt the backups, copy them off A under a retention policy, and back up the secrets separately.

Two terms interviewers love. **RPO** (recovery point objective) is how much committed data you can afford to lose. **RTO** (recovery time objective) is how long you can afford to be down. Worked example: a nightly dump at 02:00 and a disk failure at 23:00 the same day means up to 21 hours of turns lost. Adding a third GPU changes that by exactly zero hours. Only more frequent backups, WAL archiving or replication do (none of which this lab includes).

And the one command to tattoo on your hand: **never `docker compose down -v`** as a restart. The `-v` deletes named volumes, and that includes `postgres_data`.

### Rolling back a release, not a tag

Treat a deployment as a **bundle**: API source or image, database schema, model hash, runtime digest and generation flags. Save the previous `.env` and manifest securely before changing anything.

```text
 Model upgrade / rollback on the inference tier (no admin drain endpoint in this scaffold)

  freeze new chats ──► change ONE host's pinned image/model ──► up -d ──► direct probes
        (maintenance       (LLAMA_IMAGE / MODEL_* in .env)                 (/health + auth stream)
         window)                                                                  │
                                                                                  ▼
  reopen chats ◄── probes pass on both ◄── bring 2nd host to the SAME approved identity
        │
        └── failure at any step? restore previous .env values ──► up -d ──► re-probe ──► reopen
```

There's no admin endpoint for draining, meaning stopping new work to one replica while in-flight work finishes. So the simplest safe model change uses a maintenance window that stops new chats. Change one host, test it directly, bring the second host to the *same* identity, and only then reopen shared routing. Serving two different models behind one `corp-qwen` alias makes answers depend on which GPU a request happened to land on, and regressions become very hard to diagnose.

For an **API** rollback, restore the previous source or image, then recreate the APIs *and* the gateway. Check database compatibility separately, because an old API may not understand a new schema. Schema rollback needs a compatible migration or a tested restore. And changing the Postgres image's major-version tag isn't an upgrade or a rollback. The data directory is tied to its major version. Recreating any container cuts off the streams it's serving, so announce maintenance and have clients reconcile snapshots afterwards.

### The honest limits list

Say these out loud in a design review. This lab has one `Admin` account, one control host, no HA database, no SSO, no global admission limiter, no durable job worker, no migration system, no automatic drain, no TLS configuration and no backup scheduler. The API's numbers are 120 s timeout, 150 s stale lease, 512 output tokens and 16,000 output characters, 2000-character user messages, 6000 characters of selected history, a 16 KiB JSON body, 4 handlers per replica, 8-hour sessions and 20 login attempts per process per minute. They're scaffold limits to inspect, not sizing advice. The way to defend a small design in an interview is to name the invariants it *does* enforce (durable user turn before inference, honest terminal status, no silent replay, one authoritative database) and then propose the next mechanism only when a real requirement demands it.

> 🎬 **Animation — backup that isn't a backup until restored:** Step 1: a `.dump` file appears in `backups/` with a checkmark labelled "exists". Step 2: `pg_restore --list` scrolls a table of contents: second checkmark. Step 3: restore into `farm_restore_test`; show row counts on the live DB (e.g. 3 conversations, 12 turns, toy numbers) and on the restored DB side by side, and they match: third, bold checkmark labelled "recovery capability". Step 4: a timeline from 02:00 (backup) to 23:00 (disk dies) shades the 21 hours of lost turns in red, captioned "RPO: GPUs don't help here".

# Interview

## Question

You've deployed this two-GPU farm. After adding the second GPU host, users sometimes see a long pause and then an incomplete answer. An operator proposes three fixes: turn on sticky sessions at Nginx, double `--ctx-size`, and let Nginx retry any failed POST. How do you investigate, and what do you say about each proposal?

## Answer

I'd start by separating correctness from latency, because they have different causes. I'd pick one affected request and follow its `requestId` through the API logs, the turn's status in Postgres, and which inference host it went to. If the pauses and incomplete answers cluster on the new host, I'd diff its manifest against the old one: image digest, model SHA256, flags (`--parallel`, `--ctx-size`, `--no-kv-unified`), driver, and firewall reachability from the control host. I'd also confirm it rejects unauthenticated generation, since a misconfigured key produces confusing failures.

For the pause itself, I'd look at queueing first. Each API admits 4 handlers, so two replicas admit 8, while the GPUs have 4 slots in total. Waiting inside llama.cpp shows up as a long time to first token. I'd also check that `proxy_buffering off` is still in effect, because a buffering proxy makes a healthy stream look like a pause and then a burst. And I'd check whether the 120 s API timeout is firing, which would produce exactly "pause, then incomplete".

On the proposals. **Sticky sessions** can only improve KV-cache warmth. They can't fix a lost commit or a slow GPU, and the APIs are stateless by design. GPU affinity is already handled inside the API by a conversation hash. **Doubling context** with non-unified KV raises the per-slot share (8192 → 16384 total means 4096 → 8192 per slot) and roughly doubles KV memory (≈448 → ≈896 MiB with a 16-bit cache). That could cause load failures or pressure, and it does nothing for queueing. I'd change it only after measuring token counts and memory. **Retrying failed POSTs** is the dangerous one. If the first API already committed the user turn or streamed tokens, a replay duplicates generation or stitches two answers together. Keep `proxy_next_upstream off`. The client should reconcile with the snapshot and send a deliberate new request under the current revision.

Finally, I'd run a controlled failover test with synthetic content: stop one GPU mid-stream, confirm the turn ends `partial` or `error` and the next turn goes to the healthy host, and write down the capacity lost with one GPU down.

## Follow-ups

- How would you replace per-replica admission with a global budget without holding a database transaction open during generation?
- Which fields in a deployment manifest establish that two inference endpoints are interchangeable?
- How would you drain active SSE streams before replacing an API replica, given Nginx resolves upstream names at start?
- What RPO and RTO does this lab actually offer, and what would you add to get RPO down to minutes?

# Pitfalls

- Treating two GPU hosts as high availability. They add inference capacity, but the only gateway and database are both on A.
- Expecting the Compose service name `llama` on B to resolve from A. Compose DNS only works inside one project's network, so cross-host traffic needs LAN addresses or managed DNS.
- Treating a matching `corp-qwen` alias as proof of identical models. Only the file hash, runtime digest, template and flags together establish that.
- Treating a character cap or `--ctx-size` as a verified token budget. Tokens per character vary a lot, and the real template overhead has to be measured.
- Editing `schema.sql` to change an existing database, or deleting the Postgres volume to "rerun init". Init scripts only run on an empty data directory, and deleting the volume destroys history.
- Letting a proxy or client silently retry a generation after tokens were emitted. The result can be duplicated or stitched-together answers instead of an honest `partial` or `error`.
- Assuming the host firewall's INPUT policy protects Docker-published ports. Forwarded container traffic can take a different path, so test from an unauthorized machine.
- Running `docker compose down -v` as a routine restart. It deletes the named database volume.
- Saying "we tested it" when only the config and docs were reviewed. Nothing in this guide was executed.

# Checklist

- Reserve and verify private addresses and routing on both hosts, and confirm TCP 8080 is refused from an unauthorized machine.
- Pass the GPU gates in order: host `nvidia-smi`, toolkit configured, image pinned by digest, `--help` shows every required flag, GPU visible inside the image.
- Approve the model by revision, SHA256, LICENSE and model card, then download one file and pass `sha256sum --check` on each host.
- Generate three separate hex secrets, keep B free of database, Admin and cloud secrets, and validate with `config --quiet`.
- Compute the per-slot budget (8192 ÷ 2 = 4096; 4096 − 512 − 256 = 3328 illustrative) and the KV size (≈448 MiB), then confirm the per-slot context in the startup logs.
- Probe `/health` and an authenticated stream on both endpoints from A, and confirm an unauthenticated request is refused.
- Bring up the control project and check `ps`, `nginx -t` and `/healthz` through the gateway.
- Run the replay, stale-revision, no-cookie, cloud-off, GPU-stop and API-stop tests, and reconcile with the snapshot each time.
- Take a backup, restore it into a test database, compare contents, and copy it off-host before any upgrade.
- Upgrade or roll back one host at a time, and reopen shared routing only when both hosts match the approved identity.

# Sources

- [llama.cpp HTTP server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) — Flag spellings and meanings (`--parallel`, `--ctx-size`, `--kv-unified`/`--no-kv-unified`, `--no-context-shift`, `--n-gpu-layers`, `--n-predict`, `--alias`), the `LLAMA_API_KEY` variable, and `/health` being public with 503 while loading and 200 when ready. Upstream changes, so check the pinned image's `--help`.
- [llama.cpp Docker documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md) — The official `server-cuda` image family on ghcr.io and GPU `docker run` examples.
- [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — `nvidia-ctk runtime configure --runtime=docker`, the Docker restart, and the different rootless procedure.
- [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/) — The device reservation syntax. `capabilities` must be set, and `count` and `device_ids` are mutually exclusive.
- [Docker Compose variable interpolation](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/) — `--env-file`, `${VAR:?error}` and `${VAR:-default}`, and that `.env` values interpolate the file rather than being injected into containers automatically.
- [Qwen2.5-7B-Instruct-GGUF on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) — Available quantizations including Q3_K_M, split files for larger quantizations, and the Apache-2.0 license listing.
- [Qwen2.5-7B-Instruct config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) — 28 layers, 28 attention heads, 4 KV heads and hidden size 3584, used for the KV-cache arithmetic.
- [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) — `hf download` with explicit filenames, `--revision` and `--local-dir`, and installation options.
- [Docker official Postgres image](https://hub.docker.com/_/postgres) — The Postgres 18 version-specific PGDATA (mount `/var/lib/postgresql`), and init scripts that run only on an empty data directory.
- [Nginx proxy module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html) — `proxy_buffering`, `proxy_read_timeout` as a gap between reads, `proxy_next_upstream` and non-idempotent POST behaviour, and the `proxy_http_version` default change.
- [Nginx upstream module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html) — `least_conn` balances by active connections, `keepalive` needs HTTP/1.1 with a cleared Connection header, and the optional `resolve` parameter.
- [PostgreSQL pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html) — Consistent exports during concurrent use, and the custom `-Fc` format for `pg_restore`.
- [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) — The `text/event-stream` format, `data:` lines, and blank-line event separation.

# Flashcards

## topology

**Q:** Why don't two GPU hosts make this lab highly available?

They're two independent *replicas* of the same model, which gives more generation capacity. But the only gateway (Nginx) and the only database (Postgres) both live on host A. If B dies you lose half your capacity. If A dies, nobody can log in or read history, even though B's GPU is fine. Real HA would need a replicated database and a redundant front door, not more GPUs.

## dns

**Q:** Why can't the API on host A reach B's inference server by the Compose service name `llama`?

Compose runs a small DNS service scoped to each project's own network on one host. B's inference project is a separate project on a separate machine, so its service names don't exist from A. Cross-host traffic uses routable LAN addresses (`INFERENCE_B_URL=http://10.20.0.12:8080`), managed DNS, or a deliberately configured overlay network.

## runtime

**Q:** What should pass before you debug a GGUF that won't load?

`nvidia-smi` on the host, then `nvidia-smi` *inside* the pinned image (`docker run --rm --gpus all --entrypoint nvidia-smi "$LLAMA_IMAGE"`). That isolates driver and container-runtime problems from model-memory and flag problems. Swapping models can't fix a GPU the container can't see.

## identity

**Q:** What establishes that two inference endpoints serve the same approved model?

Matching repo revision and full-file SHA256, plus the same runtime image digest, chat template behaviour and server flags, all recorded in a deployment manifest. The shared alias `corp-qwen` is only a label, and you could serve two different files under it.

## hash

**Q:** What does a successful SHA256 check *not* prove?

It proves the bytes on disk equal an expected value. That's all. It says nothing about whether the expected value itself came from a trustworthy source, whether the license suits you, or whether the model is good or safe. Those come from the review done at approval time (revision page, LICENSE, model card, evaluation).

## slots

**Q:** Why does the Compose file set `--no-kv-unified` as well as `--ctx-size` and `--parallel`?

How the context gets divided depends on the KV mode and the llama.cpp version. With separate per-slot KV, 8192 total ÷ 2 slots = 4096 tokens per slot. A unified buffer shares one pool instead. Defaults have changed over time, so stating the mode explicitly (and checking the startup log) keeps the intended per-request capacity reviewable.

## budget

**Q:** In the worked example, how many tokens of conversation fit in one slot?

4096 (slot) − 512 (reserved output) − 256 (assumed template/system overhead) = 3328 tokens. The 256 is illustrative, so replace it with the token count of your actual rendered template. The API's 6000-character history cap is roughly 1500 tokens of typical English, but characters aren't tokens, so it's only a crude guard.

## ready

**Q:** Does a 200 from llama.cpp's `/health` mean a new request will start immediately?

No. `/health` returns 503 while loading and 200 once the model is loaded. That's readiness. Whether a slot is free is a separate question about load. It's like a restaurant being open versus having a free table. Queueing and overload have to be measured separately (for example, time to first token).

## init

**Q:** Why doesn't editing `schema.sql` change an existing database?

The official Postgres image runs `/docker-entrypoint-initdb.d` scripts only when the data directory is empty. After the first initialization they never run again. Existing databases need explicit, reviewed migrations. Deleting the volume to force a rerun destroys every conversation.

## sse

**Q:** Why turn `proxy_buffering` off for chat?

With buffering on, Nginx collects upstream output in buffers and sends it in bigger pieces, so a token-by-token SSE stream looks like a long pause and then a burst. With it off, each chunk goes to the client as it arrives. Timeouts still need to be bounded, which is why the API aborts at 120 s, before Nginx's 180 s read gap.

## retry

**Q:** Why is `proxy_next_upstream off` set instead of letting Nginx retry failed requests?

A failed chat POST is ambiguous. The first API may already have committed the user turn or streamed tokens. Replaying it could duplicate work or splice two answers together. Nginx already won't resend a POST once it's been sent, and `off` also disables retries after connection errors, giving one simple rule: the proxy never replays. The client reads the snapshot and makes an explicit new request.

## restore

**Q:** When does a backup count as real recovery evidence?

Only after you've restored it (here, into a fresh test database) and checked representative data: row counts, sample conversations, ownership, request IDs and terminal statuses. A dump file that exists, or even a `pg_restore --list` that works, hasn't yet proved you can bring the service back.

## rollback

**Q:** Why isn't going back to an older image tag enough after a schema change?

The older application may not understand the newer schema. A release is a bundle: code or image, schema, model hash, runtime digest and flags. Rolling back safely needs a compatible migration path or a tested database restore, not just an older binary. And changing the Postgres major-version tag is neither an upgrade nor a rollback.

## lablimit

**Q:** Why don't two logged-in browsers test multi-tenant isolation in this lab?

Every login is the single `Admin` demonstration identity, so both sessions belong to the same owner. Isolation tests need distinct users and checks that one can't read the other's conversations, which this scaffold doesn't have.

## env-interpolation

**Q:** Does `docker compose --env-file .env` put every variable from `.env` into every container?

No. `--env-file` supplies values for *interpolation*, filling `${VAR}` placeholders in the YAML. Each service only receives what its `environment:` block lists. That's why B's inference container never sees the database password even if you forget to remove it, although you should still remove it from B's file. `${VAR:?msg}` makes a missing secret fail at `config` time.

## kv-size

**Q:** Roughly how much KV cache does the example's 8192-token total context need for Qwen2.5-7B with a 16-bit cache?

Per token: 2 (K and V) × 28 layers × 4 KV heads × 128 dims × 2 bytes = 57,344 bytes ≈ 56 KiB. Times 8192 tokens = 469,762,048 bytes = 448 MiB, on top of roughly 3 GB of Q3_K_M weights (illustrative) and runtime buffers. GQA's 4 KV heads (instead of 28) is what keeps it this small.
