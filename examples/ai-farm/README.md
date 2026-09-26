# Two-host AI farm teaching example

These are Linux Docker instructions, separate from the Windows learning app. Nothing here has been deployed or benchmarked by the authors. The core example requires your GPU/driver compatibility checks, secrets, model verification and image digest selection. It is a single-account teaching scaffold, not a production identity service. Download the complete example from `/examples/ai-farm.zip` in the learning app.

## Layout and boundary

Host A, example private address `10.20.0.11`, runs inference plus the `control` Compose project: Postgres, api1, api2 and Nginx. Host B, `10.20.0.12`, runs a second identical inference service. Two GPUs are independent replicas, not tensor parallel shards. A CPU-only third control host is also possible: update firewall sources accordingly. Reserve real addresses through your network administrator/DHCP service. Example addresses are not automatically assigned by Compose.

Browser -> Nginx :8088 -> Express :3000 -> inference A/B :8080. Only Express reads/writes Postgres. The gateway publishes loopback by default; use SSH forwarding from your workstation (`ssh -N -L 8088:127.0.0.1:8088 user@10.20.0.11`) then open `http://localhost:8088`. Avoid raw inference UI for normal chat: it bypasses central authentication/history. Permit inference TCP8080 only from control hosts at your network firewall; verify Docker-published-port filtering for your Linux distribution. Do not assume a host firewall's default INPUT rule filters Docker forwarding. Inference keys travel over HTTP here; use a trusted isolated lab network, or TLS/mTLS between machines before using a shared/untrusted network.

## 1. Prepare both Linux hosts

Install Docker Engine and its Compose v2 plugin from Docker's instructions for your distribution, a compatible NVIDIA driver, and NVIDIA Container Toolkit from its official installation guide. Check architecture and CUDA compatibility before choosing a llama.cpp image. This example uses rootful Docker; rootless setup differs. Then on each GPU host:

```bash
nvidia-smi
docker version
docker compose version
ip -brief address
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Restarting Docker interrupts existing containers; schedule this when appropriate. Copy the extracted `ai-farm` directory to both hosts. Run all subsequent commands from that directory unless stated otherwise. On A, `ip -brief address` must show its assigned private address; B must show its own. Test routing with `ip route get 10.20.0.12` on A and reverse on B. Container service name `llama` does not resolve across these two independent Compose projects.

## 2. Configure secrets and reproducible identities

```bash
cp .env.example .env
chmod 600 .env
openssl rand -hex 32
```

Run the last command three times and put different generated values into `POSTGRES_PASSWORD`, `ADMIN_PASSWORD`, and `INFERENCE_API_KEY`. Hex passwords avoid URI escaping ambiguity in `DATABASE_URL`. Use the same inference key on the control host and both inference services. Only the control host needs DB/Admin/cloud secrets; remove those values from B's file. Replace `INFERENCE_BIND_IP` on each host and both endpoint URLs on A. Keep the model identity equal on both. `.env` is operator-controlled configuration: never execute an untrusted downloaded environment file as shell code. For commands below, after manually reviewing your own `.env`, load its plain assignments with `set -a; . ./.env; set +a`.

Select a candidate official image, inspect its help and record the immutable registry digest on your Linux host:

```bash
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda --format '{{index .RepoDigests 0}}'
# Put the returned ghcr.io/...@sha256:... in LLAMA_IMAGE in .env.
# Reload your reviewed environment, then inspect the exact image:
docker run --rm "$LLAMA_IMAGE" --help
docker run --rm --gpus all --entrypoint nvidia-smi "$LLAMA_IMAGE"
```

The placeholder digest intentionally cannot run. Verify support for `--no-kv-unified`, `--parallel`, `--ctx-size`, `--n-gpu-layers`, `--no-context-shift`, and `LLAMA_API_KEY`; reject a candidate that lacks them. Pin NGINX_IMAGE and POSTGRES_IMAGE to tested digests as well. Pin the API base image and dependency lock during production packaging; this scaffold's API build is not a hermetic release. Keep a deployment manifest of digests, CLI version/help, driver, model revision/hash and configuration with secrets redacted.

## 3. Verify and download one GGUF per host

Install Hugging Face's `hf` CLI using its official instructions. The chosen Q3_K_M file is one file, simplifying this lab; lower precision needs independent task-quality evaluation. It is not a recommendation over Q4 or Q5. Model revision is `bb5d59e06d9551d752d08b292a50eb208b07ab1f`; the SHA256 observed for the Q3_K_M file is `a96b16179dc6cc9afdf0cf7a96a80c199cbd00b9be207c3465be21cb721cca5e`. Before downloading, manually open the model's file page **at that revision**, verify its SHA256 against `.env`, review its Apache-2.0 LICENSE and model card at that revision, and save the evidence. The documentation author opened the commit, current file metadata and revision LICENSE; immutable file-page rendering failed, so your revision/hash check remains required. Do not substitute the Xet hash for the content SHA256.

```bash
# Run after loading your reviewed .env as above. Provision a writable directory.
sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "$MODEL_DIR"
hf download Qwen/Qwen2.5-7B-Instruct-GGUF "$MODEL_FILE" LICENSE README.md \
  --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_DIR/$MODEL_FILE" | sha256sum --check -
```

Stop on any mismatch. Repeat independently on B or securely copy the approved artifact and recheck the hash. Keep license/model-card files with the artifact. Do not download all quantizations or rename an LFS pointer to a GGUF. The model bind mount is read-only and Compose will refuse to create a missing model directory. No weights are included in the zip.

## 4. Start inference on A and B

```bash
docker compose --env-file .env -f inference/compose.yml config --quiet
docker compose --env-file .env -f inference/compose.yml up -d
docker compose --env-file .env -f inference/compose.yml logs --tail=100 llama
```

Read startup logs for actual CUDA offload and per-slot context. The explicit `--no-kv-unified`, total context8192 and parallel2 aim for4096 tokens per slot on a matching tested build. Reserve output512 plus chat-template tokens; the API character bound is not an exact tokenizer guarantee. Two slots are not a promise of two requests at full single-user speed. Change only one capacity variable at a time and load-test on the target hardware.

From A (with its reviewed environment loaded):

```bash
curl --fail-with-body "$INFERENCE_A_URL/health"
curl --fail-with-body "$INFERENCE_B_URL/health"
curl --fail-with-body -N "$INFERENCE_A_URL/v1/chat/completions" \
  -H "Authorization: Bearer $INFERENCE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"corp-qwen","messages":[{"role":"user","content":"Reply with hello."}],"max_tokens":32,"stream":true}'
```

Repeat streaming test against B. `/health` is public and model readiness is distinct from available capacity. The inference image need not include curl; these probes run from the host. Confirm unauthenticated generation is rejected. Do not record headers containing credentials.

## 5. Start the control project on A

```bash
docker compose --env-file .env -f control/compose.yml config --quiet
docker compose --env-file .env -f control/compose.yml up -d --build
docker compose --env-file .env -f control/compose.yml ps
docker compose --env-file .env -f control/compose.yml exec gateway nginx -t
curl --fail-with-body http://localhost:8088/healthz
```

The API build context is `../api`, resolved from control/compose.yml, and both replicas use internal port3000. The schema mounts as an init script for an **empty** Postgres volume. It does not rerun on every start. Subsequent schema changes need explicit reviewed migrations, not volume deletion. Postgres18 stores its versioned data directory beneath `/var/lib/postgresql`; the named volume mounts that parent. DB and API have no published host ports. `depends_on` health gates startup, not ongoing failover. Gateway least_conn balances active connections, not tokens or GPU load. SSE buffering and proxy retry are disabled. A replaced API container can get a new IP; recreate the gateway after API recreation so its static upstream names resolve again.

## 6. Test central auth, revision and streaming

Open the SSH-forwarded UI, log in as `Admin` with your generated password, create a conversation and send a short local request. To test HTTP directly, `curl`, `jq`, and `uuidgen` must be installed:

```bash
origin=http://localhost:8088
umask 077
jq -n --arg password "$ADMIN_PASSWORD" '{username:"Admin",password:$password}' > login.json
curl --fail-with-body -c cookies.txt -H "Origin: $origin" -H 'Content-Type: application/json' \
  --data-binary @login.json "$origin/api/login"
rm login.json
created=$(curl --fail-with-body -b cookies.txt -H "Origin: $origin" -H 'Content-Type: application/json' \
  -d '{}' "$origin/api/conversations")
cid=$(printf '%s' "$created" | jq -r .id)
rev=$(printf '%s' "$created" | jq -r .revision)
request_id=$(uuidgen)
jq -n --arg requestId "$request_id" --argjson expectedRevision "$rev" \
  '{requestId:$requestId,expectedRevision:$expectedRevision,text:"Explain KV cache briefly.",provider:"local"}' > turn.json
curl --fail-with-body -N -b cookies.txt -H "Origin: $origin" -H 'Content-Type: application/json' \
  --data-binary @turn.json "$origin/api/conversations/$cid/chat"
curl --fail-with-body -b cookies.txt "$origin/api/conversations/$cid"
```

Replay exactly the same turn.json to inspect idempotency; it must not create another user turn. Use a new requestId with the old revision to inspect conflict handling. Delete cookies.txt and turn.json after testing: both may carry sensitive data. Confirm a request without the session cookie cannot read the conversation. All Admin sessions share the same demonstration owner; this does not test multi-tenant isolation.

During another stream, stop a GPU service deliberately in a maintenance test. Expect partial/error state, never silently concatenated replacement output. Restore it; a new turn can use a healthy endpoint. Stop api1 and send a request: with automatic proxy retries disabled, a request selecting the stopped replica can fail. Remove/drain it from Nginx and reload, or restore it. GET the authoritative snapshot after ambiguity. The fixed150s stale lease is recovered on subsequent API activity, not by a background durable worker. Cloud calls are disabled unless ALLOW_CLOUD, provider key/model, and per-request consent all permit them; review data policy before enabling.

## 7. Backup, upgrade and rollback

Run from ai-farm on A. A logical dump is consistent without copying live database files. This lab stops API writers to make the operational boundary obvious:

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

Use a new restore-test DB name on repeated drills. Verify restored row counts, representative messages and ownership invariants; listing tables alone is not a full restore test. Encrypt/copy backups off-host under your retention policy. Preserve redacted image/model manifests and securely back up required secrets separately. Do not use `down -v`: it deletes the named database volume.

For a model upgrade, drain the affected endpoint in application routing first; this scaffold has no admin drain endpoint, so stop new chat traffic for the simplest maintenance window. Change only one host's pinned artifact, test direct generation, then bring both back to matching approved identity before reopening shared routing. Save prior `.env` securely. Roll back by restoring the old image/model settings and running inference `up -d`, then re-probe. For API changes restore the previous source/image and recreate API and gateway. Schema rollback requires a compatible migration or tested database restore; replacing the Postgres major tag is not an upgrade or rollback procedure. Existing SSE requests are interrupted by recreation, so communicate maintenance and reconcile snapshots afterward.

## Exact limits

No real deployment, GPU benchmark, image pull, model download, Compose execution or end-to-end network validation occurred while writing this example. Operator placeholders must be replaced. Single Admin account; one DB/control host; no HA database, SSO, global admission limiter, durable job worker, schema migration system, automatic drain, TLS configuration, backup scheduler or production audit policy. API timeout 120s, stale lease 150s, maximum output 512 tokens and 16000 characters, user text 2000 characters, selected prompt history 6000 characters, JSON body 16 KiB, and four active chat handlers per replica (eight across two API replicas versus four total inference slots). Sessions last eight hours; login throttling allows twenty attempts per process per minute. The six-thousand-character prompt cap is not a token guarantee. Stateful correctness lives in Postgres; restarting inference loses optional KV warmth. API replica failure can lose uncommitted partial text; the client must reconcile authoritative history.

Official references: [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [Docker images](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md), [NVIDIA runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/), [Qwen model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF), [hf CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), [Postgres image](https://hub.docker.com/_/postgres), [Nginx proxy](https://nginx.org/en/docs/http/ngx_http_proxy_module.html).

A `complete` turn means successful stream termination, not verified answer quality or absence of token-limit truncation. The scaffold does not store upstream `finish_reason` metadata.
