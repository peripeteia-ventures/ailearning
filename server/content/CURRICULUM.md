# Curriculum map

This file says what each article owns, so writers can point ahead instead of re-teaching, and so no two articles fight over a topic. If something belongs to another article, give a short intuition and link to it by title. Order and categories match `shared/catalog.ts`.

## Shared notation (use consistently)

| Symbol | Meaning |
|---|---|
| `T` (or `n`) | sequence length in tokens |
| `B` | batch size (sequences) |
| `V` | vocabulary size |
| `d` (`d_model`) | hidden/embedding width |
| `h`, `d_head` | number of attention heads, per-head width (`d = h · d_head`) |
| `h_kv` | number of key/value heads (GQA/MQA) |
| `L` | number of transformer layers (blocks) |
| `N`, `D` | parameter count, training tokens |
| `W_Q, W_K, W_V, W_O` | attention projection matrices |

Rows of an activation matrix are **token positions**, and columns are **features**. Say which explicitly whenever you show a matrix.

## 1. How LLMs work (`foundations`)

1. **`tokens-and-embeddings`: Text in, next token out: tokens, embeddings, and sampling** (Core, no prereqs). This is the whole pipeline at a bird's-eye view, and the "black box" middle is explicitly deferred. It covers the next-token-prediction contract and the autoregressive loop; tokenization (why not characters or words, BPE by hand, token counts, cost and context budget, multilingual/code quirks); the embedding table as a lookup; vectors and similarity intuition; the transformer stack as a black box (pointing to the next two articles); the output projection (unembedding, tied weights) to logits; softmax to probabilities; and decoding (greedy, temperature, top-k, top-p, stop conditions, determinism). It briefly touches on why generation is sequential.
2. **`attention-from-scratch`: Attention from scratch: how tokens talk to each other** (Core, prereq tokens-and-embeddings). It covers why a token needs its context; the Q/K/V intuition (a soft lookup, like a library or a search); dot-product similarity; scaling by √d_k; softmax weights; the weighted sum of values; a full tiny numeric worked example; the causal mask; multi-head attention and the output projection; shapes end to end; the O(T²) cost; and a preview that K and V get cached at inference (pointing to serving-kv-cache) and that cheaper variants exist (pointing to efficient-attention).
3. **`transformer-foundations`: The transformer block: assembling the full model** (Core, prereqs the two above). It covers why attention alone is order-blind; positional information (learned, sinusoidal, RoPE intuition); the residual stream as a shared "bus"; LayerNorm vs RMSNorm and pre-norm; the MLP/SwiGLU as per-token processing; stacking L blocks; counting parameters (a worked example, e.g. a 7B-class model); what training looks like (shifted targets, all positions in parallel, cross-entropy briefly, pointing to optimization-generalization) versus generation (one token at a time); and why that asymmetry creates the KV cache (pointing to serving-kv-cache).

## 2. Training (`training`)

4. **`optimization-generalization`: How a model learns: loss, gradients, and optimizers** (Core). It covers cross-entropy on next tokens (and perplexity); gradients and backprop intuition (the chain rule on a tiny graph); gradient descent; the learning rate, warmup and cosine decay; momentum, Adam and AdamW (optimizer state per parameter); batch size and gradient noise; gradient accumulation; mixed precision (bf16/fp32 master weights, loss scaling); gradient clipping; loss spikes; and overfitting vs generalization, validation and held-out sets.
5. **`pretraining-data-scaling`: Pretraining: data, compute, and scaling laws** (Advanced). It covers what pretraining produces (a base model); building the corpus (sources, filtering, quality classifiers, dedup, contamination, mixtures, repetition); the tokenizer's effect on budget; the compute estimate C ≈ 6·N·D; scaling laws (Kaplan vs Chinchilla, ~20 tokens/param as a compute-optimal heuristic); over-training smaller models for cheaper inference; and what the fits can't promise. Cluster mechanics are pointed to distributed-training.
6. **`distributed-training`: Distributed training: fitting a training run onto a cluster** (Systems, *new*). It covers memory accounting per parameter (weights, gradients and Adam states, ~16 bytes/param in mixed precision) plus activations and activation checkpointing; data parallelism and all-reduce; ZeRO/FSDP sharding stages; tensor parallelism (splitting matmuls); pipeline parallelism and bubbles (micro-batches); sequence/context parallelism briefly; expert parallelism (pointing to moe-and-parallelism); mapping to hardware (NVLink inside a node, InfiniBand/Ethernet across nodes); MFU; and checkpointing, failures and restarts at scale.

## 3. Post-training (`alignment`)

7. **`post-training-alignment`: From base model to assistant: SFT, RLHF, DPO, and verifiable rewards** (Advanced). It covers base vs assistant behaviour; chat templates; SFT with loss masking; reward models (Bradley–Terry); PPO-style RLHF with the KL anchor; DPO; RL with verifiable rewards and reasoning models; reward hacking; and evaluating the pipeline.
8. **`parameter-efficient-finetuning`: Fine-tuning on a budget: LoRA and QLoRA** (Advanced). It covers when to prompt, use RAG or fine-tune; full fine-tuning memory; LoRA math (low rank, rank r, alpha, where adapters go); QLoRA (4-bit NF4 storage, bf16 compute, paged optimizers); a memory budget worked example; data (templates, loss masks, quality over quantity); evaluation; and serving adapters (merge vs hot-swap, multi-LoRA).

## 4. Inference & serving (`inference`)

9. **`serving-kv-cache`: What happens at inference: prefill, decode, and the KV cache** (Systems). It covers the two phases (prefill is compute-bound, decode is memory-bandwidth-bound, with an arithmetic intensity intuition); the metrics TTFT, TPOT/ITL and throughput; the KV cache from scratch (what's stored and why it's reusable); the size formula and a worked example; memory as the concurrency limit; PagedAttention blocks and fragmentation; prefix caching; and preemption and eviction.
10. **`inference-scheduling`: Serving many users: batching, scheduling, and speculative decoding** (Systems). It covers static vs continuous batching; prefill/decode interference and chunked prefill; SLOs and admission control; queueing basics (Little's law); a worked capacity example; speculative decoding (draft/verify, acceptance rate); and prefill/decode disaggregation.

## 5. Efficiency levers (`architecture`)

11. **`efficient-attention`: Making attention cheaper: GQA, FlashAttention, and long context** (Advanced). It covers the three different costs (FLOPs, memory traffic, KV cache size); MHA vs MQA vs GQA; KV-size arithmetic; FlashAttention (tiling, online softmax, same math with less HBM traffic); RoPE and context extension (scaling, "lost in the middle"); and sliding-window and sparse attention.
12. **`quantization-and-compression`: Quantization: spending fewer bits per weight** (Advanced). It covers what's scarce (memory vs bandwidth vs compute); number formats (fp16/bf16/fp8/int8/int4); scale and zero-point with a worked example; granularity (per-tensor/channel/group); PTQ vs QAT; GPTQ and AWQ; weight-only vs activation vs KV-cache quantization; kernels and formats (GGUF etc.) versus actual speedups; measuring quality; and pruning and distillation as alternatives.
13. **`moe-and-parallelism`: Mixture of experts: more parameters, same compute per token** (Advanced). It covers dense vs sparse; the router and top-k gating; total vs active parameters (a worked example); load balancing (aux loss, capacity factor, token dropping); expert parallelism and all-to-all communication; serving implications (all experts in memory, batching effects); and MoE vs dense tradeoffs. General DP/TP/PP is pointed to distributed-training.

## 6. RAG & agent systems (`applications`)

14. **`rag-retrieval`: RAG: giving the model the right evidence** (Advanced). It covers why RAG exists; ingestion and chunking; embedding models and vector similarity; ANN/HNSW; hybrid (BM25 + dense) and reranking; permissions and freshness; packing context and citations; evaluating retrieval vs generation; and RAG vs fine-tuning.
15. **`agents-tool-systems`: Agents and tools: from model decisions to safe actions** (Advanced). It covers the agent loop; tool calling and structured output; state and recovery; permissions and approvals; idempotent retries; context and memory; termination; evaluating trajectories; and when not to build an agent.

## 7. Evaluation & safety (`evaluation`)

16. **`evaluation-observability`: Evaluation and observability: knowing whether it actually works** (Systems). It covers eval as a decision; datasets and contamination; evaluator types (exact, programmatic, human, LLM-as-judge and its calibration); paired comparisons and statistical uncertainty; production traces and metrics; cost accounting; canary gates; and feedback loops.
17. **`llm-security`: LLM security: treating model output as untrusted** (Systems). It covers the threat model (assets, authority); direct and indirect prompt injection; RAG authorization; tool authorization; exfiltration channels; sandboxing; resource abuse; and red-teaming and change governance.

## 8. The corporate AI farm (`ai-farm`)

These three must stay consistent with `server/content/FARM-CONTRACT.md` and `examples/ai-farm/`.

18. **`farm-blueprint`: AI farm, part 1: designing a two-server Qwen deployment**
19. **`farm-deployment`: AI farm, part 2: deploying it with Docker, step by step**
20. **`farm-routing-consistency`: AI farm, part 3: routing and keeping conversations consistent**

## 9. System design interview (`system-design`)

21. **`interview-design`: The senior LLM system design interview**. It pulls everything together. It references earlier articles rather than re-teaching them, while briefly re-stating any fact it relies on.
