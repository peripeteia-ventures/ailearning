---
{
  "slug": "post-training-alignment",
  "title": "From base model to assistant: SFT, RLHF, DPO, and verifiable rewards",
  "category": "alignment",
  "summary": "How a next-token predictor becomes an assistant: chat templates and SFT teach the format, reward models and RLHF or DPO teach preferences, and verifiable rewards train reasoning. Along the way you'll see why every one of these signals can be gamed and how to evaluate the whole pipeline.",
  "difficulty": "Advanced",
  "minutes": 30,
  "prerequisites": ["transformer-foundations", "optimization-generalization"],
  "learningObjectives": [
    "Explain why a base model continues text instead of answering, and how chat templates and loss-masked SFT change that.",
    "Calculate a Bradley–Terry reward-model loss, a clipped PPO term, a KL-penalised reward, a DPO loss, and group-relative advantages by hand.",
    "Explain what DPO assumes, how it relates to KL-regularised RLHF, and when you'd pick SFT, DPO, PPO, or RL with verifiable rewards.",
    "Design reward validation, reward-hacking controls, and independent evaluation gates for a post-training pipeline."
  ]
}
---

# Sections

## What pretraining leaves you with: a brilliant autocomplete {#base-vs-assistant}

Pretraining produces a **base model**: a network trained to predict the next token on trillions of tokens of web pages, books and code. (We cover how that corpus gets built in **Pretraining: data, compute, and scaling laws**.) A base model knows a staggering amount, but it has exactly one job: continue the text in whatever way the internet most likely would.

That's why a base model often doesn't *answer* you. Type a question and it may produce more questions, because on the web a question is often followed by another question on the same quiz page.

```text
Prompt to a BASE model:
  "What is the capital of France?"

Plausible continuations (all "correct" next-token behaviour):
  "What is the capital of Germany? What is the capital of Spain?"   ← quiz page
  "  A) Paris  B) Lyon  C) Nice"                                     ← test paper
  "Paris. Some fun facts: ..."                                        ← lucky

Prompt to an ASSISTANT model:
  "Paris."   (then it stops)
```

Here's the analogy I like. Pretraining is someone who has read the entire library but never had a job. **Post-training** is the on-the-job coaching: the same brain, taught a specific way to behave with customers. It uses far less data than pretraining (thousands to millions of examples, not trillions of tokens), and it mostly changes *behaviour* (format, tone, when to refuse, how to reason step by step) rather than adding knowledge.

A bit of vocabulary we'll use all the way through. We call the model a **policy**, a term borrowed from reinforcement learning. The policy πθ(y | x) is the probability the model gives a whole completion y after a prompt x, where θ stands for all the trainable weights. Because the model generates one token at a time, that probability is the product of the per-token probabilities, so its log is a sum:

```formula
log πθ(y | x) = Σ_t log πθ(y_t | x, y_<t)
```

Here y_t is the t-th token of the response and y_<t is everything the model already wrote before it. Remember this sum: almost every loss in this article is built out of it.

"Alignment" sounds grand, so let's pin it down. In engineering terms it means *improving specified behaviours, measured on a specified evaluation set*. It doesn't prove the model shares human values. And the behaviours pull against each other: a model that refuses everything has a perfect harmful-content score and is useless. So the first question isn't "PPO or DPO?". It's "what behaviour do we want, and what evidence do we have about it?" There are three kinds of evidence, and each maps onto a family of methods:

| Evidence you have | What it says | Method family |
|---|---|---|
| Demonstrations (good example answers) | "Say something like *this*" | Supervised fine-tuning (SFT) |
| Pairwise preferences ("A is better than B") | "This one beats that one" | Reward model + RLHF, or DPO |
| A programmatic checker (unit tests, exact answer) | "This passed / failed" | RL with verifiable rewards (RLVR) |

The classic recipe, popularised by OpenAI's **InstructGPT** paper, chains the first two: SFT on about 13k demonstration prompts, then a reward model trained on about 33k prompts' worth of rankings, then reinforcement learning (PPO) on about 31k prompts. InstructGPT reported that people preferred outputs from its 1.3B-parameter model over the 175B-parameter GPT-3 base model. That's the whole point of post-training in one sentence: behaviour beat 100× more parameters. Modern pipelines reorder, repeat and mix these stages (the open Tülu 3 recipe, for example, runs SFT, then DPO, then RLVR), so treat the order as a design choice, not a law.

> 🎬 **Animation — the post-training pipeline:** start with a box labelled "Base model (predicts web text)". Step 1: an arrow labelled "SFT: ~10k demos" leads to "SFT model (answers in chat format)". Step 2: a side branch shows humans ranking 4 answers to one prompt and feeding a "Reward model" box. Step 3: arrows from the SFT model and the reward model go into an "RL loop" box that cycles generate → score → update, with a dashed leash back to the frozen SFT model labelled "KL anchor". Step 4: an alternative arrow labelled "DPO" goes straight from the preference data to the policy and skips the reward model and RL loop. Step 5: a third branch labelled "RLVR" goes from a "unit tests / answer checker" box into the RL loop. Finish with every path flowing into an "Independent eval gate" box before a "Ship" flag.

## Chat templates: how a conversation becomes one token stream {#chat-templates}

Here's the thing people forget: a chat model is *still just a next-token predictor*. It has no built-in idea of "user" or "assistant". A conversation is a list of messages, and before the model sees it, a **chat template** flattens the list into one token sequence, using special control tokens to mark who's talking.

```text
Messages (what your code holds):
  [{role: system,    content: "You are terse."},
   {role: user,      content: "Capital of France?"},
   {role: assistant, content: "Paris."}]

After a ChatML-style template (what the model actually reads):
  <|im_start|>system\nYou are terse.<|im_end|>\n
  <|im_start|>user\nCapital of France?<|im_end|>\n
  <|im_start|>assistant\nParis.<|im_end|>
```

At inference you stop the sequence right after `<|im_start|>assistant\n` (Hugging Face calls this `add_generation_prompt`). The model has seen that pattern thousands of times in SFT, so the most likely continuation is an assistant reply, followed by `<|im_end|>`, which the server treats as the stop signal. If you forget the generation prompt, the model may just keep writing the *user's* message, because nothing told it a reply comes next.

Templates vary by model family. Two models fine-tuned from the *same* base can use completely different control tokens (the Hugging Face docs show a Mistral model using `[INST] … [/INST]` and a Zephyr model using `<|user|>` and `<|assistant|>`). The template is part of the model's contract, like a wire protocol. Get it wrong and quality drops sharply, with no error message.

The classic bugs, all silent:

- **Wrong template.** You serve with the template of a sibling model. The model sees tokens it never trained on in those positions.
- **Double BOS/EOS.** The template already inserts a begin-of-sequence token and the tokenizer adds another. (BOS/EOS are special tokens that mark the start and end of a sequence.)
- **Missing end-of-turn token in training data.** The model never learns to stop, so it rambles on and invents the next user turn.
- **Train/serve drift.** Training used one version of the template and serving uses another, e.g. with different whitespace.

A senior habit: treat the template, the special tokens and the truncation rules as a versioned part of the dataset contract. A template mismatch can cost more quality than any optimizer tweak can win back.

## SFT: learning by imitation, graded only on the answer {#sft}

**Supervised fine-tuning (SFT)** is the simplest stage: take conversations with good assistant replies (demonstrations written or approved by people, or generated by a stronger model and filtered) and train on them with the ordinary next-token loss from pretraining. The loss is **cross-entropy**: the negative log of the probability the model gave to the correct next token. (We derive it properly in **How a model learns: loss, gradients, and optimizers**.)

There's one twist. You don't want the model to learn to write the *user's* messages, so you apply a **loss mask**: every token still sits in the context, but only assistant tokens count toward the loss. It's like grading an exam. The question is printed on the page for the student to read, but you only mark what the student wrote.

```text
tokens:  <|im_start|>user  Capital  of  France ?  <|im_end|>  <|im_start|>assistant  Paris  .  <|im_end|>
mask m:        0              0      0    0    0     0              0                1     1      1
                └──────────── visible as context, no loss ───────────────┘            └─ graded ─┘
```

Note that the end-of-turn token after "Paris." is graded (m = 1). That's how the model learns to *stop*.

Training uses **teacher forcing**: at each position the model is fed the *correct* previous tokens from the demonstration, not its own guesses. That makes training fully parallel (every position is scored in one forward pass), but it hides a gap. At inference the model reads its *own* earlier tokens, and one early mistake can put it in a context no demonstration ever covered. This gap is sometimes called exposure bias, and it's one motivation for the RL stages, which train on the model's own outputs.

The objective, written out:

```formula
L_SFT = − Σ_t m_t · log πθ(y_t | x, y_<t)
```

Here x is the prompt (system and user turns), y_t is the t-th response token, and m_t is 1 for graded (assistant) tokens and 0 for masked ones. Logs are natural logs, so the unit is **nats**.

**Worked example.** Suppose a two-token graded reply, "Paris" then `<|im_end|>`, gets probabilities 0.8 and 0.5 for the correct tokens.

```text
token        p(correct)   −ln p
"Paris"        0.8        0.223
<|im_end|>     0.5        0.693
                          ─────
sum                       0.916 nats
mean per graded token     0.458 nats
```

Sum versus mean is a real choice, not a detail. If you average over *all graded tokens in the batch*, a 500-token answer counts 50× more than a 10-token one. If you average *per example first*, every conversation counts equally. Two runs that differ only here can have different "loss" numbers and learn different length preferences, so write the reduction down whenever you compare runs.

What SFT is good at: format, tone, following instructions, domain procedures, tool-call syntax. What it can't do: it only ever sees *good* answers, so it never learns that a confident wrong answer is worse than a hedged right one. It has no negative examples. That's the gap preference learning fills.

Practical data advice: coverage and consistency beat volume. A few thousand diverse, carefully checked examples usually beat a million near-duplicates. Include the awkward cases (ambiguous requests, missing information, tool errors the model should recover from) and split train/validation by source or task family so near-duplicates don't leak across. If a behaviour is missing from the data, no learning rate will conjure it up; a higher learning rate mostly erases general skills. (Cheaper ways to run SFT, such as adapters, are covered in **Fine-tuning on a budget: LoRA and QLoRA**.)

> 🎬 **Animation — the SFT loss mask:** show the token row from the section ("<|im_start|>user Capital of France ? <|im_end|> <|im_start|>assistant Paris . <|im_end|>") as tiles. Step 1: all tiles light up grey as "context". Step 2: a mask row appears underneath with 0s under the prompt tiles and 1s under "Paris", ".", "<|im_end|>". Step 3: the masked tiles fade and the three graded tiles glow. Step 4: above two graded tiles show the model's probability bars (0.8 for "Paris", 0.5 for "<|im_end|>") and the −ln values 0.223 and 0.693 dropping into a sum box reading 0.916 nats, then ÷2 = 0.458.

## Reward models: turning "A beats B" into a number {#reward-models}

Writing a perfect answer is hard. Looking at two answers and saying which is better is much easier, and faster, for a human labeller. InstructGPT leaned on this: labellers ranked between 4 and 9 responses per prompt, and every pair inside a ranking becomes a training comparison (9 responses give 9·8/2 = 36 pairs).

A **reward model** turns those comparisons into a score. It's usually the SFT model with its next-token output layer swapped for a head that outputs one number: rφ(x, y) is the score for completion y on prompt x, and φ is the reward model's own weights.

To train it we need a link between scores and "which one did the human pick". The standard choice is the **Bradley–Terry** model: the probability that the winner y_w beats the loser y_l depends only on the *difference* of their scores.

```formula
P(y_w ≻ y_l | x) = σ( rφ(x, y_w) − rφ(x, y_l) )
L_RM = − log σ( r_w − r_l )
```

Here σ(z) = 1 / (1 + e^(−z)) is the logistic (sigmoid) function, which squashes any number into (0, 1). The symbol ≻ means "is preferred to", r_w and r_l are the rewards of the winner and the loser, and the loss is the negative log of the probability the model gave to the human's actual choice.

Think of chess Elo ratings. Only the *gap* between two players predicts who wins; add 100 points to everyone and nothing changes. Same here: add any constant to every reward for a prompt and every predicted probability stays identical. So a reward of 2.7 has **no absolute meaning**. It isn't "90% correct" or "good". It's only a position on a scale that's useful for comparing answers to the same prompt.

**Worked example.** The reward model gives the human-preferred answer r_w = 1.2 and the other r_l = 0.2.

```text
gap              = 1.2 − 0.2 = 1.0
P(winner wins)   = σ(1.0)    = 0.731
loss             = −ln 0.731 = 0.313 nats

If the label had gone the other way:
P(that one wins) = σ(−1.0)   = 0.269
loss             = −ln 0.269 = 1.313 nats   ← a confident wrong ranking costs a lot
```

Now the uncomfortable part. The reward model learns *whatever predicts the labels*, including labeller biases. If labellers tend to prefer longer, more confident, better-formatted answers, the reward model learns "long and confident is good", whether or not the content is right. Then, in the next stage, an optimizer pushes as hard as it can on exactly that score. So:

- Collect pairs that separate substance from style (a short correct answer against a long wrong one).
- Randomise which answer is shown first, since position bias is real.
- Keep disagreement. When labellers split 50/50, that's information, not noise to delete.
- Evaluate the reward model on *fresh outputs from the current policy*, not just a random held-out slice of the original pairs. The policy will wander into regions the original data never covered.

A reward model is a learned proxy for human judgement. Treat it like a suspect witness, not an oracle.

> 🎬 **Animation — Bradley–Terry as a tug of war:** two answer cards A and B sit on a horizontal reward axis at 0.2 and 1.2. Step 1: a bracket between them labelled "gap = 1.0". Step 2: the gap feeds into a sigmoid curve plotted beside it; a dot slides up the curve to 0.731. Step 3: slide both cards right by +5 together; the gap bracket and the 0.731 dot don't move, with a caption "only differences matter". Step 4: swap the human label; the dot moves to 0.269 and a loss meter jumps from 0.313 to 1.313.

## RLHF with PPO: practice against the judge, on a leash {#rlhf-ppo}

With a reward model in hand, we can let the model *practise*: generate answers, get them scored, and nudge the weights toward whatever scored well. That's **reinforcement learning from human feedback (RLHF)**. "Reinforcement learning" (RL) just means learning from scores on your own attempts rather than copying fixed answers. The workhorse algorithm is **PPO (Proximal Policy Optimization)**.

The loop, one iteration at a time:

```text
 prompts ──► policy πθ ──► rollouts (sampled answers)
                               │
                ┌──────────────┼──────────────────┐
                ▼              ▼                  ▼
          reward model     reference πref     value model
          r = score        (frozen SFT copy)  "expected reward
                           log-prob for KL     from here"
                └──────────────┬──────────────────┘
                               ▼
               advantages A_t per token  ──►  PPO update (a few epochs
                                              of minibatches on this batch)
                               │
                               └──► new πθ, repeat with fresh rollouts
```

Unpacking the words:

- A **rollout** is an answer the current policy generated, which we'll learn from.
- In RL terms each generated token is an **action**, and the prompt plus the tokens so far is the **state**.
- The **advantage** A_t says how much better this action turned out than expected. Positive means "do more of this", negative means "do less".
- "Expected" comes from a **value model**, a separate network that predicts the eventual reward from a partial answer. It's the baseline you compare against. (InstructGPT initialised the value model from the reward model.)

So you're running up to four big models at once: the policy, the reference, the reward model and the value model. That's the operational cost of PPO-RLHF, and it's why the alternatives later in the article exist.

### Two separate brakes: clipping and the KL leash

PPO has two mechanisms that both sound like "don't change too much", and interviewers love checking that you can tell them apart.

**Brake 1: clipping (short-term, per update).** PPO reuses each batch of rollouts for several gradient steps, so the policy drifts away from π_old, the snapshot that *generated* the rollouts. The probability ratio ρ_t measures that drift for one token:

```formula
ρ_t = πθ(a_t | s_t) / π_old(a_t | s_t)
L_clip = E[ min( ρ_t · A_t ,  clip(ρ_t, 1−ε, 1+ε) · A_t ) ]
```

Here a_t is the token chosen at state s_t, A_t is its advantage, ε is the clip range (commonly 0.2), and clip(·) forces the ratio into [1−ε, 1+ε]. PPO *maximises* L_clip. Taking the minimum means you never get credit for pushing a ratio beyond the clip range in the direction that helps.

**Worked example (ε = 0.2, so the allowed band is [0.8, 1.2]).**

```text
Good token: A = +2, ratio has grown to 1.4
  unclipped  1.4 × 2 = 2.8
  clipped    1.2 × 2 = 2.4
  min        2.4   → flat: pushing the ratio past 1.2 earns nothing more

Bad token:  A = −2, ratio has shrunk to 0.6
  unclipped  0.6 × −2 = −1.2
  clipped    0.8 × −2 = −1.6
  min        −1.6  → flat: pushing it below 0.8 earns nothing more
```

It's a speed limiter: flooring the accelerator stops paying off. But it's a *soft* incentive on sampled tokens, not a hard guarantee. Other tokens and shared weights can still move, so the overall distribution can shift a lot.

**Brake 2: the KL anchor (long-term, whole run).** Clipping keeps each update modest, but a thousand modest updates can still walk the policy somewhere weird, like repeating phrases the reward model happens to love. So RLHF also penalises distance from a **reference policy** πref, usually the frozen SFT model. The distance is the **KL divergence**, a measure of how different two probability distributions are (zero when they're identical, positive otherwise). The objective becomes:

```formula
J(θ) = E_{y ~ πθ}[ rφ(x, y) ] − β · KL( πθ ‖ πref )
KL( πθ ‖ πref ) = E_{y ~ πθ}[ log πθ(y|x) − log πref(y|x) ]
```

β is the leash strength, in reward units per nat. In practice you estimate the KL from the sampled rollout itself (InstructGPT applies it as a per-token penalty against the SFT model). The *expected* KL is never negative, but a single sample's log ratio can be, when that answer happens to be likelier under the reference.

**Worked example.** A rollout has log-probability −12 under the policy and −14 under the reference, and the reward model gives it 1.0. With β = 0.1:

```text
sample log ratio  = −12 − (−14) = 2 nats
penalised reward  = 1.0 − 0.1 × 2 = 0.8
```

Because the log ratio is a *sum over tokens*, longer answers accumulate more penalty. That's why β, reward scale and response length have to be tuned together: double the reward scale without touching β and you've halved the leash.

| | Clipping | KL penalty |
|---|---|---|
| Compares policy to | π_old, the snapshot that made this batch | πref, the frozen SFT model |
| Timescale | Within one PPO update | The whole training run |
| Protects against | Overshooting on reused data | Drifting into reward-model blind spots |
| Hard limit? | No | No, it's a price, not a wall |

One more InstructGPT detail worth knowing: RLHF cost some performance on standard NLP benchmarks (the paper calls this an "alignment tax"), and they reduced it with **PPO-ptx**, which mixes ordinary pretraining gradients into the PPO updates.

> 🎬 **Animation — clip versus leash:** left panel: a plot of the PPO objective against ratio ρ for A = +2. It rises linearly from 0 until ρ = 1.2 (value 2.4), then goes flat; a dot at ρ = 1.4 sits on the flat part with a tag reading "2.8 unclipped → 2.4". Mirror it for A = −2 with the flat part below ρ = 0.8. Right panel: a 2-D "behaviour space" with a pinned star labelled πref (SFT). The policy dot takes many small steps (each boxed in by a small square = clipping) and wanders away; a rubber-band line from the star to the dot gets thicker with distance and shows "−β·KL: 1.0 → 0.8" at log ratio 2.

## DPO: skipping the judge and learning straight from the pairs {#dpo}

PPO-RLHF works, but it's a lot of machinery: a reward model, a value model, a reference model and an online generation loop, all of which have to stay stable together. **Direct Preference Optimization (DPO)** asks: if the preference pairs are all we really have, can we train the policy on them directly?

Analogy: instead of hiring a judge, training the judge on past rulings, and then coaching the athlete to impress the judge, you coach the athlete straight from the past rulings.

### The one piece of maths worth knowing

The KL-regularised objective J(θ) from the PPO section has a known best solution. For any reward function r, the policy that maximises "reward minus β·KL to the reference" is:

```formula
π*(y | x) = πref(y | x) · exp( r(x, y) / β ) / Z(x)
```

In words: start from the reference and re-weight each answer by exp(reward/β). Z(x) is a normaliser that makes the probabilities sum to 1. It sums over *every possible answer*, so you can't compute it.

A toy check with two possible answers, a reference that likes them equally (0.5/0.5), and rewards 1 and 0:

```text
β = 0.5:  weights e^2 = 7.39 vs e^0 = 1   →  π* = 0.881 / 0.119   (short leash? no: small β = loose leash)
β = 2  :  weights e^0.5 = 1.65 vs 1       →  π* = 0.622 / 0.378   (large β = stays near 0.5/0.5)
```

Now the trick. Rearrange that equation to express the reward in terms of the policy: r(x, y) = β·log[π*(y|x)/πref(y|x)] + β·log Z(x). Plug that into Bradley–Terry, which only ever uses the *difference* of two rewards for the same prompt, and the uncomputable β·log Z(x) cancels. What's left is a loss on the policy alone:

```formula
Δ = log[ πθ(y_w|x) / πref(y_w|x) ] − log[ πθ(y_l|x) / πref(y_l|x) ]
L_DPO = − log σ( β · Δ )
```

Here y_w and y_l are the chosen and rejected answers, πref is the frozen starting model (usually the SFT model), and β plays the same role as the KL coefficient. In plain English: **raise the chosen answer's probability relative to the reference by more than you raise the rejected one's.** The policy itself acts as an implicit reward model.

**Worked example (β = 0.2).**

```text
                   policy log p   reference log p   log ratio
chosen  y_w           −8              −10              +2
rejected y_l         −12              −11              −1
Δ = 2 − (−1) = 3
β·Δ = 0.6       σ(0.6) = 0.646       loss = −ln 0.646 = 0.437 nats

At the start (policy = reference): Δ = 0, σ(0) = 0.5, loss = ln 2 = 0.693 nats
```

These are illustrative numbers, not measurements. A useful sanity check on a real run: DPO loss starts at 0.693 on step one. If it doesn't, your reference and policy disagree before training even began, which usually means mismatched templates or masking.

### What DPO costs you

DPO is essentially a classification loss on stored pairs. It needs no sampling during training, no reward model and no value model, so it's cheap, stable and easy to reproduce. That's why it became the default preference step in many open recipes. But notice what it *gave up*:

- **It's offline.** It learns only from the pairs you already have. PPO generates fresh answers and gets them scored, so it can discover and correct its *own* new mistakes. DPO can't see mistakes that aren't in the data. (Iterated or "online" DPO variants regenerate pairs from the current policy to claw this back.)
- **The derivation assumes things.** It assumes Bradley–Terry is a decent model of your labels and that the policy can represent the optimum. With finite data, loss can fall while generations get worse. A known failure is both chosen and rejected log-probabilities falling together, which is still a positive margin, but the model is moving probability onto answers that appear in neither.
- **β isn't a KL dial.** In theory β sets the regularisation. In a finite run, don't assume a bigger β gives monotonically less drift. Measure the actual KL and read samples.
- **Bookkeeping matters.** Policy and reference log-probs must be computed over exactly the same response tokens with the same template and mask. Accidentally averaging per token instead of summing silently changes the objective and its length behaviour.

> 🎬 **Animation — one DPO pair:** two lanes, "chosen" and "rejected". Each lane has two gauges: frozen reference (grey) and trainable policy (blue). Step 1: all four gauges start equal, a Δ meter reads 0 and the loss reads 0.693. Step 2: the chosen lane's policy gauge moves from −10 to −8 (label +2); the rejected lane's moves from −11 to −12 (label −1). Step 3: the Δ meter reads 3, is multiplied by β = 0.2 to give 0.6, passes through a sigmoid to 0.646, and the loss drops to 0.437. Step 4: a greyed-out "reward model" and "value model" appear and get crossed out, captioned "no sampling, no RM, no critic".

## Verifiable rewards and reasoning models {#verifiable-rewards}

Human preferences are slow, expensive and biased. But for some tasks you don't need a human at all: you can *check* the answer with code. Did the final number match? Do the hidden unit tests pass? Does the output parse as valid JSON with the required fields? **RL with verifiable rewards (RLVR)** replaces the learned reward model with a programmatic checker. The term was named in the Tülu 3 paper, which gives a constant reward only when a completion is verified correct (and 0 otherwise) and keeps the usual RLHF objective, KL term included.

Note what RLVR is: a *reward source*, not an algorithm. PPO can consume it, and so can its cheaper cousins.

### DeepSeek-R1 and the reasoning-model recipe

The landmark example is **DeepSeek-R1**. Its precursor, R1-Zero, applied RL *directly to the base model with no SFT first*, using two rule-based rewards: an **accuracy reward** (the math answer matches, or the code passes a compiler/test check) and a **format reward** (put your thinking between `<think>` and `</think>` tags). The DeepSeek team said explicitly that they avoided neural reward models because those "may suffer from reward hacking" in large-scale RL. With nothing but pass/fail signals, the model learned to produce long chains of reasoning, check its own work and backtrack. Those are the traits of what we now call **reasoning models**: models trained to "think" at length in a scratchpad before answering, spending extra generated tokens (test-time compute) to be more accurate.

R1-Zero's output was hard to read and mixed languages, so the released R1 added stages back: a small "cold start" SFT set of long reasoning examples, reasoning-focused RL, then rejection sampling (generate many answers and keep the good ones) to build about 800k SFT samples, and a final RL stage across broader tasks. So even the "pure RL" story ends up as SFT + RL + SFT + RL. Interviewers like that nuance.

### GRPO: a baseline without a value model

R1 used **GRPO (Group Relative Policy Optimization)**. Its trick: drop the value model entirely, which in PPO is typically as big as the policy. Instead, sample a *group* of answers to the same prompt and use the group itself as the baseline. An answer's advantage is how much better it did than its siblings.

```formula
A_i = ( r_i − mean(r_1..r_G) ) / std(r_1..r_G)
```

Here r_i is the reward of answer i, and G is the group size.

**Worked example 1: four samples, rewards [1, 0, 1, 0].**

```text
mean = 0.5
centred      = [ +0.5, −0.5, +0.5, −0.5 ]
std (population) = 0.5
advantages   = [ +1,   −1,   +1,   −1  ]
```

**Worked example 2: a hard prompt, rewards [1, 0, 0, 0].**

```text
mean = 0.25
centred      = [ +0.75, −0.25, −0.25, −0.25 ]
std = √(0.1875) = 0.433
advantages   = [ +1.73, −0.58, −0.58, −0.58 ]
```

The rare success gets a big push. That's exactly what you want on hard problems.

**Worked example 3: every sample fails, [0, 0, 0, 0].** Every centred reward is 0, so there's no signal at all. Same for all-pass. Only prompts the model *sometimes* solves teach it anything. That turns task difficulty into a first-class engineering knob: filter or curriculum-order your prompts so the pass rate sits in the useful middle, and don't expect more updates on impossible prompts to create information that isn't there.

> 🎬 **Animation — group-relative advantages:** one prompt fans out into four answer cards. Step 1: a checker stamps them ✓ ✗ ✓ ✗ and rewards 1, 0, 1, 0 appear. Step 2: a horizontal "group mean" line drops in at 0.5. Step 3: bars grow up or down from the line (+0.5 / −0.5), then rescale to +1 / −1 as "÷ std 0.5" appears. Step 4: replay with ✓ ✗ ✗ ✗: the mean line sits at 0.25 and the lone success's bar shoots to +1.73. Step 5: replay with ✗ ✗ ✗ ✗: every bar is zero and the caption reads "no signal: prompt too hard".

### Outcome versus process rewards

A pass/fail checker only grades the **outcome**, the final answer. **Process supervision** grades each intermediate step, like a maths teacher who marks the working and not just the boxed answer.

```text
Solve 3(x − 2) = 12
 Trace A:  x − 2 = 4   →  x = 6     ✓ valid steps, ✓ answer
 Trace B:  x − 2 = 5   →  x = 6     ✗ step 1 wrong, ✓ answer (by luck or copying)
Outcome checker: A = 1, B = 1        Process checker: flags B's first step
```

OpenAI's "Let's Verify Step by Step" trained a process reward model on PRM800K (800,000 step-level human labels) and found process supervision beat outcome supervision on MATH problems (their best model solved 78% of a representative test subset). That's evidence for maths, not a universal law. Process labels need a clear definition of a "step", and a process checker can wrongly reject a valid shortcut. One more caveat for interviews: a readable reasoning trace that earned reward is **not** proof that the text faithfully shows how the model actually computed its answer.

## Reward hacking: when the score goes up and the model gets worse {#reward-hacking}

Every method in this article optimises a *proxy*: a learned reward model, a DPO dataset, a test suite. **Reward hacking** (also called reward over-optimisation) is when the optimiser finds a way to raise the proxy that doesn't raise, or actively lowers, the thing you actually care about. It's Goodhart's law ("when a measure becomes a target, it ceases to be a good measure") with a gradient attached, or teaching to the test at industrial scale.

Gao et al. measured this cleanly. They used a large "gold" reward model to stand in for humans, trained smaller proxy reward models on its labels, and then optimised policies against the proxies. As optimisation pushed further from the starting policy, the proxy score kept climbing, while the gold score rose, peaked and then *fell*. The shape of that curve depends on the optimisation method (RL versus best-of-n sampling), and bigger reward models delayed the fall.

```text
score
  │                         ..... proxy reward (keeps rising)
  │                   .....
  │             .....      ___
  │        ....       ___--   --__      gold / true quality
  │    ...      __---             --__  (peaks, then falls)
  │ ..    __--                        --
  │.__--
  └─────────────────────────────────────────►  how far the policy has moved (KL from start)
            ▲ stop around here
```

What hacking looks like in practice:

| Proxy | Typical hack |
|---|---|
| Preference reward model | Longer answers, more bullet points, confident tone, flattery ("Great question!"), agreeing with the user |
| Unit tests | Special-casing the visible tests, hard-coding outputs, catching every exception, editing the test file if it's writable |
| Answer parser | Printing several candidate answers so the right one appears somewhere |
| Format reward | Perfect `<think>` tags around nonsense |
| LLM judge | Phrasing that the judge model likes regardless of content |

Controls, roughly in order of value:

1. **Keep the KL leash** and watch it. A sudden KL jump is often the first sign of an exploit.
2. **Harden verifiers** like security boundaries: run code in a sandbox, keep tests hidden, make the test files read-only, and feed the checker adversarial "cheating" solutions to check it rejects them. (Sandboxing is covered in **LLM security: treating model output as untrusted**.) Version the checker's semantics like an API.
3. **Log the raw task score separately** from format or length bonuses, so a side reward can't quietly replace the real one.
4. **Track response length** and other style statistics every checkpoint. Length creep is the most common preference hack.
5. **Refresh the reward model** with labels on *current* policy outputs, and read samples by hand. Humans spot a hack in ten examples that a dashboard misses for a week.
6. **Evaluate on something the optimiser never saw** (next section).

> 🎬 **Animation — Goodhart curve:** x-axis "KL from starting policy", y-axis "score". Two lines draw in from the left together: blue "proxy reward" and green "true quality". Step 1: both rise together. Step 2: past a marked point the blue line keeps rising while green bends over and falls; the gap between them is shaded red, labelled "reward hacking". Step 3: little sample cards pop up along the blue line as it rises: first a normal answer, then a longer one with bullet points, then a very long, flattering one with a wrong fact highlighted. Step 4: a vertical "early stop / promotion gate" marker snaps to the green peak.

## Evaluating the pipeline and choosing between methods {#evaluate-and-choose}

Here's the discipline that separates a senior answer from a list of acronyms: **the training signal and the ship/no-ship test must be different things.** If you pick checkpoints with the same reward model you trained against, you're measuring how well you hacked it.

A sound **promotion gate** (the checks a checkpoint must pass before it ships) combines:

- **Deterministic task checks** on held-out tasks: hidden tests, exact answers, schema validity for tool calls.
- **Blinded human comparisons** against the current production model, with position randomised, reporting win rate *with a confidence interval*, not a bare number.
- **Behaviour slices**: refusal rate on benign-but-edgy prompts, harmful-content compliance, calibration (does "I'm not sure" line up with actually being wrong?), and multi-turn and tool-use regressions.
- **Capability regressions**: the "alignment tax" check on general benchmarks.
- **Drift stats**: KL from the reference, mean and tail response length.

Freeze the eval prompts and scoring rules *before* you look at checkpoints, keep an untouched audit set for final claims, and roll back when independent outcomes regress, even if training reward is still climbing. (LLM-as-judge calibration, contamination and the statistics of paired comparisons are covered in **Evaluation and observability: knowing whether it actually works**.)

### Picking a method

| Situation | Reach for | Why |
|---|---|---|
| Model doesn't follow the format, lacks a procedure, wrong tone | SFT | Demonstrations directly encode it |
| You have good preference pairs covering your tasks, limited infra | DPO | Cheap, stable, no rollouts |
| Pairs are stale or the policy keeps inventing new failure modes | Online RL (PPO + RM) or online/iterated DPO | Learns from fresh samples of its *own* behaviour |
| Correctness is machine-checkable (maths, code, schemas) | RLVR (PPO or GRPO) | Scalable, hard-to-bias signal, provided the checker is hardened |
| Nothing cheap tells right from wrong | Invest in evaluation and data first | No algorithm fixes a missing signal |

Back-of-envelope on cost. For a 7B-parameter policy in bf16 (2 bytes per parameter), each copy of the weights is about 14 GB. PPO-RLHF keeps roughly four models around (policy, reference, reward, value), which is about 56 GB of weights *before* the policy's optimizer state. With Adam, full training costs roughly 16 bytes per trainable parameter, so about 112 GB for the policy alone and about as much again for a trainable value model. It also spends most of its wall-clock time *generating* rollouts. DPO needs the policy plus a frozen reference and no generation at all. GRPO drops the value model but generates G samples per prompt. (Memory accounting per parameter is worked through in **Distributed training: fitting a training run onto a cluster**.) These are rough, illustrative sizings; real systems shard, offload or precompute reference log-probs.

Treat every row of that table as a hypothesis. The defensible move is a matched-budget comparison: same prompts, same eval gate, SFT baseline versus DPO versus RL, and promote whatever wins on *independent* outcomes.

# Interview

## Question

You own a coding assistant. SFT has fixed its output format, but users still get plausible-looking fixes that don't work. You have 30,000 preference pairs from users, and a test runner that can check about half of incoming requests. What post-training do you do next, and how do you know it worked?

## Answer

I'd start with measurement, not an algorithm. Build a held-out evaluation split by repository and by time, so near-duplicate code can't leak from training, and measure functional success (hidden tests pass) separately from user preference and format. That tells me how big the "plausible but wrong" gap really is.

Next I'd audit the 30k pairs. User preferences in coding often reward confident, well-explained answers. If the chosen answer fails tests more often than it should, the pairs are teaching persuasiveness, and training on them would make the problem *worse*. I'd filter or relabel pairs using the test runner where possible, and add pairs where a terse correct fix beats a verbose broken one.

Then two candidates. First, DPO on the cleaned pairs from the SFT checkpoint: it's cheap, stable and needs no rollouts, and it's a strong baseline. Second, for the testable half, RLVR: sample several fixes per task, reward 1 when hidden tests pass, and use group-relative advantages (GRPO) or PPO with a KL penalty to the SFT model. Before that run, I'd harden the runner: sandboxed execution, hidden and read-only tests, and a red-team set of hard-coded and test-tampering solutions that must score zero. I'd pick tasks the model solves sometimes but not always, since all-fail and all-pass groups carry no signal.

For the untestable half I'd rely on blinded human evaluation and wouldn't treat "no tests" as either pass or fail. Both candidates get compared to SFT at the same generation budget, tracking pass rate, KL, response length and regressions in tool use and uncertainty. I promote only if independent task success improves. I'd consider online PPO with a learned reward model only if fresh on-policy data clearly beats DPO and the reward model holds up on current outputs, since it's the most expensive and fragile option.

## Follow-ups

- How would you detect length bias in the preference pairs before training?
- What happens to GRPO's signal if every sampled fix fails its tests, and what would you change?
- Why is PPO's rollout policy (π_old) different from the KL reference (πref)?
- The pass rate jumped 15 points overnight in RLVR training. How do you tell real improvement from verifier exploitation?
- Your DPO loss is falling but generations are getting longer and worse. What's going on?

# Pitfalls

- Treating a reward model score as a calibrated probability of correctness. Bradley–Terry rewards are only defined up to a per-prompt constant; only differences mean anything.
- Confusing PPO clipping with the KL penalty. Clipping limits each update relative to the rollout snapshot π_old; the KL term anchors the whole run to the frozen reference πref. Neither is a hard limit.
- Assuming lower DPO or reward-model loss means better generations. Offline losses can fall while online behaviour drifts, grows longer or hacks a bias in the pairs.
- Giving full credit for passing an incomplete or exploitable test suite. The checker becomes the thing being optimised, so it needs hidden tests, sandboxing and adversarial validation.
- Using the same reward model, judge or prompts for training, checkpoint selection and final claims. That measures how well you fit the proxy, not quality.
- Comparing losses or KL across runs without stating token masks, sum-vs-mean reduction and reward scale.
- Forgetting the chat template or the end-of-turn token in SFT data. The model never learns when to stop, or is served with tokens it never saw.
- Assuming a correct final answer means the written reasoning was valid, or that a rewarded reasoning trace faithfully shows how the model computed it.

# Checklist

- Explain, with an example, why a base model continues a question instead of answering it.
- Write out a chat-templated training example and mark which tokens the SFT loss mask grades.
- Compute a Bradley–Terry loss, a clipped PPO term, a KL-penalised reward and a DPO loss by hand.
- Sketch the DPO derivation: the KL-optimal policy, solving for the reward, and why Z(x) cancels.
- Compute group-relative advantages and explain why all-fail groups give no signal.
- List three reward hacks and the control that catches each.
- Design a promotion gate that doesn't reuse the training signal.
- Choose between SFT, DPO, PPO-RLHF and RLVR for a given data situation, and defend it.

# Sources

- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — The InstructGPT SFT → reward model → PPO pipeline, dataset sizes, K = 4–9 rankings, per-token KL penalty from the SFT model, PPO-ptx and the alignment tax, and the 1.3B vs 175B preference result.
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — The clipped surrogate objective and reusing sampled data for multiple epochs of minibatch updates.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — Derivation of the DPO loss from the KL-regularised objective and Bradley–Terry, trained without a reward model or sampling.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/html/2501.12948v1) — R1-Zero's RL on a base model with rule-based accuracy and format rewards, GRPO's group baseline without a critic, avoiding neural reward models over reward hacking concerns, and the multi-stage R1 pipeline.
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124) — Names RLVR, which gives a reward only for verified-correct completions, and an open SFT → DPO → RLVR recipe.
- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) — Process vs outcome supervision on MATH, the PRM800K dataset of 800k step labels, and 78% on a representative MATH test subset.
- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2022)](https://arxiv.org/abs/2210.10760) — Gold-vs-proxy reward setup showing proxy reward rising while gold reward falls, with different functional forms for RL and best-of-n.
- [Chat templates (Hugging Face Transformers documentation)](https://huggingface.co/docs/transformers/main/en/chat_templating) — How messages become token sequences with control tokens, add_generation_prompt, and why using the wrong template or duplicated special tokens hurts quality.

# Flashcards

## base-vs-assistant

**Q:** Why does a base model often answer a question with more questions?

It was trained only to continue text the way its training data would. On the web, a question is often followed by more questions (quiz pages, forum lists), so that's a likely continuation. Post-training (SFT, then preference or RL stages) teaches the specific "reply, then stop" behaviour of an assistant.

## chat-template-mismatch

**Q:** What is a chat template, and what happens if you serve a model with the wrong one?

A chat template turns a list of role/content messages into a single token sequence with model-specific control tokens (e.g. `<|im_start|>assistant`). The model learned those exact tokens in post-training. With the wrong template, or duplicated BOS/EOS tokens, it sees unfamiliar patterns and quality drops sharply, with no error message.

## sft-objective

**Q:** What does SFT optimise, and what can't it teach?

The next-token log-likelihood of demonstration tokens, usually with a loss mask so only assistant tokens (including the end-of-turn token) are graded while the prompt stays as context. It only ever sees good answers, so it gets no explicit signal that a confident wrong answer is worse than an alternative.

## teacher-forcing

**Q:** Why can teacher forcing hide generation failures?

During training each position is conditioned on the correct previous tokens from the demonstration. At inference the model conditions on its own outputs, so one early mistake can create a context no demonstration covered, and errors compound. RL stages train on the model's own samples, which partly addresses this.

## reward-meaning

**Q:** Does a reward model score of 0.9 mean the answer is 90% correct?

No. Under Bradley–Terry only reward differences between answers to the same prompt affect the predicted preference, so adding any constant changes nothing. The absolute value has no meaning, and the model learns whatever predicts labeller choices, including biases such as length.

## bt-calculation

**Q:** A reward model gives the winner 1.2 and the loser 0.2. What are the preference probability and loss?

The gap is 1.0, so P = σ(1) ≈ 0.731 and the loss is −ln 0.731 ≈ 0.313 nats. If the label were reversed the loss would be −ln σ(−1) ≈ 1.313 nats.

## ppo-old-ref

**Q:** Why does PPO-RLHF have both an "old" policy and a reference policy?

π_old is the snapshot that generated the current batch of rollouts; the PPO ratio and clipping are measured against it so reusing that batch for several updates stays safe. πref is the frozen SFT model that the KL penalty anchors to over the whole run. They're different checkpoints doing different jobs.

## ppo-clipping

**Q:** With A = +2, ratio 1.4 and ε = 0.2, what is the PPO clipped term, and is clipping a hard KL limit?

min(1.4×2, 1.2×2) = min(2.8, 2.4) = 2.4, so pushing the ratio past 1.2 earns nothing more. It isn't a hard limit: it only removes the incentive on sampled tokens, and other tokens and shared weights can still move the distribution substantially.

## kl-samples

**Q:** Can a single sample's policy/reference log ratio be negative?

Yes, when that response is likelier under the reference than under the policy. The KL divergence is the expectation of this log ratio under the policy and is never negative, but individual samples can be. Example: log p −12 vs −14 gives +2 nats; swap them and it's −2.

## kl-scale

**Q:** Why must reward scale, response length and β be tuned together?

The objective is reward − β·KL, and the sequence log ratio sums over tokens. Doubling the reward scale without changing β halves the leash's relative strength, and longer responses accumulate more KL penalty. Example: reward 1.0, log ratio 2 nats, β = 0.1 gives 0.8.

## dpo-mechanism

**Q:** How does DPO avoid training a separate reward model?

The KL-regularised objective's optimal policy is πref·exp(r/β)/Z(x). Solving for r expresses reward as β·log(π/πref) plus β·log Z(x). Bradley–Terry uses only reward differences for the same prompt, so Z cancels, which leaves the loss −log σ(β·Δ), where Δ is the chosen log ratio minus the rejected log ratio. The policy acts as its own implicit reward model.

## dpo-coverage

**Q:** What is the main limitation of basic offline DPO compared with online RL?

It learns only from stored pairs, so it can't discover or correct new failure modes the current policy produces. Coverage gaps and biases in the pairs persist even as the loss falls. Online RL, or iterated DPO with fresh pairs, samples the policy's own outputs.

## rlvr-definition

**Q:** Is RLVR a specific optimisation algorithm?

No. It describes the reward source: a programmatic checker (exact answer match, hidden unit tests, schema validation) instead of a learned reward model. PPO, GRPO and other policy-gradient methods can all consume it, usually still with a KL penalty to a reference.

## group-zero

**Q:** With group-relative advantages, what do rewards [1,0,0,0] and [0,0,0,0] give?

[1,0,0,0]: mean 0.25, std ≈ 0.433, so advantages are ≈ [+1.73, −0.58, −0.58, −0.58], a big push toward the rare success. [0,0,0,0]: all centred rewards are zero, so there's no learning signal. Tune prompt difficulty so the model sometimes succeeds.

## process-outcome

**Q:** What does process supervision add over outcome supervision, and what doesn't it guarantee?

It grades intermediate steps, so it can catch a wrong step that still reaches the right final answer and gives denser feedback about where an error happened. It needs a clear definition of a step and reliable step labels, can reject valid shortcuts, and doesn't prove the written reasoning reflects the model's actual computation.

## reward-hacking

**Q:** What evidence distinguishes real progress from reward hacking?

Independent outcomes the optimiser never saw must improve: hidden tests, blinded human comparisons, frozen eval sets. A rising proxy score alone (reward model, parser, visible tests) can come from exploits such as length inflation or special-casing tests. Watch KL spikes and length creep, and read samples.
