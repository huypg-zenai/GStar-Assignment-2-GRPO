# Problem 2: Teaching a small LLM to reason about poker with GRPO

**Question.** Countdown has one right answer per prompt. Many real decisions don't: what is right depends on
hidden information and on what other agents do. Can RL with verifiable rewards (GRPO, Problem 1's pipeline) teach
Qwen3-1.7B to reason in such a setting? And how much does the *design of the reward* matter when the environment's
own feedback is noisy?

**Testbed: Leduc Hold'em.** Leduc is a small poker game, standard in game-theory research:
- a 6-card deck (J J Q Q K K);
- one private card per player and one public card;
- two betting rounds.

It still has the features of real poker: hidden information, bluffing, value betting and pot odds. It is also small
enough (288 decision points) that we can:
- solve it exactly with CFR+, giving a perfect grader for every decision;
- compute the exact **exploitability** of any strategy, a gold-standard measure of play strength in game theory.

The engine and solver (`leduc.py`) reproduce OpenSpiel's `leduc_poker` exactly. Exploitability of the uniform,
always-call and always-raise policies matches OpenSpiel's to 6 decimals, and the equilibrium has exploitability
6·10⁻⁶ chips/hand and value −0.0856 for player 1 (the known value of the game).

## Problem formulation

- **One prompt = one decision.** The prompt states the rules, the private card, the public card (if revealed),
  the betting so far, the pot, and the legal actions, all in plain English (`leduc.py: user_prompt`).
- **Output.** The model reasons briefly in plain text, then answers with one legal action word in `<answer>`.
  Qwen3's built-in thinking mode is turned off (`enable_thinking=False`; see the pilot below).
- **Opponent.** The prompt says the opponent plays optimally. All rewards assume the opponent follows the CFR+
  equilibrium strategy σ\*.

This framing changes as little as possible from Problem 1: single-turn prompts, a verifiable reward, and the
same GRPO code (`starter.py`) and GPU setup (`kaggle/grpo_t4.py`). Only the task changes.

## Pilot: the prompt has to give GRPO a signal

The first full run used Qwen3's thinking mode and asked for reasoning in `<think>` tags. The model wrote long
analyses that often looped ("The user's action. The user's action. …"). At step 0 every rollout hit the
512-token cap, so every reward was 0, every group had identical rewards, and the gradient was exactly 0. We
stopped it after 11 steps and compared prompts:

| Prompt (Qwen3-1.7B, fp16, temperature 1) | Valid answers | Hit the cap | Avg tokens | Groups with reward signal |
|---|---|---|---|---|
| Thinking mode, max 512 (Kaggle, 288 × 8) | 0.04% | 99.96% | 512 | – |
| Thinking mode, max 1,024 (Kaggle, 288 × 8) | 0.3% | 76% | 858 | – |
| Thinking off, same instruction (16 × 4) | 45% | 53% | 326 | 15/16 |
| **Thinking off + "think briefly (a few sentences)"** (16 × 4) | **94%** | **0%** | **146** | **11/16** |

With the last prompt, every answer reasons before acting, but only 32% of valid answers pick the best action.
That leaves plenty of room for RL. It is the prompt used for all reported runs. `train` now stops if the step-0
valid rate is under 5% (`--min_valid`), so a run with no learning signal cannot use up a session.

## Hypotheses (written before running)

- **H1 – RL learns strategy, not just format.** GRPO with the exact reward lowers exploitability and regret well
  below zero-shot Qwen3-1.7B, and the gains carry over to **held-out situations** never seen in training.
- **H2 – Reward noise matters.** A realistic reward (play the hand out once, score the chips won) is an unbiased
  but noisy estimate of the exact reward (std ≈ 4 chips against action-value gaps of 1–3 chips). With the same
  compute, it learns more slowly and reaches higher exploitability than the exact reward.
- **H3 – Exploitative collapse.** Both rewards score a single decision against a *fixed* opponent. Any best
  response to σ\* gets full reward, and where σ\* mixes (bluffs), several actions score the same. So we expect the
  trained policy to become nearly deterministic (low entropy where σ\* mixes). High reward would then not mean
  equilibrium play, and the policy would stay partly exploitable. This is the "reward ≠ objective" gap that RL
  practitioners face.

## Training data

- **Situations.** Information sets are sampled 50% from equilibrium self-play, matching how often situations
  occur in real play, and 50% uniformly over all info sets, for coverage of rare and off-path situations
  (`StateSampler`).
- **Held-out split.** Round-2 situations with the (private, public) pairs (J,K), (Q,J) and (K,Q) are never
  shown in training. That is 90 of the 288 info sets, used to test generalization.
- **No dataset to download.** Situations are generated from the game, so the stream never runs out.

## Reward functions

In both cases, no `<answer>` tag or an illegal action gets reward 0, and a legal action gets 0.1 + 0.9·score,
mirroring Problem 1's format credit.

| Reward | Score for action *a* at info set *I* | Noise |
|---|---|---|
| `ev` (exact) | (Q\*(I,a) − min Q\*) / (max Q\* − min Q\*). Q\* is the expected chips of *a* under σ\*, with the opponent's hidden card weighted by its equilibrium belief. | none |
| `outcome` (realistic) | (chips won + 13) / 26 from **one** simulated hand: deal the hidden card from the belief, play *a*, then both players follow σ\* to the end. 13 is the largest possible pot share. | high |

The `outcome` reward is an unbiased Monte Carlo estimate of the `ev` reward, so the two arms differ only in
variance. That makes H2 a clean comparison.

## Evaluation design

At every eval, the model answers **k times at all 288 info sets**: k=2 during training and k=8 at the end and
for zero-shot. The answer frequencies define a stochastic policy, and invalid answers become a passive forfeit
(fold if facing a bet, otherwise check). The policy is then scored **exactly**, with no sampled games:

| Metric | What it measures |
|---|---|
| **Exploitability** (chips/hand) | How much a perfect adversary wins against the policy. 0 = equilibrium. Main metric. |
| Chips/hand vs equilibrium, random, always-call, always-raise | Practical strength against fixed opponents, averaged over both seats. |
| EV reward and regret (chips) per decision, **seen vs held-out** | Decision quality and generalization (H1). |
| Entropy at info sets where σ\* mixes | Whether bluffing / mixing survives training (H3). |
| Valid-answer rate, output tokens, truncation | Format learning and reasoning length. |
| Reasoning traces | Qualitative check: does the model reason about hand strength, the opponent's range and pot odds? |

**Baselines.** The CFR+ equilibrium (lower bound, exploitability 0); uniform random (2.37); always check/call
(1.47); always bet/raise (2.37); zero-shot Qwen3-1.7B with the final prompt (512 tokens). The thinking-mode pilot
is also a baseline. With almost no valid answers, its policy is "always check/fold" (exploitability 1.0).

## Compute (Kaggle 2× T4, same setup as Problem 1)

Hyperparameters match Problem 1: LoRA r=32, lr 5e-5, 80 steps × 128 rollouts (16 situations × 8), max 512
tokens, GRPO loss.

| Session | Contents | Time |
|---|---|---|
| A | bots + zero-shot (L512) + GRPO with `ev` reward | ~3–4 h |
| B | GRPO with `outcome` reward | ~3 h |

## Other formulations this setup supports

The same code makes it easy to vary the problem:
- **Exploitation instead of equilibrium:** score Q against a weak opponent (e.g. always-call) to test whether the
  model learns to exploit a known style.
- **Full-hand, multi-turn RL:** the model plays whole hands against σ\* (or itself) and is rewarded once at the end,
  so credit assignment spans several decisions.
- **Self-play:** compute rewards against the model's own current policy, which in principle pushes towards
  equilibrium and directly targets H3.
- **Reward shaping for H3:** e.g. reward matching σ\*'s action *distribution* within a group.
- **Transfer:** train on Leduc, test on Kuhn poker or a larger deck to see whether the reasoning is general.

## Files

| File | Purpose |
|---|---|
| `leduc.py` | Rules, CFR+ solver, exact action values, best response / exploitability, prompts, parsing, rewards, sampler |
| `leduc_equilibrium.json` | CFR+ solution (20k iterations) with Q-values for all 288 info sets |
| `grpo_leduc.py` | `zeroshot` and `train` modes, reusing `kaggle/grpo_t4.py` and the `starter.py` GRPO functions |
| `report_leduc.py` | Results table and plots |
| `../tests/test_leduc.py` | CPU tests, including the OpenSpiel reference values |
| `../kaggle/problem2_leduc_kaggle.ipynb` | Kaggle notebook (2× T4) |
