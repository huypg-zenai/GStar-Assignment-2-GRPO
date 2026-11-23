# GRPO for Countdown Math Problems

This project implements the **Group Relative Policy Optimization (GRPO)** algorithm to fine-tune a language model on the "Countdown" math task. The goal is to train an LLM to generate correct mathematical equations that reach a target number using a given set of integers.

## Report and results

**[report.pdf](./report.pdf)** covers both problems: implementation, conceptual answers, experiments and analysis
(source and analysis scripts in [report/](./report/)).

| Final test accuracy (1,024 prompts) | L256 | L512 | L1024 |
|---|---|---|---|
| Qwen3-1.7B zero-shot | 0.0% | 9.0% | 8.7% |
| GRPO (80 steps) | 58.5% | 52.1% | – |
| Dr.GRPO (80 steps) | **60.7%** | **55.9%** | – |

Problem 2 (extra credit) adapts the pipeline to Leduc Hold'em poker decisions graded by an exact CFR+ solution;
see [problem2/README.md](./problem2/README.md) and Section 4 of the report.

## Setup

```
pip install --upgrade uv
uv venv
source .venv/bin/activate
uv pip install vllm==0.7.2 triton==3.1.0 datasets transformers==4.51.3 tensorboard torch gpustat datasets python-dotenv
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Assignment Structure

Instruction: [gstar_assignment2.pdf](./gstar_assignment2.pdf)

Main File: `starter.py`:

### Problem 1
#### Part 1: GRPO Pipeline Implementation
-   **Task 1**: Implement reward helper functions (`_extract_answer`, `_validate_numbers`, `_evaluate_equation`).
-   **Task 2**: Implement the main `reward_fn`.
-   **Task 3**: Implement `compute_group_normalized_advantages`.
-   **Task 5**: Implement `masked mean` for GRPO and `masked_mean_drgrpo` for DR.GRPO.

#### Part 2: Main Experiments and Report (please refer to section 1.6 of [gstar_assignment2.pdf](./gstar_assignment2.pdf) for the Main Report and Experiments

### Problem 2: (Optional) Open-ended Investigation. Please refer to Section 2 of [gstar_assignment2.pdf](./gstar_assignment2.pdf)

## Running Problem 1 on Kaggle (2× T4)

`starter.py` holds the Problem-1 implementations (Tasks 1–5); `tests/test_tasks.py` checks them on CPU
(`python -m pytest -q tests/test_tasks.py`).

`kaggle/problem1_kaggle.ipynb` clones this repo and runs `kaggle/grpo_t4.py`, which uses those implementations
unchanged. A T4 can't hold the starter's single-A100 setup, so the runtime differs:
- vLLM rollouts run on GPU 1 in fp16;
- the policy (Qwen3-1.7B with LoRA adapters on a frozen fp16 base) trains on GPU 0;
- the merged weights are pushed into vLLM before each rollout and each eval.

The other hyperparameters follow `starter.main()`. Run the notebook once with `RUN = "L256"` (which also runs
zero-shot at L256/L512/L1024) and once with `RUN = "L512"`. Then build the combined table and plots:
`python kaggle/report.py outputs_L256/ outputs_L512/ --save results/`.
