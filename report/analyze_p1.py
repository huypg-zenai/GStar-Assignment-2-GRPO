"""Failure-mode breakdown of the Problem 1 evaluations (zero-shot and final GRPO / Dr.GRPO checkpoints).

  python report/analyze_p1.py ~/Downloads --out report/p1_analysis.json

Each response is put in exactly one bucket, using the same verifier functions as the reward (starter.py):
  correct            reward 1.0
  wrong_value        <answer> found, numbers valid, evaluates, but result != target
  bad_numbers        <answer> found, uses a number that is not available or uses one too often
  bad_expression     <answer> found, numbers valid, but the expression does not evaluate (syntax, '=', unicode...)
  no_tag_truncated   no <answer>, generation hit the max-token limit
  no_tag_finished    no <answer>, generation ended on its own (e.g. \\boxed{}, plain-text answer)
"""
import argparse
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_tasks  # noqa: E402,F401  (stubs vllm / tensorboard so starter imports on a laptop)
import starter as S  # noqa: E402

RUNS = {  # name -> (path relative to the downloads dir, max tokens)
    "zero-shot L256": ("outputs_L256/zeroshot/zeroshot_L256_examples.jsonl", 256),
    "zero-shot L512": ("outputs_L256/zeroshot/zeroshot_L512_examples.jsonl", 512),
    "zero-shot L1024": ("outputs_L256/zeroshot/zeroshot_L1024_examples.jsonl", 1024),
    "GRPO L256": ("outputs_L256/train_L256_grpo/eval_step080.jsonl", 256),
    "Dr.GRPO L256": ("outputs_L256_dr/train_L256_dr_grpo/eval_step080.jsonl", 256),
    "GRPO L512": ("outputs_L512/train_L512_grpo/eval_step080.jsonl", 512),
    "Dr.GRPO L512": ("outputs_L512_dr/train_L512_dr_grpo/eval_step080.jsonl", 512),
}
BUCKETS = ["correct", "wrong_value", "bad_numbers", "bad_expression", "no_tag_truncated", "no_tag_finished"]


def bucket(r, max_tokens):
    text, gt = r["response"], r["answer"]
    eq = S._extract_answer(text)
    if eq is None:
        return "no_tag_truncated" if r["output_tokens"] >= max_tokens else "no_tag_finished"
    if not S._validate_numbers(eq, gt["numbers"]):
        return "bad_numbers"
    val = S._evaluate_equation(eq)
    if val is None:
        return "bad_expression"
    return "correct" if abs(val - gt["target"]) < 1e-6 else "wrong_value"


def think_len(text):
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return None if m is None else len(m.group(1).strip())


def numbers_used(eq):
    return len(re.findall(r"\d+", eq))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("downloads")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {}
    for name, (rel, L) in RUNS.items():
        path = os.path.join(os.path.expanduser(a.downloads), rel)
        if not os.path.exists(path):
            print("missing", path)
            continue
        rows = [json.loads(x) for x in open(path) if x.strip()]
        b = [bucket(r, L) for r in rows]
        n = len(rows)
        m = {"n": n, **{k: b.count(k) for k in BUCKETS}}
        assert m["correct"] == sum(r["reward"] == 1.0 for r in rows), name   # same verdict as the logged reward
        for k_num in (3, 4):
            idx = [i for i, r in enumerate(rows) if len(r["answer"]["numbers"]) == k_num]
            m[f"acc_{k_num}nums"] = sum(b[i] == "correct" for i in idx) / max(1, len(idx))
            m[f"n_{k_num}nums"] = len(idx)
        corr = [r for r, x in zip(rows, b) if x == "correct"]
        m["correct_using_subset"] = sum(numbers_used(S._extract_answer(r["response"])) < len(r["answer"]["numbers"]) for r in corr)
        tl = [think_len(r["response"]) for r in rows]
        m["empty_think"] = sum(t == 0 for t in tl)
        m["has_think_block"] = sum(t is not None for t in tl)
        m["avg_tokens"] = statistics.mean(r["output_tokens"] for r in rows)
        m["avg_tokens_correct"] = statistics.mean(r["output_tokens"] for r in corr) if corr else None
        res[name] = m
        print(f"{name:16s} " + " ".join(f"{k}={m[k]}" for k in BUCKETS) +
              f" | acc3={m['acc_3nums']:.3f} acc4={m['acc_4nums']:.3f} subset_correct={m['correct_using_subset']}"
              f" empty_think={m['empty_think']} think_blocks={m['has_think_block']} tok={m['avg_tokens']:.0f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
