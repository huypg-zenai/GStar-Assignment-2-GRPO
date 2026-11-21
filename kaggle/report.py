"""Tables and plots for Problem 1 (Section 1.6) from one or more output folders.

  python kaggle/report.py OUT_DIR [OUT_DIR2 ...] --save REPORT_DIR

Each OUT_DIR may contain zeroshot/zeroshot.json and/or train_L*/ (metrics.jsonl, summary.json). Run it on each
Kaggle session's output, or on both folders together after downloading them, to get the combined table.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(dirs):
    zs, runs = {}, {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "zeroshot.json"), recursive=True):
            zs.update(json.load(open(f)))
        for f in glob.glob(os.path.join(d, "**", "metrics.jsonl"), recursive=True):
            run = os.path.basename(os.path.dirname(f))
            recs = [json.loads(line) for line in open(f) if line.strip()]
            summ = os.path.join(os.path.dirname(f), "summary.json")
            runs[run] = dict(train=[r for r in recs if r["type"] == "train"],
                             eval=[r for r in recs if r["type"] == "eval"],
                             summary=json.load(open(summ)) if os.path.exists(summ) else None)
    return zs, runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--save", required=True)
    a = p.parse_args()
    os.makedirs(a.save, exist_ok=True)
    zs, runs = load(a.dirs)

    lines = ["| Model | Max tokens | Accuracy (%) | Mean reward | Correct / partial / failed | Avg output tokens |",
             "|---|---|---|---|---|---|"]
    for k in sorted(zs, key=lambda s: int(s[1:])):
        z = zs[k]
        lines.append(f"| Qwen3-1.7B zero-shot | {k} | {z['accuracy']:.2f} | {z['mean_reward']:.3f} | "
                     f"{z['count_correct']} / {z['count_partial']} / {z['count_failed']} | {z['avg_output_tokens']:.0f} |")
    for name, r in sorted(runs.items()):
        if not r["eval"]:
            continue
        last, first = r["eval"][-1], r["eval"][0]
        best = max(r["eval"], key=lambda e: e["accuracy"])
        lines.append(f"| {name} (step {last['step']}) | L{name.split('_L')[-1].split('_')[0]} | {last['accuracy']:.2f} | "
                     f"{last['mean_reward']:.3f} | {last['count_correct']} / {last['count_partial']} / {last['count_failed']} | "
                     f"{last['avg_output_tokens']:.0f} |")
        lines.append(f"|   (step 0 = zero-shot in the same run: {first['accuracy']:.2f}%; best eval {best['accuracy']:.2f}% "
                     f"at step {best['step']}) | | | | | |")
    table = "\n".join(lines)
    open(os.path.join(a.save, "results.md"), "w").write(table + "\n")
    print(table)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    colors = {}
    for name, r in sorted(runs.items()):
        if r["eval"]:
            ln, = ax[0].plot([e["step"] for e in r["eval"]], [e["accuracy"] for e in r["eval"]], "o-", label=name)
            colors[name] = ln.get_color()
        if r["train"]:
            c = colors.get(name)
            ax[1].plot([t["step"] for t in r["train"]], [t["reward_mean"] for t in r["train"]], color=c, label=name)
            ax[2].plot([t["step"] for t in r["train"]], [t["avg_output_tokens"] for t in r["train"]], color=c, label=name)
    for k, z in zs.items():
        ax[0].axhline(z["accuracy"], ls="--", lw=1, color="gray")
        ax[0].text(0, z["accuracy"] + 0.5, f"zero-shot {k}", fontsize=8, color="gray")
    ax[0].set(title="Eval accuracy (1,024 test prompts)", xlabel="GRPO step", ylabel="accuracy (%)")
    ax[1].set(title="Train reward (mean over rollouts)", xlabel="GRPO step", ylabel="reward")
    ax[2].set(title="Avg output tokens (train rollouts)", xlabel="GRPO step", ylabel="tokens")
    for x in ax:
        x.grid(alpha=0.3)
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.save, "problem1_curves.png"), dpi=140)

    if zs or runs:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        names, vals = [], []
        for k in sorted(zs, key=lambda s: int(s[1:])):
            names.append(f"zero-shot\n{k}"); vals.append(zs[k]["accuracy"])
        for name, r in sorted(runs.items()):
            if r["eval"]:
                names.append(name.replace("train_", "GRPO\n")); vals.append(r["eval"][-1]["accuracy"])
        bars = ax.bar(names, vals, color=["#9aa5b1"] * len(zs) + ["#2b6cb0"] * (len(names) - len(zs)))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
        ax.set(ylabel="accuracy (%)", title="Final task accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(a.save, "problem1_final_accuracy.png"), dpi=140)
    print("saved to", a.save)


if __name__ == "__main__":
    main()
