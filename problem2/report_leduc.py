"""Tables and plots for Problem 2 (Leduc Hold'em) from one or more output folders.

  python problem2/report_leduc.py OUT_DIR [OUT_DIR2 ...] --save REPORT_DIR
"""
import argparse
import glob
import json
import os

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt  # noqa: E402


def load(dirs):
    base, zs, runs = None, {}, {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "baselines.json"), recursive=True):
            base = json.load(open(f))
        for f in glob.glob(os.path.join(d, "**", "zeroshot.json"), recursive=True):
            zs.update(json.load(open(f)))
        for f in glob.glob(os.path.join(d, "**", "metrics.jsonl"), recursive=True):
            recs = [json.loads(x) for x in open(f) if x.strip()]
            runs[os.path.basename(os.path.dirname(f))] = dict(
                train=[r for r in recs if r["type"] == "train"], eval=[r for r in recs if r["type"] == "eval"])
    return base, zs, runs


ROW = ("| {name} | {expl} | {veq} | {vrand} | {vcall} | {evr} | {seen} | {held} | {regret} | {valid} | {tok} |")


def row(name, m, decision=True):
    f = lambda k, s="{:.3f}": s.format(m[k]) if decision and k in m else "–"  # noqa: E731
    return ROW.format(name=name, expl=f"{m['exploitability']:.3f}", veq=f"{m['vs_equilibrium']:+.3f}",
                      vrand=f"{m['vs_random']:+.3f}", vcall=f"{m['vs_always_call']:+.3f}", evr=f("ev_reward"),
                      seen=f("seen/ev_reward"), held=f("heldout/ev_reward"), regret=f("regret_chips"),
                      valid=f("valid_rate"), tok=f("avg_output_tokens", "{:.0f}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--save", required=True)
    a = p.parse_args()
    os.makedirs(a.save, exist_ok=True)
    base, zs, runs = load(a.dirs)

    lines = ["| Policy | Exploitability ↓ (chips/hand) | vs equilibrium | vs random | vs always-call | "
             "EV reward | seen | held-out | Regret (chips) ↓ | Valid answers | Avg tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, m in (base or {}).items():
        lines.append(row(f"{n} (bot)", m, decision=False))
    for k in sorted(zs, key=lambda s: int(s[1:])):
        lines.append(row(f"Qwen3-1.7B zero-shot {k} (k={zs[k]['samples_per_infoset']})", zs[k]))
    for name, r in sorted(runs.items()):
        if r["eval"]:
            e0, e1 = r["eval"][0], r["eval"][-1]
            lines.append(row(f"{name} step 0 (k={e0['samples_per_infoset']})", e0))
            lines.append(row(f"**{name} step {e1['step']}** (k={e1['samples_per_infoset']})", e1))
    table = "\n".join(lines)
    open(os.path.join(a.save, "results.md"), "w").write(table + "\n")
    print(table)

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    for name, r in sorted(runs.items()):
        ev = r["eval"]
        if ev:
            s = [e["step"] for e in ev]
            ln, = ax[0].plot(s, [e["exploitability"] for e in ev], "o-", label=name)
            ax[1].plot(s, [e["seen/ev_reward"] for e in ev], "o-", color=ln.get_color(), label=f"{name} seen")
            ax[1].plot(s, [e["heldout/ev_reward"] for e in ev], "s--", color=ln.get_color(), label=f"{name} held-out")
            c = ln.get_color()
        else:
            c = None
        tr = r["train"]
        if tr:
            ax[2].plot([t["step"] for t in tr], [t["true_ev_reward"] for t in tr], color=c, label=f"{name} (true EV)")
            ax[2].plot([t["step"] for t in tr], [t["reward_mean"] for t in tr], color=c, alpha=0.35, label=f"{name} (train reward)")
            ax[3].plot([t["step"] for t in tr], [t["avg_output_tokens"] for t in tr], color=c, label=name)
    if base:
        for n, m in base.items():
            if n.startswith(("uniform", "always check")):
                ax[0].axhline(m["exploitability"], ls=":", lw=1, color="gray")
                ax[0].text(0, m["exploitability"] + 0.03, n, fontsize=8, color="gray")
    for k, m in zs.items():
        ax[0].axhline(m["exploitability"], ls="--", lw=1, color="black", alpha=0.5)
        ax[0].text(0, m["exploitability"] - 0.12, f"zero-shot {k}", fontsize=8)
    ax[0].set(title="Exploitability (chips/hand, lower is better)", xlabel="GRPO step")
    ax[1].set(title="Eval EV reward: seen vs held-out situations", xlabel="GRPO step")
    ax[2].set(title="Train: reward used vs true EV of chosen actions", xlabel="GRPO step")
    ax[3].set(title="Avg output tokens (train rollouts)", xlabel="GRPO step")
    for x in ax:
        x.grid(alpha=0.3)
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.save, "problem2_curves.png"), dpi=140)
    print("saved to", a.save)


if __name__ == "__main__":
    main()
