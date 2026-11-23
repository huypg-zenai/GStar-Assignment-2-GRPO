"""Report figures from report/p1_analysis.json (python report/analyze_p1.py first).

  python report/make_figures.py
"""
import json
import os

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# Reference palette (dataviz skill, light mode), validated: categorical slots 1-5, neutral gray for baselines.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2, GRID = "#9b9a94", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
P1 = json.load(open(os.path.join(HERE, "p1_analysis.json")))


def acc(name):
    return 100 * P1[name]["correct"] / P1[name]["n"]


def fig_accuracy():
    groups = ["L256", "L512", "L1024"]
    series = [("Zero-shot", GRAY, lambda L: acc(f"zero-shot {L}")),
              ("GRPO", BLUE, lambda L: acc(f"GRPO {L}") if L != "L1024" else None),
              ("Dr.GRPO", ORANGE, lambda L: acc(f"Dr.GRPO {L}") if L != "L1024" else None)]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    w = 0.26
    for j, (lab, col, f) in enumerate(series):
        for i, L in enumerate(groups):
            v = f(L)
            if v is None:
                continue
            x = i + (j - 1) * w
            ax.bar(x, v, w - 0.03, color=col, label=lab if i == 0 else None, zorder=3)
            ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(37, ls="--", lw=1, color=INK2, zorder=2)
    ax.text(2.45, 38.2, "course zero-shot\nL1024 ≈ 37%", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(3), groups)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 72)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=1, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_accuracy.pdf"))
    fig.savefig(os.path.join(FIG, "p1_accuracy.png"), dpi=160)


def fig_failures():
    runs = ["zero-shot L256", "zero-shot L512", "zero-shot L1024", "GRPO L256", "Dr.GRPO L256", "GRPO L512", "Dr.GRPO L512"]
    cats = [("correct", BLUE, ["correct"]), ("wrong value", ORANGE, ["wrong_value"]),
            ("invalid numbers/expr.", AQUA, ["bad_numbers", "bad_expression"]),
            ("no tag: hit limit", YELLOW, ["no_tag_truncated"]),
            ("no tag: stopped", MAGENTA, ["no_tag_finished"])]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for i, r in enumerate(runs):
        left = 0.0
        for lab, col, keys in cats:
            v = 100 * sum(P1[r][k] for k in keys) / P1[r]["n"]
            ax.barh(i, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.2,
                    label=lab if i == 0 else None, zorder=3)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
            left += v
    ax.set_yticks(range(len(runs)), runs)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of test responses (%)")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.35, 1.0), handlelength=1, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_failures.pdf"))
    fig.savefig(os.path.join(FIG, "p1_failures.png"), dpi=160)


if __name__ == "__main__":
    fig_accuracy()
    fig_failures()
    print("saved to", FIG)
