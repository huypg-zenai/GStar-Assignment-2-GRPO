"""Problem 2 analysis: playing style vs the equilibrium, where the exploitability comes from, and what the reasoning
traces talk about. Uses the saved policies (k=8 samples per info set) of zero-shot and the two final GRPO models.

  python report/analyze_p2.py ~/Downloads --out report/p2_analysis.json
"""
import argparse
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "problem2"))
import leduc as L  # noqa: E402

RUNS = {
    "zero-shot": ("outputs_p2_ev/zeroshot/zeroshot_L512_policy.json", "outputs_p2_ev/zeroshot/zeroshot_L512_samples.jsonl"),
    "GRPO ev": ("outputs_p2_ev/train_leduc_ev/eval_step080_policy.json", "outputs_p2_ev/train_leduc_ev/eval_step080_samples.jsonl"),
    "GRPO outcome": ("outputs_p2_outcome/train_leduc_outcome/eval_step080_policy.json",
                     "outputs_p2_outcome/train_leduc_outcome/eval_step080_samples.jsonl"),
}


def category(key):
    p, priv, pub, r1, r2 = L.parse_key(key)
    seq = r2 if pub else r1
    _, _, facing, _ = L.round_info(seq)
    rnd = "R2" if pub else "R1"
    if pub:
        hand = "pair" if priv == pub else ("high" if priv == "K" or (priv == "Q" and pub == "J") else "low")
    else:
        hand = priv
    return rnd, ("facing bet" if facing else "no bet"), hand


def style(pol, weights):
    """Reach-weighted action frequencies: P(fold | facing a bet), P(bet or raise | legal)."""
    fold_n = fold_d = agg_n = agg_d = 0.0
    for key, w in weights.items():
        probs = pol[key]
        if "f" in probs:
            fold_n += w * probs["f"]
            fold_d += w
        if "r" in probs:
            agg_n += w * probs["r"]
            agg_d += w
    return fold_n / fold_d, agg_n / agg_d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("downloads")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    eq = L.load_equilibrium()
    D = os.path.expanduser(a.downloads)
    # How often each info set is reached when BOTH players follow the equilibrium (= "real play" frequency).
    def real_reach(k):
        p = L.parse_key(k)[0]
        return L.own_reach(k, eq["policy"]) * sum(w * L.opp_reach(c, h1, h2, p, eq["policy"]) for c, h1, h2, w in L.INFOSETS[k])
    reach = {k: real_reach(k) for k in L.KEYS}
    print(f"sum of reach weights = {sum(reach.values()):.3f} (decisions per hand under equilibrium play)")
    res = {}
    pols = {"equilibrium": eq["policy"]}
    for name, (pp, sp) in RUNS.items():
        pols[name] = json.load(open(os.path.join(D, pp)))
    for name, pol in pols.items():
        fold, agg = style(pol, reach)
        br = [L.best_response(i, pol)[0] for i in (0, 1)]  # [BR as P1 vs model as P2, BR as P2 vs model as P1]
        m = dict(fold_when_facing_bet=fold, bet_raise_rate=agg, exploitability=L.exploitability(pol),
                 best_response_value_vs_seat=br)
        # regret by situation category, weighted by equilibrium reach
        cats = {}
        for key in L.KEYS:
            q = eq["q"][key]
            reg = max(q.values()) - sum(pol[key][x] * q[x] for x in q)
            c = " / ".join(category(key))
            cats.setdefault(c, []).append((reach[key], reg))
        m["regret_by_category"] = {c: sum(w * r for w, r in v) / max(1e-9, sum(w for w, _ in v)) for c, v in cats.items()}
        res[name] = m
        print(f"{name:13s} expl={m['exploitability']:.3f} BR wins vs model-as-P2/model-as-P1={br[0]:+.3f}/{br[1]:+.3f} "
              f"fold|bet={fold:.2f} bet/raise={agg:.2f}")
    print("\nReach-weighted regret (chips) by situation; equilibrium = 0")
    cats = sorted(res["GRPO ev"]["regret_by_category"])
    print(f"{'situation':30s}" + "".join(f"{n:>14s}" for n in RUNS))
    for c in cats:
        print(f"{c:30s}" + "".join(f"{res[n]['regret_by_category'][c]:14.3f}" for n in RUNS))

    # reasoning traces: what do they mention?
    print("\nReasoning traces (share of responses mentioning ...)")
    pats = {"pair": r"\bpair", "strong/weak hand": r"\b(strong|weak)\b", "opponent's range/hand": r"opponent (likely|may|might|could|has|holds?)",
            "pot / chips arithmetic": r"\bpot\b|\d+ chips", "bluff": r"bluff", "probability / odds": r"probab|odds|chance|%"}
    for name, (_, sp) in RUNS.items():
        rows = [json.loads(x) for x in open(os.path.join(D, sp)) if x.strip()]
        txt = [r["response"].lower() for r in rows]
        m = {k: sum(bool(re.search(v, t)) for t in txt) / len(txt) for k, v in pats.items()}
        m["median_tokens"] = statistics.median(r["tokens"] for r in rows)
        res[name]["trace_mentions"] = m
        print(f"{name:13s} " + " ".join(f"{k}={v:.2f}" for k, v in m.items()))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
