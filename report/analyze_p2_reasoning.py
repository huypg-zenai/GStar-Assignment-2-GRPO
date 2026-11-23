"""Problem 2: is the model's stated reasoning correct? For every round-2 answer, compare the trace's hand-strength
claim ("you have a weak/strong hand") with the hand's real strength: its showdown equity against a random unseen
opponent card (strong if > 0.5). Also counts invented poker terms.

  python report/analyze_p2_reasoning.py ~/Downloads --out report/p2_reasoning.json
"""
import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "problem2"))
import leduc as L  # noqa: E402

RUNS = {"zero-shot": "outputs_p2_ev/zeroshot/zeroshot_L512_samples.jsonl",
        "GRPO ev": "outputs_p2_ev/train_leduc_ev/eval_step080_samples.jsonl",
        "GRPO outcome": "outputs_p2_outcome/train_leduc_outcome/eval_step080_samples.jsonl"}
RANK = {"J": 0, "Q": 1, "K": 2}
CLAIM = re.compile(r"(you have|giving you|so you have|so your hand is|your hand is)[^.<]{0,40}?\b(weak|strong|low|good|high)")
INVENTED = re.compile(r"\b(flush|straight|ace|10|qt|kt|two pair|three of a kind)\b")


def equity(priv, pub):
    deck = ["J", "J", "Q", "Q", "K", "K"]
    deck.remove(priv)
    deck.remove(pub)
    s = 0.0
    for o in deck:
        mine, theirs = (priv == pub, RANK[priv]), (o == pub, RANK[o])
        s += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return s / len(deck)


def claim(text):
    m = CLAIM.search(text.lower())
    return None if m is None else ("strong" if m.group(2) in ("strong", "good", "high") else "weak")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("downloads")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {}
    for name, rel in RUNS.items():
        rows = [json.loads(x) for x in open(os.path.join(os.path.expanduser(a.downloads), rel)) if x.strip()]
        r2 = [r for r in rows if L.parse_key(r["key"])[2] and r["action"] is not None]
        c = collections.Counter()
        by_hand = collections.defaultdict(collections.Counter)
        for r in r2:
            _, priv, pub, _, _ = L.parse_key(r["key"])
            cl = claim(r["response"])
            if cl is None:
                c["no_claim"] += 1
                continue
            truth = "strong" if equity(priv, pub) > 0.5 else "weak"
            ok = cl == truth
            c["claim_correct" if ok else "claim_wrong"] += 1
            c[f"{'right' if ok else 'wrong'}_reason_{'best' if r['best'] else 'notbest'}_action"] += 1
            if priv != pub and equity(priv, pub) > 0.5:          # favourite without a pair: K on J/Q, Q on K
                by_hand["unpaired favourite"][cl] += 1
            if priv == pub:
                by_hand["pair"][cl] += 1
        m = dict(c)
        m["n_round2"] = len(r2)
        m["claim_accuracy"] = c["claim_correct"] / (c["claim_correct"] + c["claim_wrong"])
        m["invented_terms"] = sum(bool(INVENTED.search(r["response"].lower())) for r in rows)
        m["n_all"] = len(rows)
        m["by_hand"] = {k: dict(v) for k, v in by_hand.items()}
        res[name] = m
        print(f"{name:13s} claim accuracy {m['claim_accuracy']:.0%} | best action with wrong reason "
              f"{c['wrong_reason_best_action']} | unpaired favourite called {dict(by_hand['unpaired favourite'])} | "
              f"invented terms {m['invented_terms']}/{len(rows)}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
