"""CPU tests for problem2/leduc.py. Reference numbers come from OpenSpiel's leduc_poker (checked with
open_spiel.python.algorithms.exploitability on the same policies)."""
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "problem2"))
import leduc as L  # noqa: E402

EQ = L.load_equilibrium()


def test_rules_match_openspiel():
    assert len(L.KEYS) == 288 and len(L.HISTORIES) == 36
    assert abs(sum(L.DEALS.values()) - 1) < 1e-12
    assert abs(2 * L.exploitability(L.uniform_policy()) - 4.747222222222222) < 1e-9   # OpenSpiel NashConv
    assert abs(L.exploitability(L.constant_policy("c")) - 1.466667) < 1e-6
    assert abs(L.exploitability(L.constant_policy("rc")) - 2.366667) < 1e-6


def test_payoffs():
    assert L.payoff0(("K", "Q", "J"), "cc", "cc") == 1                    # higher card wins the antes
    assert L.payoff0(("K", "Q", "Q"), "rc", "rc") == -7                   # pair beats K: 1 + 2 + 4
    assert L.payoff0(("J", "J", "K"), "rrc", "rrc") == 0                  # split
    assert L.payoff0(("J", "K", "Q"), "rf", "") == 1                      # player 1 folds to a bet
    assert L.payoff0(("J", "K", "Q"), "cc", "crf") == -1                  # player 0 checks, then folds to a bet
    assert L.payoff0(("J", "K", "Q"), "cc", "rrf") == -5                  # player 0 bets, folds to a raise: 1 + 4
    assert L.legal_actions("", "") == ["c", "r"] and L.legal_actions("r", "") == ["f", "c", "r"]
    assert L.legal_actions("rr", "") == ["f", "c"]                         # max 2 raises


def test_equilibrium():
    assert EQ["exploitability"] < 1e-4
    assert abs(L.game_value(EQ["policy"], EQ["policy"]) - (-0.0856)) < 1e-3  # known value of Leduc for player 1
    pair = EQ["q"]["1|Q|Q|cc|r"]
    assert pair["f"] < pair["c"]                                             # never fold a pair


def test_prompt_and_parse():
    k = "1|K|Q|cc|r"
    p = L.user_prompt(k)
    assert "Your private card: K." in p and "Public card: Q." in p and "Legal actions: fold, call, raise." in p
    assert "Opponent bets (4 chips)" in p and "To call: 4 chips." in p
    assert L.parse_action("<think>x</think><answer>call</answer>", k) == "c"
    assert L.parse_action("<answer> Raise </answer>", k) == "r"
    assert L.parse_action("<answer>check</answer>", k) is None             # not legal when facing a bet
    assert L.parse_action("I call", k) is None                             # no tag
    assert L.parse_action("<answer>fold</answer>", "0|J|||") is None       # fold not allowed without a bet
    assert L.parse_action("<answer>bet</answer>", "0|J|||") == "r"


def test_rewards():
    k = "1|K|Q|cc|r"
    ev = L.Rewards(EQ, "ev")
    q = EQ["q"][k]
    best = max(q, key=q.get)
    word = dict(zip(L.legal_actions("cc", "r"), L.action_words("cc", "r")))
    assert abs(ev(f"<answer>{word[best]}</answer>", {"key": k}) - 1.0) < 1e-9
    assert abs(ev("<answer>fold</answer>", {"key": k}) - 0.1) < 1e-9     # worst legal action
    assert ev("no answer", {"key": k}) == 0.0
    out = L.Rewards(EQ, "outcome", seed=1)
    xs = [out.sample_outcome(k, "c") for _ in range(8000)]
    assert abs(statistics.mean(xs) - q["c"]) < 0.15                        # unbiased Monte Carlo estimate of Q
    r = out("<answer>call</answer>", {"key": k})
    assert 0.1 <= r <= 1.0


def test_sampler_never_returns_heldout():
    s = L.StateSampler(EQ, seed=0)
    keys = [s.sample() for _ in range(3000)]
    assert not any(map(L.is_heldout, keys))
    assert sum(map(L.is_heldout, L.KEYS)) == 90


def test_policy_from_actions():
    k = "1|K|Q|cc|r"
    pol = L.policy_from_actions({k: ["c", "c", None, "r"]})
    assert pol[k] == {"f": 0.25, "c": 0.5, "r": 0.25}                    # None -> fold (fallback when facing a bet)
    rep = L.policy_report(EQ["policy"], EQ)
    assert abs(rep["vs_equilibrium"]) < 1e-9 and rep["vs_random"] > 0.5
    random.seed(0)
