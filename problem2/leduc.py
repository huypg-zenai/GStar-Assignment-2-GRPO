"""Leduc Hold'em for Problem 2: rules, exact equilibrium (CFR+), action values, best response / exploitability,
natural-language prompts, answer parsing and reward functions.

Rules (identical to OpenSpiel's `leduc_poker`, 2 players):
  * 6 cards: J J Q Q K K. Each player antes 1 chip and receives one private card.
  * Two betting rounds. Player 1 (index 0) acts first in both. Bet/raise size: 2 chips in round 1, 4 in round 2.
    At most 2 bets/raises per round. Fold is only allowed when facing a bet.
  * After round 1 a public card is revealed. Showdown: pairing the public card wins, else the higher card wins,
    equal ranks split.
Actions: 'f' fold, 'c' check/call, 'r' bet/raise. A history is (r1, r2), the action strings of the two rounds.
Suits never matter strategically, so information sets use ranks; chance probabilities still account for card
removal by enumerating the 6 physical cards.

  python problem2/leduc.py solve --iters 20000    # CFR+ -> problem2/leduc_equilibrium.json
"""
import argparse
import itertools
import json
import math
import os
import random
import re

RANKS = "JQK"
RAISE = (2, 4)
MAX_RAISES = 2
HERE = os.path.dirname(os.path.abspath(__file__))
EQ_PATH = os.path.join(HERE, "leduc_equilibrium.json")
# Held-out generalization split: round-2 situations with these (private, public) rank pairs never appear in training.
HELDOUT_PAIRS = {("J", "K"), ("Q", "J"), ("K", "Q")}


# ------------------------------------------------------------------ rules
def round_info(seq):
    """Actions of one round (no fold) -> (to_act, n_raises, facing_a_bet, round_done)."""
    return len(seq) % 2, seq.count("r"), bool(seq) and seq[-1] == "r", len(seq) >= 2 and seq[-1] == "c"


def round1_done(r1):
    return round_info(r1)[3]


def is_terminal(r1, r2):
    return r1.endswith("f") or r2.endswith("f") or (round1_done(r1) and round_info(r2)[3])


def current_player(r1, r2):
    return len(r2 if round1_done(r1) else r1) % 2


def legal_actions(r1, r2):
    _, n_r, facing, _ = round_info(r2 if round1_done(r1) else r1)
    acts = ["f", "c"] if facing else ["c"]
    if n_r < MAX_RAISES:
        acts.append("r")
    return acts


def child(r1, r2, a):
    return (r1, r2 + a) if round1_done(r1) else (r1 + a, r2)


def contributions(r1, r2):
    c = [1, 1]
    for k, seq in enumerate((r1, r2)):
        for i, a in enumerate(seq):
            if a == "c":
                c[i % 2] = max(c)
            elif a == "r":
                c[i % 2] = max(c) + RAISE[k]
    return c


def payoff0(cards, r1, r2):
    """Chips won by player 0 at a terminal history. cards = (rank0, rank1, public_rank)."""
    c = contributions(r1, r2)
    seq = r2 if r2 else r1
    if seq.endswith("f"):
        folder = (len(seq) - 1) % 2
        return -c[0] if folder == 0 else c[1]
    a, b, pub = cards
    s0, s1 = (a == pub, RANKS.index(a)), (b == pub, RANKS.index(b))
    return c[1] if s0 > s1 else (-c[0] if s1 > s0 else 0)


def deals():
    """{(rank0, rank1, public_rank): probability}, from the 6*5*4 equally likely physical deals."""
    out = {}
    for x, y, z in itertools.permutations(range(6), 3):
        k = (RANKS[x // 2], RANKS[y // 2], RANKS[z // 2])
        out[k] = out.get(k, 0.0) + 1.0 / 120
    return out


DEALS = deals()


def infokey(p, cards, r1, r2):
    pub = cards[2] if round1_done(r1) else ""
    return f"{p}|{cards[p]}|{pub}|{r1}|{r2}"


def parse_key(key):
    p, priv, pub, r1, r2 = key.split("|")
    return int(p), priv, pub, r1, r2


def all_histories():
    res = []

    def rec(r1, r2):
        if is_terminal(r1, r2):
            return
        res.append((r1, r2))
        for a in legal_actions(r1, r2):
            rec(*child(r1, r2, a))
    rec("", "")
    return res


HISTORIES = all_histories()


def infosets():
    """{key: [(cards, r1, r2, chance_prob), ...]} for every decision node of both players."""
    out = {}
    for cards, w in DEALS.items():
        for r1, r2 in HISTORIES:
            p = current_player(r1, r2)
            out.setdefault(infokey(p, cards, r1, r2), []).append((cards, r1, r2, w))
    return out


INFOSETS = infosets()
KEYS = sorted(INFOSETS)


def is_heldout(key):
    _, priv, pub, _, _ = parse_key(key)
    return pub != "" and (priv, pub) in HELDOUT_PAIRS


# ------------------------------------------------------------------ policies and values
def uniform_policy():
    return {k: {a: 1.0 / len(legal_actions(*parse_key(k)[3:])) for a in legal_actions(*parse_key(k)[3:])}
            for k in KEYS}


def constant_policy(pref):
    """Always take the first legal action in `pref` (e.g. 'c' = always check/call, 'rc' = always bet/raise)."""
    pol = {}
    for k in KEYS:
        acts = legal_actions(*parse_key(k)[3:])
        a = next(x for x in pref if x in acts)
        pol[k] = {x: float(x == a) for x in acts}
    return pol


def ev0(cards, r1, r2, pol0, pol1):
    """Expected chips for player 0 from history (r1, r2) when player i follows pol_i."""
    if is_terminal(r1, r2):
        return payoff0(cards, r1, r2)
    p = current_player(r1, r2)
    dist = (pol0, pol1)[p][infokey(p, cards, r1, r2)]
    return sum(pr * ev0(cards, *child(r1, r2, a), pol0, pol1) for a, pr in dist.items() if pr > 0)


def game_value(pol0, pol1):
    return sum(w * ev0(cards, "", "", pol0, pol1) for cards, w in DEALS.items())


def head_to_head(pol, opp):
    """Average chips per hand for `pol` against `opp`, averaged over both seats."""
    return 0.5 * (game_value(pol, opp) - game_value(opp, pol))


def opp_reach(cards, r1, r2, p, pol_opp):
    """Probability that the opponent of p takes its actions along (r1, r2)."""
    reach, h1, h2 = 1.0, "", ""
    for a in r1 + r2:
        q = current_player(h1, h2)
        if q != p:
            reach *= pol_opp[infokey(q, cards, h1, h2)].get(a, 0.0)
        h1, h2 = child(h1, h2, a)
    return reach


def own_reach(key, pol):
    """Probability that the player at `key` takes its own actions on the way there (same for all nodes in it)."""
    p, _, _, r1, r2 = parse_key(key)
    cards = INFOSETS[key][0][0]
    reach, h1, h2 = 1.0, "", ""
    for a in r1 + r2:
        if current_player(h1, h2) == p:
            reach *= pol[infokey(p, cards, h1, h2)].get(a, 0.0)
        h1, h2 = child(h1, h2, a)
    return reach


def best_response(i, pol_opp):
    """Exact best response of player i to pol_opp -> (value for player i, deterministic BR policy)."""
    br = {}
    keys_i = sorted((k for k in KEYS if k.startswith(f"{i}|")),
                    key=lambda k: -(len(parse_key(k)[3]) + len(parse_key(k)[4])))  # deepest first

    def util(cards, r1, r2):
        pols = (br, pol_opp) if i == 0 else (pol_opp, br)
        v = ev0(cards, r1, r2, *pols)
        return v if i == 0 else -v

    for k in keys_i:
        _, _, _, r1, r2 = parse_key(k)
        vals = {}
        for a in legal_actions(r1, r2):
            vals[a] = sum(w * opp_reach(cards, h1, h2, i, pol_opp) * util(cards, *child(h1, h2, a))
                          for cards, h1, h2, w in INFOSETS[k])
        best = max(vals, key=vals.get)
        br[k] = {a: float(a == best) for a in vals}
    value = sum(w * util(cards, "", "") for cards, w in DEALS.items())
    return value, br


def exploitability(pol):
    """(BR_0 vs pol + BR_1 vs pol) / 2 in chips per hand; 0 exactly at a Nash equilibrium (OpenSpiel convention)."""
    return 0.5 * (best_response(0, pol)[0] + best_response(1, pol)[0])


def action_values(key, pol):
    """Q(I, a) for the player at info set `key`: expected chips after taking a, with both players following `pol`
    afterwards, and the opponent's hidden card weighted by chance x opponent reach under `pol` (a Bayesian belief).
    If `pol` never reaches this info set from the opponent's side, chance weights alone are used."""
    p, _, _, r1, r2 = parse_key(key)
    nodes = INFOSETS[key]
    wts = [w * opp_reach(cards, h1, h2, p, pol) for cards, h1, h2, w in nodes]
    if sum(wts) < 1e-12:
        wts = [w for _, _, _, w in nodes]
    tot = sum(wts)
    out = {}
    for a in legal_actions(r1, r2):
        v = 0.0
        for (cards, h1, h2, _), wt in zip(nodes, wts):
            u = ev0(cards, *child(h1, h2, a), pol, pol)
            v += wt * (u if p == 0 else -u)
        out[a] = v / tot
    return out, wts


# ------------------------------------------------------------------ CFR+ solver
def solve(iters=20000, log_every=5000):
    regrets = {k: {a: 0.0 for a in legal_actions(*parse_key(k)[3:])} for k in KEYS}
    avg = {k: {a: 0.0 for a in v} for k, v in regrets.items()}

    def current(k):
        r = regrets[k]
        pos = {a: max(x, 0.0) for a, x in r.items()}
        s = sum(pos.values())
        return {a: (x / s if s > 0 else 1.0 / len(r)) for a, x in pos.items()}

    def cfr(cards, r1, r2, i, pi_i, pi_o, sig, t):
        if is_terminal(r1, r2):
            u = payoff0(cards, r1, r2)
            return u if i == 0 else -u
        p = current_player(r1, r2)
        k = infokey(p, cards, r1, r2)
        s = sig[k]
        if p != i:
            return sum(pr * cfr(cards, *child(r1, r2, a), i, pi_i, pi_o * pr, sig, t) for a, pr in s.items() if pr > 0)
        vals = {a: cfr(cards, *child(r1, r2, a), i, pi_i * s[a], pi_o, sig, t) for a in s}
        v = sum(s[a] * vals[a] for a in s)
        for a in s:
            delta[k][a] += pi_o * (vals[a] - v)
            avg[k][a] += t * pi_i * s[a]
        return v

    for t in range(1, iters + 1):
        for i in (0, 1):
            sig = {k: current(k) for k in KEYS}
            delta = {k: {a: 0.0 for a in v} for k, v in regrets.items()}
            for cards, w in DEALS.items():
                cfr(cards, "", "", i, 1.0, w, sig, t)
            for k in KEYS:
                if k.startswith(f"{i}|"):
                    for a in regrets[k]:
                        regrets[k][a] = max(regrets[k][a] + delta[k][a], 0.0)   # CFR+ floor
        if log_every and (t % log_every == 0 or t == iters):
            pol = normalize(avg)
            print(f"iter {t}: exploitability {exploitability(pol):.6f} chips/hand, "
                  f"value(P1) {game_value(pol, pol):+.5f}", flush=True)
    return normalize(avg)


def normalize(table):
    out = {}
    for k, v in table.items():
        s = sum(v.values())
        out[k] = {a: (x / s if s > 0 else 1.0 / len(v)) for a, x in v.items()}
    return out


def build_equilibrium(iters):
    pol = solve(iters)
    q = {}
    for k in KEYS:
        vals, wts = action_values(k, pol)
        q[k] = vals
    data = dict(policy=pol, q=q, exploitability=exploitability(pol), value_p1=game_value(pol, pol), iters=iters)
    json.dump(data, open(EQ_PATH, "w"), indent=0, sort_keys=True)
    print(f"saved {EQ_PATH}: exploitability {data['exploitability']:.6f}, P1 value {data['value_p1']:+.5f}")
    return data


def load_equilibrium():
    return json.load(open(EQ_PATH))


# ------------------------------------------------------------------ language interface
WORDS = {"f": "fold", "c": "check", "r": "bet"}


def action_words(r1, r2):
    """Legal actions as words, in order. Facing a bet: fold / call / raise; otherwise: check / bet."""
    facing = round_info(r2 if round1_done(r1) else r1)[2]
    return [({"f": "fold", "c": "call", "r": "raise"} if facing else WORDS)[a] for a in legal_actions(r1, r2)]


def describe_round(seq, k, me):
    parts, contrib, raises = [], [0, 0], 0
    for i, a in enumerate(seq):
        who = "you" if i % 2 == me else "opponent"
        facing = bool(seq[:i]) and seq[i - 1] == "r"
        if a == "c":
            parts.append(f"{who} {'call' if facing else 'check'}" + ("s" if who == "opponent" else "")
                         + (f" ({contrib[1 - i % 2] - contrib[i % 2]} chips)" if facing else ""))
            contrib[i % 2] = max(contrib)
        elif a == "r":
            amt = max(contrib) + RAISE[k] - contrib[i % 2]
            verb = "raise" if raises else "bet"
            parts.append(f"{who} {verb}" + ("s" if who == "opponent" else "") + f" ({amt} chips)")
            contrib[i % 2] = max(contrib) + RAISE[k]
            raises += 1
    text = ", ".join(parts) if parts else "no actions yet"
    return text[0].upper() + text[1:]


RULES = """You are playing Leduc Hold'em, a two-player poker game, against a strong opponent who plays a game-theoretically optimal strategy.
Rules:
- The deck has 6 cards: two Jacks (J), two Queens (Q) and two Kings (K). K > Q > J.
- Each player puts 1 chip (ante) into the pot and gets one private card. Only you can see your card.
- There are two betting rounds. Before round 2, one public card is revealed and shared by both players.
- Player 1 acts first in each round. Bets and raises are 2 chips in round 1 and 4 chips in round 2. At most 2 bets/raises per round.
- When facing a bet you can fold, call or raise; otherwise you can check or bet. A round ends when a bet is called or both players check.
- If a player folds, the other player wins the pot. At showdown, a player whose private card matches the public card (a pair) wins; otherwise the higher private card wins; equal cards split the pot."""


def render_state(key):
    p, priv, pub, r1, r2 = parse_key(key)
    c = contributions(r1, r2)
    lines = [f"You are Player {p + 1}. Your private card: {priv}.",
             f"Public card: {pub if pub else 'not revealed yet'}.",
             f"Round 1 (bets of 2 chips): {describe_round(r1, 0, p)}."]
    if round1_done(r1):
        lines.append(f"Round 2 (bets of 4 chips): {describe_round(r2, 1, p)}.")
    to_call = c[1 - p] - c[p]
    lines.append(f"Pot: {c[0] + c[1]} chips (you put in {c[p]}, opponent put in {c[1 - p]})."
                 + (f" To call: {to_call} chips." if to_call else ""))
    lines.append(f"Legal actions: {', '.join(action_words(r1, r2))}.")
    return "\n".join(lines)


TEMPLATE = """{rules}

Current situation:
{state}

Choose the action that maximizes your expected chips. Think about your hand strength and what your opponent may hold. Show your reasoning in <think> </think> tags and give only the action in <answer> </answer> tags, for example <answer>{example}</answer>."""


def user_prompt(key):
    _, _, _, r1, r2 = parse_key(key)
    return TEMPLATE.format(rules=RULES, state=render_state(key), example=action_words(r1, r2)[-1])


def parse_action(text, key):
    """Last <answer>...</answer> -> engine action ('f'/'c'/'r'), or None if missing or not a legal action."""
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL))
    if not m:
        return None
    _, _, _, r1, r2 = parse_key(key)
    words = action_words(r1, r2)
    found = re.findall(r"[a-z]+", m[-1].group(1).lower())
    for w in found:
        if w in words:
            return legal_actions(r1, r2)[words.index(w)]
    return None


def fallback_action(key):
    """What a non-answer means in actual play: fold when facing a bet, otherwise check (a passive forfeit)."""
    return legal_actions(*parse_key(key)[3:])[0]


class Rewards:
    """reward(response, gt) for GRPO; gt = {"key": info set}. Legal action -> 0.1 + 0.9 * score; no tag/illegal -> 0.

    ev      : score = (Q*(a) - min Q*) / (max Q* - min Q*), Q* = action values under the CFR+ equilibrium (exact,
              noiseless "how good was this decision against an optimal opponent").
    outcome : score = (chips won + 13) / 26 from ONE simulated hand: the opponent's card is sampled from the
              equilibrium belief at this info set, then both players follow the equilibrium to the end (noisy,
              realistic "play it out and see" reward). 13 chips is the largest possible win or loss.
    """
    def __init__(self, eq, mode, seed=0):
        self.pol, self.q, self.mode = eq["policy"], eq["q"], mode
        self.rng = random.Random(seed)
        self._belief = {}

    def ev_score(self, key, a):
        q = self.q[key]
        lo, hi = min(q.values()), max(q.values())
        return 1.0 if hi - lo < 1e-9 else (q[a] - lo) / (hi - lo)

    def regret(self, key, a):
        q = self.q[key]
        return max(q.values()) - q[a]

    def sample_outcome(self, key, a):
        if key not in self._belief:
            self._belief[key] = action_values(key, self.pol)[1]
        nodes = INFOSETS[key]
        cards, h1, h2, _ = self.rng.choices(nodes, weights=self._belief[key])[0]
        r1, r2 = child(h1, h2, a)
        while not is_terminal(r1, r2):
            q = current_player(r1, r2)
            dist = self.pol[infokey(q, cards, r1, r2)]
            act = self.rng.choices(list(dist), weights=list(dist.values()))[0]
            r1, r2 = child(r1, r2, act)
        u = payoff0(cards, r1, r2)
        return u if parse_key(key)[0] == 0 else -u

    def __call__(self, response, gt):
        key = gt["key"]
        a = parse_action(response, key)
        if a is None:
            return 0.0
        if self.mode == "ev":
            return 0.1 + 0.9 * self.ev_score(key, a)
        return 0.1 + 0.9 * (self.sample_outcome(key, a) + 13) / 26


class StateSampler:
    """Training situations: 50% from equilibrium self-play (on-distribution), 50% uniform over info sets (coverage).
    Held-out info sets (HELDOUT_PAIRS in round 2) are never returned."""
    def __init__(self, eq, seed=0):
        self.pol = eq["policy"]
        self.rng = random.Random(seed)
        self.train_keys = [k for k in KEYS if not is_heldout(k)]

    def sample(self):
        if self.rng.random() < 0.5:
            return self.rng.choice(self.train_keys)
        while True:
            cards = self.rng.choices(list(DEALS), weights=list(DEALS.values()))[0]
            r1, r2, seen = "", "", []
            while not is_terminal(r1, r2):
                p = current_player(r1, r2)
                k = infokey(p, cards, r1, r2)
                seen.append(k)
                dist = self.pol[k]
                r1, r2 = child(r1, r2, self.rng.choices(list(dist), weights=list(dist.values()))[0])
            k = self.rng.choice(seen)
            if not is_heldout(k):
                return k


def policy_from_actions(actions_by_key):
    """{key: [action or None, ...]} sampled from the LLM -> stochastic policy table (None -> fallback action).
    Info sets with no samples get the uniform policy."""
    pol = uniform_policy()
    for k, acts in actions_by_key.items():
        if acts:
            legal = legal_actions(*parse_key(k)[3:])
            cnt = {a: 0 for a in legal}
            for a in acts:
                cnt[a if a is not None else fallback_action(k)] += 1
            pol[k] = {a: c / len(acts) for a, c in cnt.items()}
    return pol


def policy_report(pol, eq):
    """Game-level metrics of a full policy table (exact, no sampling)."""
    star = eq["policy"]
    mixed = [k for k in KEYS if own_reach(k, star) > 0.01 and sum(1 for x in star[k].values() if x > 0.05) >= 2]
    ent = lambda d: -sum(x * math.log(x) for x in d.values() if x > 0)  # noqa: E731
    return dict(
        exploitability=exploitability(pol),
        vs_equilibrium=head_to_head(pol, star),
        vs_random=head_to_head(pol, uniform_policy()),
        vs_always_call=head_to_head(pol, constant_policy("c")),
        vs_always_raise=head_to_head(pol, constant_policy("rc")),
        entropy_at_eq_mixed=sum(ent(pol[k]) for k in mixed) / max(len(mixed), 1),
        n_eq_mixed_infosets=len(mixed),
    )


def baseline_reports(eq):
    return {name: policy_report(pol, eq) for name, pol in [
        ("equilibrium (CFR+)", eq["policy"]), ("uniform random", uniform_policy()),
        ("always check/call", constant_policy("c")), ("always bet/raise", constant_policy("rc"))]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["solve", "baselines", "show"])
    ap.add_argument("--iters", type=int, default=20000)
    a = ap.parse_args()
    if a.cmd == "solve":
        build_equilibrium(a.iters)
    elif a.cmd == "baselines":
        for n, r in baseline_reports(load_equilibrium()).items():
            print(f"{n:22s}", {k: round(v, 4) for k, v in r.items()})
    else:
        print(f"{len(KEYS)} info sets ({sum(map(is_heldout, KEYS))} held out)\n")
        print(user_prompt("1|K|Q|cc|r"))
