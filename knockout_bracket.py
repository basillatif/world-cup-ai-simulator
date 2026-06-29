"""
Frozen 2026 FIFA World Cup knockout bracket (Round of 32 -> Final).

Design contract (do not violate):
  * The statistical model owns every prediction. This module defines bracket
    STRUCTURE and runs Monte Carlo given a caller-supplied match-outcome
    function. It never invents a probability and never narrates.
  * Knockout matches have NO group tiebreak logic. Ties resolve via extra time
    then a penalty shootout. That resolution lives INSIDE `match_fn`, which must
    draw all randomness from the supplied numpy Generator -- never random.random().

Topology sourced from FIFA regulations (Annex C) via the official combination
table. R32 seeding below is the frozen result of the 2026 group stage and has
been cross-checked against combination row 67 (3rd-place teams from groups
B,D,E,F,I,J,K,L). Freeze this dict to the append-only ledger before kickoff.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# A match source is one of:
#   ("seed", match_id)  -> use R32_SEEDING[match_id]
#   ("W", match_id)     -> winner of an earlier match
#   ("L", match_id)     -> loser of an earlier match (third-place game only)
Source = Tuple[str, int]

# --- Fixed forward topology: match_id -> (source_a, source_b) ------------------
# R32 are seeded directly; everything from the R16 on references prior matches.
BRACKET: Dict[int, Tuple[Source, Source]] = {
    # Round of 32 (seeded)
    **{m: (("seed", m), ("seed", m)) for m in range(73, 89)},
    # Round of 16
    89: (("W", 74), ("W", 77)),
    90: (("W", 73), ("W", 75)),
    91: (("W", 76), ("W", 78)),
    92: (("W", 79), ("W", 80)),
    93: (("W", 83), ("W", 84)),
    94: (("W", 81), ("W", 82)),
    95: (("W", 86), ("W", 88)),
    96: (("W", 85), ("W", 87)),
    # Quarter-finals
    97: (("W", 89), ("W", 90)),
    98: (("W", 93), ("W", 94)),
    99: (("W", 91), ("W", 92)),
    100: (("W", 95), ("W", 96)),
    # Semi-finals
    101: (("W", 97), ("W", 98)),
    102: (("W", 99), ("W", 100)),
    # Third-place play-off
    103: (("L", 101), ("L", 102)),
    # Final
    104: (("W", 101), ("W", 102)),
}

# Which match a winner advances INTO (for reaching-round bookkeeping).
ROUND_OF: Dict[int, str] = {
    **{m: "R32" for m in range(73, 89)},
    **{m: "R16" for m in range(89, 97)},
    **{m: "QF" for m in range(97, 101)},
    **{m: "SF" for m in (101, 102)},
    103: "3RD",
    104: "F",
}

# --- Frozen R32 seeding (verified vs FIFA combination row 67) ------------------
# slot label kept in a comment so it can be regenerated from standings later.
R32_SEEDING: Dict[int, Tuple[str, str]] = {
    73: ("South Africa", "Canada"),                 # A2 v B2
    74: ("Germany", "Paraguay"),                     # E1 v 3D
    75: ("Netherlands", "Morocco"),                  # F1 v C2
    76: ("Brazil", "Japan"),                         # C1 v F2
    77: ("France", "Sweden"),                        # I1 v 3F
    78: ("Ivory Coast", "Norway"),                   # E2 v I2
    79: ("Mexico", "Ecuador"),                       # A1 v 3E
    80: ("England", "DR Congo"),                     # L1 v 3K
    81: ("United States", "Bosnia and Herzegovina"),  # D1 v 3B
    82: ("Belgium", "Senegal"),                      # G1 v 3I
    83: ("Portugal", "Croatia"),                     # K2 v L2
    84: ("Spain", "Austria"),                        # H1 v J2
    85: ("Switzerland", "Algeria"),                  # B1 v 3J
    86: ("Argentina", "Cape Verde"),                 # J1 v H2
    87: ("Colombia", "Ghana"),                       # K1 v 3L
    88: ("Australia", "Egypt"),                      # D2 v G2
}

ROUNDS_ORDER = ["R32", "R16", "QF", "SF", "F", "WIN"]

# match_fn(team_a, team_b, rng) -> winning team name. Must resolve draws via
# ET/penalties internally using `rng` (a numpy Generator), never random.random().
MatchFn = Callable[[str, str, np.random.Generator], str]


def _resolve(source: Source, seeding: Dict[int, Tuple[str, str]],
             winners: Dict[int, str], losers: Dict[int, str], slot: int) -> str:
    kind, ref = source
    if kind == "seed":
        return seeding[ref][slot]
    if kind == "W":
        return winners[ref]
    if kind == "L":
        return losers[ref]
    raise ValueError(f"unknown source kind: {kind}")


def _play_once(match_fn: MatchFn, seeding: Dict[int, Tuple[str, str]],
               rng: np.random.Generator,
               decided: Optional[Dict[int, str]]) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Play one full bracket. Returns (winners, losers) keyed by match id."""
    decided = decided or {}
    winners: Dict[int, str] = {}
    losers: Dict[int, str] = {}
    for m in sorted(BRACKET):  # ascending ids => dependencies already resolved
        (sa, sb) = BRACKET[m]
        a = _resolve(sa, seeding, winners, losers, 0)
        b = _resolve(sb, seeding, winners, losers, 1)
        if m in decided:                       # condition on a known result
            w = decided[m]
            if w not in (a, b):
                raise ValueError(f"decided[{m}]={w!r} not in matchup {a} v {b}")
        else:
            w = match_fn(a, b, rng)
        winners[m] = w
        losers[m] = b if w == a else a
    return winners, losers


def simulate_bracket(match_fn: MatchFn,
                     seeding: Dict[int, Tuple[str, str]] = R32_SEEDING,
                     n: int = 10_000,
                     seed: int = 0,
                     decided: Optional[Dict[int, str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Monte Carlo over the knockout bracket.

    `decided` lets you freeze already-played results (e.g. {73: "Canada"}) so the
    sim re-runs CONDITIONED on them -- the "latency of alpha" re-simulation hook.

    Returns per team:
        {"reach": {round: prob}, "champion": prob}
    where reach[r] = P(team plays in round r). Rounds: R32,R16,QF,SF,F. Probs are
    rounded to 2 dp to match the cache-keying convention and absorb MC jitter.
    """
    teams = sorted({t for pair in seeding.values() for t in pair})
    # reach[team][round_index] count of sims where team appears in that round
    reach = {t: defaultdict(int) for t in teams}
    champ = defaultdict(int)
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)

    # match id -> the round its two participants are "reaching"
    participate_round = {m: ROUND_OF[m] for m in BRACKET}

    for _ in range(n):
        winners, losers = _play_once(match_fn, seeding, rng, decided)
        # a team "reaches" a round if it is a participant in that round's match
        seen = {r: set() for r in ("R32", "R16", "QF", "SF", "F")}
        for m, (sa, sb) in BRACKET.items():
            r = participate_round[m]
            if r == "3RD":
                continue
            a = _resolve(sa, seeding, winners, losers, 0)
            b = _resolve(sb, seeding, winners, losers, 1)
            seen[r].add(a)
            seen[r].add(b)
        for r, ts in seen.items():
            for t in ts:
                reach[t][r] += 1
        champ[winners[104]] += 1

    out: Dict[str, Dict[str, float]] = {}
    for t in teams:
        out[t] = {
            "reach": {r: round(reach[t][r] / n, 2) for r in ("R32", "R16", "QF", "SF", "F")},
            "champion": round(champ[t] / n, 2),
        }
    return out


if __name__ == "__main__":
    # Smoke run with a transparent placeholder engine (NOT for production):
    # stronger = earlier alphabetically, just to prove plumbing. Replace with the
    # real Elo + Dixon-Coles match function via the Codex wiring step.
    ratings = {t: i for i, t in enumerate(
        sorted({x for p in R32_SEEDING.values() for x in p}))}

    def demo_match(a: str, b: str, rng: np.random.Generator) -> str:
        pa = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 10))
        return a if rng.random() < pa else b

    res = simulate_bracket(demo_match, n=2000, seed=42, decided={73: "Canada"})
    top = sorted(res.items(), key=lambda kv: kv[1]["champion"], reverse=True)[:5]
    for team, r in top:
        print(f"{team:24s} win={r['champion']:.2f}  SF={r['reach']['SF']:.2f}")
