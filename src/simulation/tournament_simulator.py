"""Monte Carlo World Cup 2026 tournament simulator (48 teams, 12 groups).

Performance design
------------------
`predict_probs` computes a full Poisson score matrix on every call (~0.1 ms).
With 10 000 simulations × 135 matches each that adds up to ~180 s.

The fix: pre-compute a probability lookup table for every ordered team pair
*once* before the simulation loop (≈ 250 ms for 2 256 pairs).  The inner loop
then does only dict look-ups and a single `rng.random()` draw per match.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.elo import EloRatings
from src.models.match_predictor import MatchPredictor
from src.simulation.live_tracker import build_locked_group_results


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GroupResult:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def _build_prob_table(
    teams: list[str],
    predictor: MatchPredictor,
) -> tuple[dict[tuple[str, str], tuple[float, float, float]],
           dict[tuple[str, str], tuple[float, float]]]:
    """
    Pre-compute (home_win_prob, draw_prob, away_win_prob) and
    (lambda_home, lambda_away) for every ordered pair of teams.

    Called once before the simulation loop — ~250 ms for 48 teams.
    """
    prob_table: dict[tuple[str, str], tuple[float, float, float]] = {}
    goals_table: dict[tuple[str, str], tuple[float, float]] = {}

    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                continue
            p = predictor.predict_probs(t1, t2, neutral=True)
            prob_table[(t1, t2)] = (p["home_win"], p["draw"], p["away_win"])
            goals_table[(t1, t2)] = predictor.poisson.expected_goals(t1, t2, neutral=True)

    return prob_table, goals_table


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def simulate_group_stage(
    group_structure: dict[str, list[str]],
    prob_table: dict[tuple[str, str], tuple[float, float, float]],
    goals_table: dict[tuple[str, str], tuple[float, float]],
    rng: np.random.Generator,
    completed_results: dict[frozenset[str], tuple[str, str, int, int]] | None = None,
) -> dict[str, list[GroupResult]]:
    """Simulate group stage. Returns group → [1st…4th] GroupResult list."""
    standings: dict[str, list[GroupResult]] = {}

    for group, teams in group_structure.items():
        table: dict[str, GroupResult] = {t: GroupResult(team=t) for t in teams}

        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                completed = (completed_results or {}).get(frozenset((t1, t2)))
                if completed:
                    home, away, hg, ag = completed
                    _update_table(table, home, away, hg, ag)
                    continue

                hw, d, aw = prob_table[(t1, t2)]
                lam_h, lam_a = goals_table[(t1, t2)]

                # Outcome from blended probs (single rng.random() call)
                r = rng.random()
                if r < hw:
                    outcome = "home_win"
                elif r < hw + d:
                    outcome = "draw"
                else:
                    outcome = "away_win"

                # Scoreline sampled from Poisson (for GD / GF tiebreakers)
                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))

                # Force scoreline to be consistent with blended outcome
                if outcome == "home_win" and hg <= ag:
                    hg, ag = ag + 1, max(ag - 1, 0)
                elif outcome == "away_win" and ag <= hg:
                    ag, hg = hg + 1, max(hg - 1, 0)
                elif outcome == "draw":
                    ag = hg  # make it a draw

                _update_table(table, t1, t2, hg, ag)

        ranked = sorted(
            table.values(),
            key=lambda r: (r.points, r.gd, r.gf, random.random()),
            reverse=True,
        )
        standings[group] = ranked

    return standings


def _update_table(
    table: dict[str, GroupResult],
    home: str,
    away: str,
    hg: int,
    ag: int,
) -> None:
    table[home].played += 1
    table[away].played += 1
    table[home].gf += hg
    table[home].ga += ag
    table[away].gf += ag
    table[away].ga += hg

    if hg > ag:
        table[home].wins += 1
        table[away].losses += 1
    elif hg == ag:
        table[home].draws += 1
        table[away].draws += 1
    else:
        table[away].wins += 1
        table[home].losses += 1


# ---------------------------------------------------------------------------
# 3rd-place selection
# ---------------------------------------------------------------------------

def _select_best_third_place(
    standings: dict[str, list[GroupResult]],
    n: int = 8,
) -> list[str]:
    thirds = [ranked[2] for ranked in standings.values() if len(ranked) >= 3]
    thirds.sort(key=lambda r: (r.points, r.gd, r.gf, random.random()), reverse=True)
    return [r.team for r in thirds[:n]]


# ---------------------------------------------------------------------------
# Knockout stage
# ---------------------------------------------------------------------------

# Standard 16-slot single-elimination seeding order: bracket_slot_order[i] is
# the 1-indexed seed placed in slot i, arranged so seed 1 and seed 2 can only
# meet in the final, seeds 1-4 can only meet from the semifinals on, etc.
_BRACKET_SLOT_ORDER = [1, 16, 8, 9, 4, 13, 5, 12, 2, 15, 7, 10, 3, 14, 6, 11]


def _seed_round_of_32_pairs(
    group_winners: list[str],
    runners_up: list[str],
    best_thirds: list[str],
    elo: EloRatings,
) -> list[tuple[str, str]]:
    """Pot-based R32 pairing: group winners vs. weaker non-winners, with the
    16 resulting matchups placed in the bracket so the strongest teams are
    kept apart for as long as possible.
    """
    # Pot 1: group winners, strongest first.
    pot1 = sorted(group_winners, key=lambda t: elo.get(t), reverse=True)
    # Pot 2: runners-up + best thirds, strongest first.
    pot2 = sorted(runners_up + best_thirds, key=lambda t: elo.get(t), reverse=True)

    # Split pot 2 so each group winner faces one of the weakest pot-2 teams,
    # leaving the strongest leftover pot-2 teams to play each other.
    n_extra = len(pot2) - len(pot1)
    strong_pot2 = pot2[:n_extra]
    weak_pot2 = pot2[n_extra:]
    matchups = [
        (pot1[i], weak_pot2[len(weak_pot2) - 1 - i]) for i in range(len(pot1))
    ]
    for i in range(len(strong_pot2) // 2):
        matchups.append((strong_pot2[i], strong_pot2[len(strong_pot2) - 1 - i]))

    # Rank matchups by their strongest team's Elo (best matchup first) and
    # place them into bracket slots so top matchups are maximally separated.
    matchups.sort(key=lambda m: max(elo.get(m[0]), elo.get(m[1])), reverse=True)
    slots: list[tuple[str, str] | None] = [None] * len(matchups)
    for seed, matchup in enumerate(matchups, start=1):
        slot = _BRACKET_SLOT_ORDER.index(seed)
        slots[slot] = matchup
    return slots


def _knockout_match(
    team_a: str,
    team_b: str,
    prob_table: dict[tuple[str, str], tuple[float, float, float]],
    rng: np.random.Generator,
) -> str:
    """Single-leg knockout; draws resolved by penalty coin-flip."""
    hw, d, aw = prob_table[(team_a, team_b)]
    r = rng.random()
    if r < hw:
        return team_a
    elif r < hw + d:
        # Penalties: 50/50
        return team_a if rng.random() < 0.5 else team_b
    else:
        return team_b


def _play_round(
    pairs: list[tuple[str, str]],
    prob_table: dict[tuple[str, str], tuple[float, float, float]],
    rng: np.random.Generator,
) -> tuple[list[str], list[tuple[str, str]]]:
    winners = [_knockout_match(a, b, prob_table, rng) for a, b in pairs]
    next_pairs = [(winners[i], winners[i + 1]) for i in range(0, len(winners) - 1, 2)]
    return winners, next_pairs


def simulate_knockout_stage(
    group_standings: dict[str, list[GroupResult]],
    prob_table: dict[tuple[str, str], tuple[float, float, float]],
    rng: np.random.Generator,
    elo: EloRatings,
) -> dict[str, Any]:
    """R32 → R16 → QF → SF → Final."""
    groups = sorted(group_standings.keys())
    group_winners = [group_standings[g][0].team for g in groups]
    runners_up    = [group_standings[g][1].team for g in groups]
    best_thirds   = _select_best_third_place(group_standings, n=8)

    r32_pairs = _seed_round_of_32_pairs(group_winners, runners_up, best_thirds, elo)

    r32_winners, r16_pairs  = _play_round(r32_pairs, prob_table, rng)
    r16_winners, qf_pairs   = _play_round(r16_pairs, prob_table, rng)
    qf_winners,  sf_pairs   = _play_round(qf_pairs,  prob_table, rng)
    sf_winners,  _          = _play_round(sf_pairs,  prob_table, rng)

    sf_losers = [
        b if sf_winners[i] == a else a
        for i, (a, b) in enumerate(sf_pairs)
    ]
    third_place = (
        _knockout_match(sf_losers[0], sf_losers[1], prob_table, rng)
        if len(sf_losers) == 2 else None
    )
    champion  = _knockout_match(sf_winners[0], sf_winners[1], prob_table, rng)
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]

    def _to_dict(pairs, winners):
        return {f"{a} vs {b}": w for (a, b), w in zip(pairs, winners)}

    return {
        "round_of_32":   _to_dict(r32_pairs, r32_winners),
        "round_of_16":   _to_dict(r16_pairs, r16_winners),
        "quarterfinals": _to_dict(qf_pairs,  qf_winners),
        "semifinals":    _to_dict(sf_pairs,  sf_winners),
        "third_place":   third_place,
        "runner_up":     runner_up,
        "champion":      champion,
    }


# ---------------------------------------------------------------------------
# Monte Carlo entry point
# ---------------------------------------------------------------------------

def run_monte_carlo(
    groups_df: pd.DataFrame,
    predictor: MatchPredictor,
    n_simulations: int = 10_000,
    seed: int | None = 42,
    completed_results_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run `n_simulations` full 2026 World Cup simulations.

    Pre-computes all pairwise probabilities once, then the inner loop is
    O(n_simulations × n_matches) with only dict look-ups and rng.random() calls.
    """
    rng = np.random.default_rng(seed)
    all_teams = groups_df["team"].tolist()

    # ── Pre-compute once (outside the hot loop) ──────────────────────────────
    # Group structure as plain dict — avoids pandas groupby on every iteration
    group_structure: dict[str, list[str]] = {
        group: gdf["team"].tolist()
        for group, gdf in groups_df.groupby("group")
    }
    # Pairwise outcome + expected-goals tables (~250 ms for 48 teams)
    prob_table, goals_table = _build_prob_table(all_teams, predictor)
    completed_results = (
        build_locked_group_results(completed_results_df)
        if completed_results_df is not None
        else {}
    )
    # ─────────────────────────────────────────────────────────────────────────

    counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stage_keys = [
        "group_winner", "group_advance", "round_of_32", "round_of_16",
        "quarterfinal", "semifinal", "final", "champion",
    ]

    for _ in range(n_simulations):
        standings = simulate_group_stage(
            group_structure,
            prob_table,
            goals_table,
            rng,
            completed_results=completed_results,
        )
        knockout  = simulate_knockout_stage(standings, prob_table, rng, predictor.elo)

        # Group advancement: top 2 from each group + 8 best 3rd-place
        for ranked in standings.values():
            counters[ranked[0].team]["group_winner"] += 1
            for r in ranked[:2]:
                counters[r.team]["group_advance"] += 1
        for t in _select_best_third_place(standings, n=8):
            counters[t]["group_advance"] += 1

        for stage, key in [
            ("round_of_32",   "round_of_32"),
            ("round_of_16",   "round_of_16"),
            ("quarterfinals", "quarterfinal"),
            ("semifinals",    "semifinal"),
        ]:
            for matchup in knockout[stage]:
                a, b = matchup.split(" vs ")
                counters[a][key] += 1
                counters[b][key] += 1

        counters[knockout["runner_up"]]["final"]    += 1
        counters[knockout["champion"]]["final"]     += 1
        counters[knockout["champion"]]["champion"]  += 1

    results = {
        team: {key: round(counters[team][key] / n_simulations, 4) for key in stage_keys}
        for team in all_teams
    }

    return {
        "probabilities":  results,
        "n_simulations":  n_simulations,
        "top_contenders": sorted(
            results.items(), key=lambda x: x[1]["champion"], reverse=True
        )[:8],
    }
