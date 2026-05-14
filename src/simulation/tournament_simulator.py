"""Monte Carlo World Cup tournament simulator."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.match_predictor import MatchPredictor


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

    def sort_key(self) -> tuple:
        return (self.points, self.gd, self.gf)


def simulate_group_stage(
    groups_df: pd.DataFrame,
    predictor: MatchPredictor,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """
    Simulate the full group stage.
    Returns a dict mapping group label → [1st place team, 2nd place team, 3rd, 4th].
    """
    standings: dict[str, list[str]] = {}

    for group, gdf in groups_df.groupby("group"):
        teams = gdf["team"].tolist()
        table: dict[str, GroupResult] = {t: GroupResult(team=t) for t in teams}

        # Round-robin fixtures
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                result = predictor.simulate_match(t1, t2, neutral=True, rng=rng)
                _update_table(table, result)

        # Sort: points → GD → GF → random tiebreak
        ranked = sorted(
            table.values(),
            key=lambda r: (r.points, r.gd, r.gf, random.random()),
            reverse=True,
        )
        standings[group] = [r.team for r in ranked]

    return standings


def _update_table(table: dict[str, GroupResult], result: dict) -> None:
    h, a = result["home_team"], result["away_team"]
    hg, ag = result["home_goals"], result["away_goals"]

    table[h].played += 1
    table[a].played += 1
    table[h].gf += hg
    table[h].ga += ag
    table[a].gf += ag
    table[a].ga += hg

    if hg > ag:
        table[h].wins += 1
        table[a].losses += 1
    elif hg == ag:
        table[h].draws += 1
        table[a].draws += 1
    else:
        table[a].wins += 1
        table[h].losses += 1


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    predictor: MatchPredictor,
    rng: np.random.Generator,
) -> str:
    """Simulate a single-leg knockout tie; resolve draws via penalties."""
    result = predictor.simulate_match(team_a, team_b, neutral=True, rng=rng)
    if result["outcome"] == "home_win":
        return team_a
    elif result["outcome"] == "away_win":
        return team_b
    else:
        # Coin flip penalty shootout (each team ~50% from the spot)
        return rng.choice([team_a, team_b])


def simulate_knockout_stage(
    group_standings: dict[str, list[str]],
    predictor: MatchPredictor,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Simulate R16 → QF → SF → Final.
    Bracket order mirrors the standard 2022 World Cup draw format.
    """
    groups = sorted(group_standings.keys())

    def top(g): return group_standings[g][0]
    def second(g): return group_standings[g][1]

    # Standard 2022-style R16 pairings
    r16_pairs = [
        (top("A"), second("B")),
        (top("C"), second("D")),
        (top("B"), second("A")),
        (top("D"), second("C")),
        (top("E"), second("F")),
        (top("G"), second("H")),
        (top("F"), second("E")),
        (top("H"), second("G")),
    ]

    def play_round(pairs):
        return [
            simulate_knockout_match(a, b, predictor, rng)
            for a, b in pairs
        ]

    r16_winners = play_round(r16_pairs)
    qf_pairs = list(zip(r16_winners[::2], r16_winners[1::2]))
    qf_winners = play_round(qf_pairs)
    sf_pairs = list(zip(qf_winners[::2], qf_winners[1::2]))
    sf_winners = play_round(sf_pairs)
    sf_losers = [qf_winners[i * 2 + j] for i, (_, _) in enumerate(sf_pairs)
                 for j, w in enumerate([sf_winners[i]])
                 if (qf_winners[i * 2], qf_winners[i * 2 + 1])[j] != w]

    # Third place play-off
    third_place = simulate_knockout_match(sf_losers[0], sf_losers[1], predictor, rng) if len(sf_losers) == 2 else None
    champion = simulate_knockout_match(sf_winners[0], sf_winners[1], predictor, rng)
    runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]

    return {
        "r16": dict(zip([f"{a} vs {b}" for a, b in r16_pairs], r16_winners)),
        "quarterfinals": dict(zip([f"{a} vs {b}" for a, b in qf_pairs], qf_winners)),
        "semifinals": dict(zip([f"{a} vs {b}" for a, b in sf_pairs], sf_winners)),
        "third_place": third_place,
        "runner_up": runner_up,
        "champion": champion,
    }


def run_monte_carlo(
    groups_df: pd.DataFrame,
    predictor: MatchPredictor,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> dict[str, Any]:
    """
    Run `n_simulations` full tournament simulations.

    Returns aggregated probabilities for:
    - Winning the tournament
    - Reaching the final
    - Reaching the semis / quarters / round of 16
    - Advancing from group stage
    """
    rng = np.random.default_rng(seed)

    counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_teams = groups_df["team"].tolist()

    stage_keys = ["group_advance", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]

    for sim in range(n_simulations):
        standings = simulate_group_stage(groups_df, predictor, rng)
        knockout = simulate_knockout_stage(standings, predictor, rng)

        # Group advancement
        for g_teams in standings.values():
            for t in g_teams[:2]:
                counters[t]["group_advance"] += 1

        # Knockout rounds — infer participation from result keys
        for matchup, winner in knockout["r16"].items():
            a, b = matchup.split(" vs ")
            counters[a]["round_of_16"] += 1
            counters[b]["round_of_16"] += 1

        for matchup, winner in knockout["quarterfinals"].items():
            a, b = matchup.split(" vs ")
            counters[a]["quarterfinal"] += 1
            counters[b]["quarterfinal"] += 1

        for matchup, winner in knockout["semifinals"].items():
            a, b = matchup.split(" vs ")
            counters[a]["semifinal"] += 1
            counters[b]["semifinal"] += 1

        counters[knockout["runner_up"]]["final"] += 1
        counters[knockout["champion"]]["final"] += 1
        counters[knockout["champion"]]["champion"] += 1

    # Convert to probabilities
    results = {}
    for team in all_teams:
        tc = counters[team]
        results[team] = {
            key: round(tc[key] / n_simulations, 4)
            for key in stage_keys
        }

    return {
        "probabilities": results,
        "n_simulations": n_simulations,
        "top_contenders": sorted(
            results.items(), key=lambda x: x[1]["champion"], reverse=True
        )[:8],
    }
