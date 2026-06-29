"""Helpers for combining actual World Cup results with future simulations."""

from __future__ import annotations

from typing import Any

import pandas as pd


RESULT_COLUMNS = [
    "date",
    "stage",
    "group",
    "team_a",
    "team_b",
    "score_a",
    "score_b",
    "winner",
    "status",
]

STANDINGS_COLUMNS = [
    "group",
    "team",
    "played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_diff",
    "points",
]


def normalize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return tournament results in the canonical live-tracker schema."""
    df = results_df.copy()
    legacy_columns = {
        "home_team": "team_a",
        "away_team": "team_b",
        "home_goals": "score_a",
        "away_goals": "score_b",
    }
    df = df.rename(columns={k: v for k, v in legacy_columns.items() if k in df})

    if "stage" not in df:
        df["stage"] = "Group"
    if "status" not in df:
        df["status"] = "Final"
    if "winner" not in df:
        df["winner"] = pd.NA

    missing = set(RESULT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"results data missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    final_mask = df["status"].astype(str).str.casefold().eq("final")
    for score_column in ("score_a", "score_b"):
        df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    if df.loc[final_mask, ["score_a", "score_b"]].isna().any().any():
        raise ValueError("Final results must include numeric scores")

    calculated_winner = df["team_a"].where(
        df["score_a"] > df["score_b"],
        df["team_b"].where(df["score_b"] > df["score_a"], "Draw"),
    )
    missing_winner = df["winner"].isna() | df["winner"].astype(str).str.strip().eq("")
    fill_mask = final_mask & missing_winner
    df.loc[fill_mask, "winner"] = calculated_winner[fill_mask]
    return df[RESULT_COLUMNS].sort_values("date", na_position="last").reset_index(drop=True)


def calculate_group_standings(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate group tables from completed matches only."""
    results = normalize_results(results_df)
    finals = results[
        results["status"].astype(str).str.casefold().eq("final")
        & results["stage"].astype(str).str.casefold().eq("group")
    ]
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def team_record(group: str, team: str) -> dict[str, Any]:
        key = (group, team)
        if key not in records:
            records[key] = {
                "group": group,
                "team": team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_diff": 0,
                "points": 0,
            }
        return records[key]

    for row in finals.itertuples(index=False):
        team_a = team_record(str(row.group), str(row.team_a))
        team_b = team_record(str(row.group), str(row.team_b))
        score_a, score_b = int(row.score_a), int(row.score_b)

        team_a["played"] += 1
        team_b["played"] += 1
        team_a["goals_for"] += score_a
        team_a["goals_against"] += score_b
        team_b["goals_for"] += score_b
        team_b["goals_against"] += score_a

        if score_a > score_b:
            team_a["wins"] += 1
            team_a["points"] += 3
            team_b["losses"] += 1
        elif score_b > score_a:
            team_b["wins"] += 1
            team_b["points"] += 3
            team_a["losses"] += 1
        else:
            team_a["draws"] += 1
            team_b["draws"] += 1
            team_a["points"] += 1
            team_b["points"] += 1

    if not records:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    standings = pd.DataFrame(records.values())
    standings["goal_diff"] = standings["goals_for"] - standings["goals_against"]
    return standings.sort_values(
        ["group", "points", "goal_diff", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)[STANDINGS_COLUMNS]


def build_locked_group_results(
    results_df: pd.DataFrame,
) -> dict[frozenset[str], tuple[str, str, int, int]]:
    """Build the simulator lookup for completed group matches."""
    results = normalize_results(results_df)
    finals = results[
        results["status"].astype(str).str.casefold().eq("final")
        & results["stage"].astype(str).str.casefold().eq("group")
    ]
    locked: dict[frozenset[str], tuple[str, str, int, int]] = {}
    for row in finals.itertuples(index=False):
        key = frozenset((str(row.team_a), str(row.team_b)))
        if key in locked:
            raise ValueError(f"Duplicate final result for {row.team_a} vs {row.team_b}")
        locked[key] = (
            str(row.team_a),
            str(row.team_b),
            int(row.score_a),
            int(row.score_b),
        )
    return locked


def compare_probabilities(
    baseline: dict[str, dict[str, float]],
    updated: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compare baseline and live group-advance and title probabilities."""
    teams = sorted(set(baseline) | set(updated))
    rows = []
    for team in teams:
        baseline_advance = baseline.get(team, {}).get("group_advance", 0.0)
        updated_advance = updated.get(team, {}).get("group_advance", 0.0)
        baseline_title = baseline.get(team, {}).get("champion", 0.0)
        updated_title = updated.get(team, {}).get("champion", 0.0)
        rows.append(
            {
                "team": team,
                "baseline_advance_prob": baseline_advance,
                "updated_advance_prob": updated_advance,
                "advance_prob_change": updated_advance - baseline_advance,
                "baseline_title_prob": baseline_title,
                "updated_title_prob": updated_title,
                "title_prob_change": updated_title - baseline_title,
            }
        )
    comparison = pd.DataFrame(rows)
    return comparison.iloc[
        comparison["advance_prob_change"].abs().sort_values(ascending=False).index
    ].reset_index(drop=True)
