"""Tests for live tournament result and probability helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.simulation.live_tracker import (
    build_locked_group_results,
    calculate_group_standings,
    compare_probabilities,
    normalize_results,
)


def sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "stage": "Group",
                "group": "A",
                "team_a": "Mexico",
                "team_b": "South Africa",
                "score_a": 1,
                "score_b": 0,
                "winner": "Mexico",
                "status": "Final",
            },
            {
                "date": "2026-06-11",
                "stage": "Group",
                "group": "A",
                "team_a": "South Korea",
                "team_b": "Czechia",
                "score_a": 2,
                "score_b": 1,
                "winner": "South Korea",
                "status": "Final",
            },
            {
                "date": "2026-06-17",
                "stage": "Group",
                "group": "A",
                "team_a": "Mexico",
                "team_b": "Czechia",
                "score_a": None,
                "score_b": None,
                "winner": None,
                "status": "Scheduled",
            },
        ]
    )


def test_calculate_group_standings_uses_only_final_results():
    standings = calculate_group_standings(sample_results()).set_index("team")

    assert standings.loc["Mexico", "points"] == 3
    assert standings.loc["South Africa", "points"] == 0
    assert standings.loc["South Korea", "points"] == 3
    assert standings.loc["Czechia", "points"] == 0
    assert standings.loc["Mexico", "played"] == 1
    assert standings.loc["Czechia", "played"] == 1


def test_standings_sort_by_points_goal_difference_and_goals_for():
    standings = calculate_group_standings(sample_results())
    assert standings["team"].tolist()[:2] == ["South Korea", "Mexico"]


def test_locked_results_exclude_scheduled_matches():
    locked = build_locked_group_results(sample_results())

    assert locked[frozenset(("Mexico", "South Africa"))] == (
        "Mexico",
        "South Africa",
        1,
        0,
    )
    assert frozenset(("Mexico", "Czechia")) not in locked
    assert len(locked) == 2


def test_probability_comparison_calculates_and_sorts_deltas():
    baseline = {
        "Mexico": {"group_advance": 0.50, "champion": 0.05},
        "Czechia": {"group_advance": 0.60, "champion": 0.02},
    }
    updated = {
        "Mexico": {"group_advance": 0.75, "champion": 0.08},
        "Czechia": {"group_advance": 0.45, "champion": 0.01},
    }

    comparison = compare_probabilities(baseline, updated).set_index("team")
    assert comparison.loc["Mexico", "advance_prob_change"] == 0.25
    assert comparison.loc["Mexico", "title_prob_change"] == pytest.approx(0.03)
    assert comparison.loc["Czechia", "advance_prob_change"] == pytest.approx(-0.15)


def test_normalize_results_supports_legacy_match_columns():
    legacy = pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "group": "A",
                "home_team": "Mexico",
                "away_team": "South Africa",
                "home_goals": 1,
                "away_goals": 0,
            }
        ]
    )

    normalized = normalize_results(legacy)
    assert normalized.loc[0, "status"] == "Final"
    assert normalized.loc[0, "winner"] == "Mexico"
    assert normalized.loc[0, "team_a"] == "Mexico"
