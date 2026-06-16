"""Tests for the game_predictions module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.elo import EloRatings
from src.models.match_predictor import MatchPredictor
from src.models.poisson_model import PoissonModel
from src.simulation.game_predictions import (
    _confidence_tier,
    _predicted_scoreline,
    predict_upcoming_matches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEAMS = ["Alpha", "Beta", "Gamma", "Delta"]
TEAMS_DF = pd.DataFrame(
    [
        {"team": "Alpha", "elo_rating": 2000, "avg_goals_scored": 2.0, "avg_goals_conceded": 1.0},
        {"team": "Beta",  "elo_rating": 1800, "avg_goals_scored": 1.5, "avg_goals_conceded": 1.2},
        {"team": "Gamma", "elo_rating": 1600, "avg_goals_scored": 1.2, "avg_goals_conceded": 1.5},
        {"team": "Delta", "elo_rating": 1500, "avg_goals_scored": 1.0, "avg_goals_conceded": 1.8},
    ]
)


def _make_predictor(teams: list[str] = TEAMS) -> MatchPredictor:
    elo = EloRatings()
    for t in teams:
        row = TEAMS_DF[TEAMS_DF["team"] == t]
        rating = float(row.iloc[0]["elo_rating"]) if not row.empty else 1500.0
        elo.set(t, rating)
    poisson = PoissonModel()
    for t in teams:
        poisson.attack[t] = 1.0
        poisson.defense[t] = 1.0
    poisson._fitted = True
    return MatchPredictor(elo=elo, poisson=poisson)


def _make_results(scheduled: list[dict], final: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    for m in scheduled:
        rows.append(
            {
                "date": m.get("date", "2026-06-20"),
                "stage": "Group",
                "group": m.get("group", "A"),
                "team_a": m["team_a"],
                "team_b": m["team_b"],
                "score_a": None,
                "score_b": None,
                "winner": None,
                "status": "Scheduled",
            }
        )
    for m in (final or []):
        rows.append(
            {
                "date": m.get("date", "2026-06-15"),
                "stage": "Group",
                "group": m.get("group", "A"),
                "team_a": m["team_a"],
                "team_b": m["team_b"],
                "score_a": m.get("score_a", 1),
                "score_b": m.get("score_b", 0),
                "winner": m.get("winner", m["team_a"]),
                "status": "Final",
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------

class TestConfidenceTier:
    def test_high_when_favourite_above_60(self):
        assert _confidence_tier(0.65, 0.20) == "High"

    def test_medium_when_favourite_between_50_and_60(self):
        assert _confidence_tier(0.55, 0.25) == "Medium"

    def test_low_for_close_match(self):
        assert _confidence_tier(0.40, 0.38) == "Low"

    def test_uses_max_of_both_teams(self):
        # Away team is the favourite
        assert _confidence_tier(0.20, 0.70) == "High"


class TestPredictedScoreline:
    def test_rounds_expected_goals(self):
        assert _predicted_scoreline(1.4, 0.6) == "1–1"

    def test_clamps_negative_xg_to_zero(self):
        assert _predicted_scoreline(-0.1, 1.8) == "0–2"

    def test_zero_zero_draw(self):
        assert _predicted_scoreline(0.3, 0.4) == "0–0"


# ---------------------------------------------------------------------------
# Integration: predict_upcoming_matches
# ---------------------------------------------------------------------------

class TestPredictUpcomingMatches:
    def setup_method(self):
        self.predictor = _make_predictor()

    # -- Probability invariants -----------------------------------------------

    def test_probabilities_sum_to_one_for_each_match(self):
        results = _make_results([
            {"team_a": "Alpha", "team_b": "Beta"},
            {"team_a": "Gamma", "team_b": "Delta"},
        ])
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        assert len(df) == 2
        for _, row in df.iterrows():
            total = row["team_a_win"] + row["draw"] + row["team_b_win"]
            assert abs(total - 1.0) < 1e-6, f"Probs sum to {total} for {row['team_a']} vs {row['team_b']}"

    def test_all_probabilities_in_unit_interval(self):
        results = _make_results([{"team_a": "Alpha", "team_b": "Gamma"}])
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        for col in ("team_a_win", "draw", "team_b_win"):
            assert 0.0 <= df.iloc[0][col] <= 1.0

    # -- Completed matches excluded -------------------------------------------

    def test_completed_matches_are_excluded(self):
        results = _make_results(
            scheduled=[{"team_a": "Alpha", "team_b": "Beta"}],
            final=[{"team_a": "Gamma", "team_b": "Delta"}],
        )
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        assert len(df) == 1
        assert set(df["team_a"]) == {"Alpha"}

    def test_all_final_returns_empty(self):
        results = _make_results(
            scheduled=[],
            final=[
                {"team_a": "Alpha", "team_b": "Beta"},
                {"team_a": "Gamma", "team_b": "Delta"},
            ],
        )
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        assert df.empty

    # -- Missing ratings fallback ---------------------------------------------

    def test_missing_teams_use_fallback_and_do_not_crash(self):
        unknown_predictor = _make_predictor()
        results = _make_results([{"team_a": "Zanzibar FC", "team_b": "Atlantis United"}])
        df = predict_upcoming_matches(results, unknown_predictor, pd.DataFrame())
        assert len(df) == 1
        total = df.iloc[0]["team_a_win"] + df.iloc[0]["draw"] + df.iloc[0]["team_b_win"]
        assert abs(total - 1.0) < 1e-6

    # -- Graceful handling of edge cases --------------------------------------

    def test_empty_results_returns_empty(self):
        df = predict_upcoming_matches(pd.DataFrame(), self.predictor, TEAMS_DF)
        assert df.empty

    def test_missing_required_columns_returns_empty(self):
        bad_df = pd.DataFrame([{"foo": "bar"}])
        df = predict_upcoming_matches(bad_df, self.predictor, TEAMS_DF)
        assert df.empty

    def test_no_crash_when_date_is_missing(self):
        results = _make_results([{"team_a": "Alpha", "team_b": "Beta", "date": None}])
        results["date"] = pd.NaT
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        assert len(df) == 1

    def test_output_sorted_by_date_ascending(self):
        results = _make_results([
            {"team_a": "Alpha", "team_b": "Beta",  "date": "2026-06-22"},
            {"team_a": "Gamma", "team_b": "Delta", "date": "2026-06-20"},
        ])
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        dates = df["date"].dropna().tolist()
        assert dates == sorted(dates)

    # -- Required output columns ----------------------------------------------

    def test_output_has_all_required_columns(self):
        results = _make_results([{"team_a": "Alpha", "team_b": "Beta"}])
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        required = {"date", "group", "team_a", "team_b", "team_a_win", "draw",
                    "team_b_win", "scoreline", "confidence", "explanation"}
        assert required.issubset(df.columns)

    def test_confidence_values_are_valid(self):
        results = _make_results([
            {"team_a": "Alpha", "team_b": "Beta"},
            {"team_a": "Gamma", "team_b": "Delta"},
        ])
        df = predict_upcoming_matches(results, self.predictor, TEAMS_DF)
        assert set(df["confidence"]).issubset({"High", "Medium", "Low"})
