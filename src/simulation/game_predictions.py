"""Deterministic per-match predictions for upcoming World Cup fixtures.

This module generates model-based predictions for every scheduled (not yet
played) group-stage match.  It is intentionally free of randomness: results
are derived directly from the MatchPredictor ensemble rather than from a
Monte Carlo draw, so identical inputs always produce identical outputs.
"""

from __future__ import annotations

import pandas as pd

from src.models.match_predictor import MatchPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confidence_tier(home_win: float, away_win: float) -> str:
    """Return High / Medium / Low based on the favourite's win probability."""
    favourite_prob = max(home_win, away_win)
    if favourite_prob >= 0.60:
        return "High"
    if favourite_prob >= 0.50:
        return "Medium"
    return "Low"


def _predicted_scoreline(xg_home: float, xg_away: float) -> str:
    """Round expected-goals values to an integer scoreline string."""
    return f"{max(0, round(xg_home))}–{max(0, round(xg_away))}"


def _short_explanation(
    team_a: str,
    team_b: str,
    probs: dict[str, float],
    teams_df: pd.DataFrame,
) -> str:
    """Return a one-sentence, model-grounded explanation for this match."""
    if teams_df is None or teams_df.empty or "team" not in teams_df.columns:
        hw, d, aw = probs["home_win"], probs["draw"], probs["away_win"]
        if d >= hw and d >= aw:
            return "The model sees this as a tightly contested match where a draw is the most likely single outcome."
        favourite = team_a if hw >= aw else team_b
        return f"The model gives {favourite} the advantage in this fixture."

    a_row = teams_df[teams_df["team"] == team_a]
    b_row = teams_df[teams_df["team"] == team_b]

    hw, d, aw = probs["home_win"], probs["draw"], probs["away_win"]

    if not a_row.empty and not b_row.empty:
        a_elo = float(a_row.iloc[0].get("elo_rating", 1500))
        b_elo = float(b_row.iloc[0].get("elo_rating", 1500))
        gap = abs(a_elo - b_elo)
        stronger = team_a if a_elo >= b_elo else team_b
        weaker = team_b if a_elo >= b_elo else team_a

        if gap < 30:
            return (
                "The Elo ratings are nearly identical; either result is plausible "
                "and a draw would be no surprise."
            )
        if gap < 80:
            return (
                f"{stronger} holds a modest Elo edge over {weaker} — "
                "expect a competitive match with a narrow favourite."
            )
        if gap < 150:
            return (
                f"{stronger} is the model's clear favourite, carrying a notable "
                f"Elo rating advantage over {weaker}."
            )
        return (
            f"{stronger} is a heavy favourite; their Elo rating is "
            f"substantially higher than {weaker}'s going into this match."
        )

    # Fallback when team metadata is unavailable
    if d >= hw and d >= aw:
        return (
            "The model sees this as a tightly contested match "
            "where a draw is the most likely single outcome."
        )
    favourite = team_a if hw >= aw else team_b
    return f"The model gives {favourite} the advantage in this fixture."


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_upcoming_matches(
    results_df: pd.DataFrame,
    predictor: MatchPredictor,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame of model predictions for every scheduled match.

    Completed matches (status == "Final") are excluded automatically.
    If ``results_df`` is empty or missing required columns the function
    returns an empty DataFrame rather than raising.

    Parameters
    ----------
    results_df:
        Full results table containing both "Final" and "Scheduled" rows.
        Expected columns: date, stage, group, team_a, team_b, status.
    predictor:
        Fitted MatchPredictor used to obtain win/draw probabilities and
        expected-goals values.
    teams_df:
        Teams metadata (elo_rating etc.) used for the explanation text.

    Returns
    -------
    DataFrame with columns:
        date, group, team_a, team_b,
        team_a_win, draw, team_b_win,   (0-1 floats, summing to 1)
        scoreline, confidence, explanation
    sorted ascending by date.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    required = {"team_a", "team_b", "status"}
    if not required.issubset(results_df.columns):
        return pd.DataFrame()

    upcoming = results_df[
        results_df["status"].astype(str).str.casefold() == "scheduled"
    ].copy()

    if upcoming.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, match in upcoming.iterrows():
        team_a = str(match.get("team_a", "") or "").strip()
        team_b = str(match.get("team_b", "") or "").strip()
        if not team_a or not team_b:
            continue

        try:
            probs = predictor.predict_probs(team_a, team_b, neutral=True)
            xg = predictor.expected_goals_display(team_a, team_b, neutral=True)
        except Exception:
            # Unknown team or unfitted model — fall back to equal probabilities.
            probs = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
            xg = {"home_xg": 1.0, "away_xg": 1.0}

        hw = probs["home_win"]
        d = probs["draw"]
        aw = probs["away_win"]
        # Renormalize after rounding to guarantee exact sum of 1.0.
        total = hw + d + aw
        if total > 0:
            hw, d, aw = hw / total, d / total, aw / total

        rows.append(
            {
                "date": match.get("date"),
                "group": str(match.get("group", "") or ""),
                "team_a": team_a,
                "team_b": team_b,
                "team_a_win": hw,
                "draw": d,
                "team_b_win": aw,
                "scoreline": _predicted_scoreline(xg["home_xg"], xg["away_xg"]),
                "confidence": _confidence_tier(hw, aw),
                "explanation": _short_explanation(team_a, team_b, probs, teams_df),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date", na_position="last").reset_index(drop=True)
