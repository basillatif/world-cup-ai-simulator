"""Knockout match engine wired to the existing World Cup model stack."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_matches, load_results, load_teams
from src.models.elo import build_elo_from_seed
from src.models.match_predictor import MatchPredictor
from src.models.poisson_model import build_poisson_from_teams


@lru_cache(maxsize=1)
def _current_predictor() -> MatchPredictor:
    teams = load_teams()
    matches = load_matches()
    results = load_results(Path(__file__).parent / "data" / "sample" / "results.csv")
    completed = results[results["status"].astype(str).str.casefold() == "final"].rename(
        columns={
            "team_a": "home_team",
            "team_b": "away_team",
            "score_a": "home_goals",
            "score_b": "away_goals",
        }
    )
    completed["tournament"] = "FIFA World Cup 2026"
    completed["neutral"] = True
    model_matches = pd.concat(
        [matches, completed[matches.columns]],
        ignore_index=True,
    )
    elo = build_elo_from_seed(teams, model_matches)
    poisson = build_poisson_from_teams(teams)
    if len(model_matches) > 20:
        try:
            poisson.fit(model_matches)
        except Exception:
            pass
    return MatchPredictor(elo=elo, poisson=poisson)


@lru_cache(maxsize=None)
def _pair_model(team_a: str, team_b: str) -> tuple[float, float, float, float, float]:
    predictor = _current_predictor()
    probs = predictor.predict_probs(team_a, team_b, neutral=True)
    lam_a, lam_b = predictor.poisson.expected_goals(team_a, team_b, neutral=True)
    return probs["home_win"], probs["draw"], probs["away_win"], lam_a, lam_b


def _sample_group_style_score(
    outcome: str,
    lam_a: float,
    lam_b: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    while True:
        goals_a = int(rng.poisson(lam_a))
        goals_b = int(rng.poisson(lam_b))
        simulated_outcome = (
            "home_win" if goals_a > goals_b else ("draw" if goals_a == goals_b else "away_win")
        )
        if simulated_outcome == outcome:
            return goals_a, goals_b


def knockout_match_fn(team_a: str, team_b: str, rng: np.random.Generator) -> str:
    home_win, draw, away_win, lam_a, lam_b = _pair_model(team_a, team_b)
    raw = np.array([home_win, draw, away_win], dtype=float)
    raw /= raw.sum()
    outcome = str(rng.choice(["home_win", "draw", "away_win"], p=raw))
    goals_a, goals_b = _sample_group_style_score(outcome, lam_a, lam_b, rng)
    if goals_a > goals_b:
        return team_a
    if goals_b > goals_a:
        return team_b

    extra_a = int(rng.poisson(lam_a / 3.0))
    extra_b = int(rng.poisson(lam_b / 3.0))
    if extra_a > extra_b:
        return team_a
    if extra_b > extra_a:
        return team_b

    return team_a if rng.random() < 0.5 else team_b
