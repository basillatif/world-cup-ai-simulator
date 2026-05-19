"""Bivariate Poisson model for predicting match scorelines."""

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


MAX_GOALS = 8  # truncate goal matrix at this value


class PoissonModel:
    """
    Dixon-Coles style attack/defense strength model.

    Parameters
    ----------
    home_advantage : float
        Multiplicative boost applied to the home team's expected goals.
    """

    def __init__(self, home_advantage: float = 1.20):
        self.home_advantage = home_advantage
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches_df: pd.DataFrame, min_matches: int = 3) -> "PoissonModel":
        teams = sorted(
            set(matches_df["home_team"]) | set(matches_df["away_team"])
        )
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Pre-extract arrays once — avoids iterrows() inside the optimizer hot loop
        home_idx   = np.array([idx[t] for t in matches_df["home_team"]], dtype=np.intp)
        away_idx   = np.array([idx[t] for t in matches_df["away_team"]], dtype=np.intp)
        home_goals = matches_df["home_goals"].to_numpy(dtype=int)
        away_goals = matches_df["away_goals"].to_numpy(dtype=int)
        ha_factors = np.where(matches_df["neutral"].to_numpy(dtype=bool), 1.0, self.home_advantage)

        def _log_likelihood(params: np.ndarray) -> float:
            attack  = params[:n]
            defense = params[n:]
            lam_h = np.maximum(ha_factors * attack[home_idx] * defense[away_idx], 1e-6)
            lam_a = np.maximum(attack[away_idx] * defense[home_idx], 1e-6)
            ll = float(np.sum(poisson.logpmf(home_goals, lam_h) + poisson.logpmf(away_goals, lam_a)))
            return -ll

        x0 = np.ones(2 * n)
        constraints = [{"type": "eq", "fun": lambda p: p[:n].mean() - 1.0}]
        bounds = [(0.1, 5.0)] * (2 * n)

        result = minimize(
            _log_likelihood, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        params = result.x
        self.attack  = {t: params[idx[t]]       for t in teams}
        self.defense = {t: params[n + idx[t]]   for t in teams}
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def expected_goals(
        self, home: str, away: str, neutral: bool = True
    ) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) expected goals."""
        a_h = self.attack.get(home, 1.0)
        d_h = self.defense.get(home, 1.0)
        a_a = self.attack.get(away, 1.0)
        d_a = self.defense.get(away, 1.0)
        ha = 1.0 if neutral else self.home_advantage
        lam_h = max(ha * a_h * d_a, 0.05)
        lam_a = max(a_a * d_h, 0.05)
        return lam_h, lam_a

    def score_matrix(
        self, home: str, away: str, neutral: bool = True
    ) -> np.ndarray:
        """(MAX_GOALS+1) x (MAX_GOALS+1) probability matrix P[hg, ag]."""
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        mat = np.outer(
            poisson.pmf(range(MAX_GOALS + 1), lam_h),
            poisson.pmf(range(MAX_GOALS + 1), lam_a),
        )
        return mat / mat.sum()  # renormalize after truncation

    def outcome_probs(
        self, home: str, away: str, neutral: bool = True
    ) -> tuple[float, float, float]:
        """Return (home_win, draw, away_win) probabilities."""
        mat = self.score_matrix(home, away, neutral)
        home_win = float(np.tril(mat, -1).sum())
        draw = float(np.trace(mat))
        away_win = float(np.triu(mat, 1).sum())
        return home_win, draw, away_win

    def simulate_score(
        self, home: str, away: str, neutral: bool = True, rng: np.random.Generator | None = None
    ) -> tuple[int, int]:
        """Sample a single scoreline from the Poisson distribution."""
        rng = rng or np.random.default_rng()
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        return int(rng.poisson(lam_h)), int(rng.poisson(lam_a))


def build_poisson_from_teams(teams_df: pd.DataFrame) -> PoissonModel:
    """
    Construct a PoissonModel seeded from teams.csv when match history is sparse.
    Attack strength ∝ avg_goals_scored; Defense strength ∝ 1/avg_goals_conceded.
    """
    model = PoissonModel()
    mean_scored = teams_df["avg_goals_scored"].mean()
    mean_conceded = teams_df["avg_goals_conceded"].mean()
    for _, row in teams_df.iterrows():
        model.attack[row["team"]] = row["avg_goals_scored"] / mean_scored
        model.defense[row["team"]] = mean_conceded / max(row["avg_goals_conceded"], 0.1)
    model._fitted = True
    return model
