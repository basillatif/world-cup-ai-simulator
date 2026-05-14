"""Ensemble match predictor: blends ELO and Poisson model outputs."""

import numpy as np

from src.models.elo import EloRatings
from src.models.poisson_model import PoissonModel


class MatchPredictor:
    """
    Blends ELO win-probability with Poisson outcome probabilities.

    ELO is reliable for long-run relative strength; Poisson is better at
    capturing goal-level variance.  We weight them equally by default.
    """

    def __init__(
        self,
        elo: EloRatings,
        poisson: PoissonModel,
        elo_weight: float = 0.45,
        poisson_weight: float = 0.55,
    ):
        self.elo = elo
        self.poisson = poisson
        self.elo_weight = elo_weight
        self.poisson_weight = poisson_weight

    def predict_probs(
        self, home: str, away: str, neutral: bool = True
    ) -> dict[str, float]:
        """Return blended (home_win, draw, away_win) as a dict."""
        home_elo = self.elo.get(home)
        away_elo = self.elo.get(away)
        elo_hw, elo_d, elo_aw = self.elo.win_probability(home_elo, away_elo, neutral)
        poi_hw, poi_d, poi_aw = self.poisson.outcome_probs(home, away, neutral)

        w_e, w_p = self.elo_weight, self.poisson_weight
        hw = w_e * elo_hw + w_p * poi_hw
        d  = w_e * elo_d  + w_p * poi_d
        aw = w_e * elo_aw + w_p * poi_aw
        total = hw + d + aw

        return {
            "home_win": round(hw / total, 4),
            "draw":     round(d  / total, 4),
            "away_win": round(aw / total, 4),
        }

    def simulate_match(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """
        Simulate one match.  Returns the scoreline and outcome label.
        Scoreline is drawn from the Poisson model; outcome uses blended probs
        to correct for ELO information the Poisson may underweight.
        """
        rng = rng or np.random.default_rng()
        probs = self.predict_probs(home, away, neutral)

        # Resample outcome using blended probabilities (re-normalise to guard float rounding)
        raw = np.array([probs["home_win"], probs["draw"], probs["away_win"]], dtype=float)
        raw /= raw.sum()
        outcome = rng.choice(["home_win", "draw", "away_win"], p=raw)

        # Sample a compatible scoreline from Poisson
        while True:
            hg, ag = self.poisson.simulate_score(home, away, neutral, rng)
            simulated_outcome = (
                "home_win" if hg > ag else ("draw" if hg == ag else "away_win")
            )
            if simulated_outcome == outcome:
                break  # accept scoreline consistent with blended outcome

        return {
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "outcome": outcome,
            "home_win_prob": probs["home_win"],
            "draw_prob": probs["draw"],
            "away_win_prob": probs["away_win"],
        }

    def expected_goals_display(self, home: str, away: str, neutral: bool = True) -> dict:
        lam_h, lam_a = self.poisson.expected_goals(home, away, neutral)
        return {"home_xg": round(lam_h, 2), "away_xg": round(lam_a, 2)}
