"""ELO rating system for international football."""

import math
from dataclasses import dataclass, field


# Standard football ELO constants
K_WORLD_CUP = 60
K_CONFEDERATION = 50
K_QUALIFIER = 40
K_FRIENDLY = 20
HOME_ADVANTAGE = 100  # ELO points added to home side in non-neutral venues


TOURNAMENT_K = {
    "FIFA World Cup": K_WORLD_CUP,
    "Copa America": K_CONFEDERATION,
    "UEFA Nations League": K_CONFEDERATION,
    "UEFA Euro": K_CONFEDERATION,
    "African Cup of Nations": K_CONFEDERATION,
    "FIFA World Cup Qualifying": K_QUALIFIER,
    "CONCACAF WCQ": K_QUALIFIER,
    "CAF WCQ": K_QUALIFIER,
    "International Friendly": K_FRIENDLY,
}


@dataclass
class EloRatings:
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str, default: float = 1500.0) -> float:
        return self.ratings.get(team, default)

    def set(self, team: str, rating: float) -> None:
        self.ratings[team] = rating

    def expected_score(self, team_a: float, team_b: float) -> float:
        """Probability that team_a wins (or draws weighted)."""
        return 1.0 / (1.0 + 10 ** ((team_b - team_a) / 400.0))

    def win_probability(self, home_elo: float, away_elo: float, neutral: bool = True) -> tuple[float, float, float]:
        """Return (home_win, draw, away_win) probabilities."""
        adjusted_home = home_elo if neutral else home_elo + HOME_ADVANTAGE
        exp = self.expected_score(adjusted_home, away_elo)

        # Dixon-Coles inspired draw band: the closer the expected score to 0.5, the more draws
        draw_prob = max(0.05, 0.30 - 0.5 * abs(exp - 0.5))
        home_win = exp * (1 - draw_prob)
        away_win = (1 - exp) * (1 - draw_prob)

        total = home_win + draw_prob + away_win
        return home_win / total, draw_prob / total, away_win / total

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        tournament: str = "International Friendly",
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_home_elo, new_away_elo)."""
        home_elo = self.get(home_team)
        away_elo = self.get(away_team)

        adjusted_home = home_elo if neutral else home_elo + HOME_ADVANTAGE
        exp_home = self.expected_score(adjusted_home, away_elo)

        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # Goal difference multiplier (World Football Elo variant)
        gd = abs(home_goals - away_goals)
        if gd == 0 or gd == 1:
            gd_mult = 1.0
        elif gd == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11 + gd) / 8.0

        k = TOURNAMENT_K.get(tournament, K_FRIENDLY)
        delta = k * gd_mult * (actual_home - exp_home)

        new_home = home_elo + delta
        new_away = away_elo - delta

        self.set(home_team, new_home)
        self.set(away_team, new_away)
        return new_home, new_away

    def fit(self, matches_df) -> "EloRatings":
        """Replay historical matches to compute current ratings."""
        for _, row in matches_df.iterrows():
            self.update(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                tournament=row.get("tournament", "International Friendly"),
                neutral=bool(row.get("neutral", False)),
            )
        return self


def build_elo_from_seed(teams_df, matches_df) -> EloRatings:
    """Initialise ratings from teams.csv seed values, then replay matches."""
    elo = EloRatings()
    for _, row in teams_df.iterrows():
        elo.set(row["team"], float(row["elo_rating"]))
    elo.fit(matches_df)
    return elo
