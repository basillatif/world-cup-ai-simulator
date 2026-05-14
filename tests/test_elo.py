"""Tests for the ELO rating system."""

import pytest

from src.models.elo import EloRatings, HOME_ADVANTAGE


class TestEloExpectedScore:
    def test_equal_teams(self):
        elo = EloRatings()
        score = elo.expected_score(1500, 1500)
        assert abs(score - 0.5) < 1e-9

    def test_higher_elo_favoured(self):
        elo = EloRatings()
        assert elo.expected_score(1600, 1500) > 0.5
        assert elo.expected_score(1400, 1500) < 0.5

    def test_symmetry(self):
        elo = EloRatings()
        a = elo.expected_score(1700, 1500)
        b = elo.expected_score(1500, 1700)
        assert abs(a + b - 1.0) < 1e-9


class TestEloWinProbability:
    def test_probs_sum_to_one(self):
        elo = EloRatings()
        hw, d, aw = elo.win_probability(1800, 1600)
        assert abs(hw + d + aw - 1.0) < 1e-6

    def test_favourite_most_likely(self):
        elo = EloRatings()
        hw, d, aw = elo.win_probability(2000, 1600)
        assert hw > d
        assert hw > aw

    def test_home_advantage_shifts_probs(self):
        elo = EloRatings()
        hw_neutral, _, _ = elo.win_probability(1500, 1500, neutral=True)
        hw_home, _, _ = elo.win_probability(1500, 1500, neutral=False)
        assert hw_home > hw_neutral

    def test_all_probs_positive(self):
        elo = EloRatings()
        for hw, d, aw in [
            elo.win_probability(2100, 1500),
            elo.win_probability(1500, 2100),
            elo.win_probability(1800, 1800),
        ]:
            assert hw > 0 and d > 0 and aw > 0


class TestEloUpdate:
    def test_winner_rating_increases(self):
        elo = EloRatings()
        elo.set("A", 1500)
        elo.set("B", 1500)
        new_a, new_b = elo.update("A", "B", 2, 0)
        assert new_a > 1500
        assert new_b < 1500

    def test_ratings_conserved(self):
        elo = EloRatings()
        elo.set("A", 1600)
        elo.set("B", 1400)
        old_sum = 1600 + 1400
        new_a, new_b = elo.update("A", "B", 1, 0)
        assert abs(new_a + new_b - old_sum) < 1e-6

    def test_draw_partial_update(self):
        elo = EloRatings()
        elo.set("A", 1600)
        elo.set("B", 1400)
        new_a, new_b = elo.update("A", "B", 1, 1)
        # Favourite (A) slightly penalised for only drawing
        assert new_a < 1600

    def test_world_cup_k_higher_than_friendly(self):
        elo1 = EloRatings()
        elo1.set("A", 1500)
        elo1.set("B", 1500)
        na_wc, _ = elo1.update("A", "B", 3, 0, tournament="FIFA World Cup")

        elo2 = EloRatings()
        elo2.set("A", 1500)
        elo2.set("B", 1500)
        na_fr, _ = elo2.update("A", "B", 3, 0, tournament="International Friendly")

        assert na_wc > na_fr  # larger K → larger update

    def test_goal_difference_multiplier(self):
        elo1 = EloRatings()
        elo1.set("A", 1500); elo1.set("B", 1500)
        na_1goal, _ = elo1.update("A", "B", 1, 0)

        elo2 = EloRatings()
        elo2.set("A", 1500); elo2.set("B", 1500)
        na_4goals, _ = elo2.update("A", "B", 4, 0)

        assert na_4goals > na_1goal


class TestEloFit:
    def test_fit_updates_ratings(self):
        import pandas as pd
        elo = EloRatings()
        elo.set("Brazil", 1500)
        elo.set("Germany", 1500)
        matches = pd.DataFrame([
            {"date": "2022-01-01", "home_team": "Brazil", "away_team": "Germany",
             "home_goals": 2, "away_goals": 0, "tournament": "International Friendly", "neutral": False},
            {"date": "2022-06-01", "home_team": "Germany", "away_team": "Brazil",
             "home_goals": 1, "away_goals": 1, "tournament": "International Friendly", "neutral": False},
        ])
        elo.fit(matches)
        # Brazil won first match — should be ahead overall
        assert elo.get("Brazil") != 1500 or elo.get("Germany") != 1500
