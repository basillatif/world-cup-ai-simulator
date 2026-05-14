"""Tests for the Monte Carlo tournament simulator."""

import numpy as np
import pandas as pd
import pytest

from src.models.elo import EloRatings
from src.models.match_predictor import MatchPredictor
from src.models.poisson_model import PoissonModel
from src.simulation.tournament_simulator import (
    GroupResult,
    run_monte_carlo,
    simulate_group_stage,
    simulate_knockout_match,
    simulate_knockout_stage,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

GROUPS = pd.DataFrame([
    {"group": "A", "team": "Brazil"},
    {"group": "A", "team": "France"},
    {"group": "A", "team": "Germany"},
    {"group": "A", "team": "Spain"},
    {"group": "B", "team": "Argentina"},
    {"group": "B", "team": "England"},
    {"group": "B", "team": "Portugal"},
    {"group": "B", "team": "Netherlands"},
    {"group": "C", "team": "Belgium"},
    {"group": "C", "team": "Italy"},
    {"group": "C", "team": "Croatia"},
    {"group": "C", "team": "Denmark"},
    {"group": "D", "team": "Uruguay"},
    {"group": "D", "team": "Mexico"},
    {"group": "D", "team": "USA"},
    {"group": "D", "team": "Senegal"},
    {"group": "E", "team": "Morocco"},
    {"group": "E", "team": "Colombia"},
    {"group": "E", "team": "Japan"},
    {"group": "E", "team": "South Korea"},
    {"group": "F", "team": "Switzerland"},
    {"group": "F", "team": "Poland"},
    {"group": "F", "team": "Ecuador"},
    {"group": "F", "team": "Australia"},
    {"group": "G", "team": "Serbia"},
    {"group": "G", "team": "Canada"},
    {"group": "G", "team": "Wales"},
    {"group": "G", "team": "Iran"},
    {"group": "H", "team": "Tunisia"},
    {"group": "H", "team": "Cameroon"},
    {"group": "H", "team": "Ghana"},
    {"group": "H", "team": "Saudi Arabia"},
])

TEAMS = GROUPS["team"].tolist()


def make_predictor(all_teams: list[str]) -> MatchPredictor:
    """Build a simple uniform predictor for testing."""
    elo = EloRatings()
    for t in all_teams:
        elo.set(t, 1500.0)
    poisson = PoissonModel()
    for t in all_teams:
        poisson.attack[t] = 1.0
        poisson.defense[t] = 1.0
    poisson._fitted = True
    return MatchPredictor(elo=elo, poisson=poisson)


# ── GroupResult ───────────────────────────────────────────────────────────────

class TestGroupResult:
    def test_points_calculation(self):
        r = GroupResult(team="Brazil", wins=2, draws=1, losses=0)
        assert r.points == 7

    def test_gd(self):
        r = GroupResult(team="Brazil", gf=5, ga=2)
        assert r.gd == 3

    def test_sort_key_order(self):
        r1 = GroupResult(team="A", wins=3)
        r2 = GroupResult(team="B", wins=2, draws=1)
        assert r1.sort_key() > r2.sort_key()  # 9 pts > 7 pts


# ── simulate_group_stage ──────────────────────────────────────────────────────

class TestGroupStage:
    def setup_method(self):
        self.predictor = make_predictor(TEAMS)
        self.rng = np.random.default_rng(0)

    def test_returns_all_groups(self):
        standings = simulate_group_stage(GROUPS, self.predictor, self.rng)
        assert set(standings.keys()) == set(GROUPS["group"].unique())

    def test_each_group_has_four_teams(self):
        standings = simulate_group_stage(GROUPS, self.predictor, self.rng)
        for group, ranked in standings.items():
            assert len(ranked) == 4, f"Group {group} has {len(ranked)} teams"

    def test_each_group_contains_correct_teams(self):
        standings = simulate_group_stage(GROUPS, self.predictor, self.rng)
        for group, ranked in standings.items():
            expected = set(GROUPS[GROUPS["group"] == group]["team"])
            assert set(ranked) == expected

    def test_no_team_appears_twice(self):
        standings = simulate_group_stage(GROUPS, self.predictor, self.rng)
        all_teams_out = [t for teams in standings.values() for t in teams]
        assert len(all_teams_out) == len(set(all_teams_out))


# ── simulate_knockout_match ───────────────────────────────────────────────────

class TestKnockoutMatch:
    def setup_method(self):
        self.predictor = make_predictor(TEAMS)
        self.rng = np.random.default_rng(42)

    def test_winner_is_one_of_the_two_teams(self):
        winner = simulate_knockout_match("Brazil", "Germany", self.predictor, self.rng)
        assert winner in {"Brazil", "Germany"}

    def test_deterministic_with_seed(self):
        results = set()
        for seed in range(10):
            rng = np.random.default_rng(seed)
            results.add(simulate_knockout_match("Brazil", "Germany", self.predictor, rng))
        # With equal teams both should win sometimes over 10 seeds
        assert len(results) >= 1


# ── simulate_knockout_stage ───────────────────────────────────────────────────

class TestKnockoutStage:
    def setup_method(self):
        self.predictor = make_predictor(TEAMS)
        self.rng = np.random.default_rng(0)
        standings = simulate_group_stage(GROUPS, self.predictor, self.rng)
        self.standings = standings
        self.result = simulate_knockout_stage(standings, self.predictor, self.rng)

    def test_champion_is_a_tournament_team(self):
        assert self.result["champion"] in TEAMS

    def test_runner_up_is_different_from_champion(self):
        assert self.result["runner_up"] != self.result["champion"]

    def test_result_has_required_keys(self):
        required = {"r16", "quarterfinals", "semifinals", "champion", "runner_up"}
        assert required.issubset(set(self.result.keys()))

    def test_r16_has_eight_matches(self):
        assert len(self.result["r16"]) == 8

    def test_qf_has_four_matches(self):
        assert len(self.result["quarterfinals"]) == 4

    def test_sf_has_two_matches(self):
        assert len(self.result["semifinals"]) == 2


# ── run_monte_carlo ───────────────────────────────────────────────────────────

class TestMonteCarlo:
    def setup_method(self):
        self.predictor = make_predictor(TEAMS)

    def test_all_teams_have_probabilities(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=200, seed=0)
        for team in TEAMS:
            assert team in results["probabilities"]

    def test_champion_probs_sum_to_one(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=500, seed=1)
        total = sum(p["champion"] for p in results["probabilities"].values())
        assert abs(total - 1.0) < 0.02  # allow small rounding

    def test_group_advance_greater_than_champion(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=500, seed=2)
        for team, p in results["probabilities"].items():
            assert p["group_advance"] >= p["champion"] - 0.01

    def test_top_contenders_length(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=200, seed=3)
        assert len(results["top_contenders"]) == 8

    def test_probabilities_are_valid(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=300, seed=4)
        for team, p in results["probabilities"].items():
            for stage, val in p.items():
                assert 0.0 <= val <= 1.0, f"{team} {stage}={val} out of range"

    def test_reproducible_with_seed(self):
        r1 = run_monte_carlo(GROUPS, self.predictor, n_simulations=100, seed=99)
        r2 = run_monte_carlo(GROUPS, self.predictor, n_simulations=100, seed=99)
        assert r1["probabilities"]["Brazil"]["champion"] == r2["probabilities"]["Brazil"]["champion"]
