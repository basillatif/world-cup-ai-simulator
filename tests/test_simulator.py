"""Tests for the 48-team Monte Carlo tournament simulator."""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

from src.models.elo import EloRatings
from src.models.match_predictor import MatchPredictor
from src.models.poisson_model import PoissonModel
from src.simulation.tournament_simulator import (
    GroupResult,
    run_monte_carlo,
    simulate_group_stage,
    simulate_knockout_stage,
)


# -- Fixtures -----------------------------------------------------------------

GROUP_LABELS = list(string.ascii_uppercase[:12])
GROUPS = pd.DataFrame(
    [
        {"group": group, "team": f"{group}{seed}"}
        for group in GROUP_LABELS
        for seed in range(1, 5)
    ]
)
TEAMS = GROUPS["team"].tolist()
GROUP_STRUCTURE = {
    group: gdf["team"].tolist()
    for group, gdf in GROUPS.groupby("group")
}


def make_predictor(all_teams: list[str]) -> MatchPredictor:
    """Build a simple uniform predictor for testing."""
    elo = EloRatings()
    for team in all_teams:
        elo.set(team, 1500.0)

    poisson = PoissonModel()
    for team in all_teams:
        poisson.attack[team] = 1.0
        poisson.defense[team] = 1.0
    poisson._fitted = True

    return MatchPredictor(elo=elo, poisson=poisson)


def make_tables(
    all_teams: list[str],
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float]],
    dict[tuple[str, str], tuple[float, float]],
]:
    """Build explicit pairwise lookup tables for low-level simulator tests."""
    prob_table = {}
    goals_table = {}
    for home in all_teams:
        for away in all_teams:
            if home == away:
                continue
            prob_table[(home, away)] = (0.35, 0.30, 0.35)
            goals_table[(home, away)] = (1.0, 1.0)
    return prob_table, goals_table


# -- GroupResult ---------------------------------------------------------------

class TestGroupResult:
    def test_points_calculation(self):
        result = GroupResult(team="A1", wins=2, draws=1, losses=0)
        assert result.points == 7

    def test_goal_difference_calculation(self):
        result = GroupResult(team="A1", gf=5, ga=2)
        assert result.gd == 3

    def test_current_sorting_uses_points_goal_difference_and_goals_for(self):
        better_points = GroupResult(team="A1", wins=3, gf=3, ga=0)
        worse_points = GroupResult(team="A2", wins=2, draws=1, gf=8, ga=0)
        assert (better_points.points, better_points.gd, better_points.gf) > (
            worse_points.points,
            worse_points.gd,
            worse_points.gf,
        )

        better_goal_difference = GroupResult(team="A3", wins=2, gf=5, ga=1)
        worse_goal_difference = GroupResult(team="A4", wins=2, gf=3, ga=1)
        assert (
            better_goal_difference.points,
            better_goal_difference.gd,
            better_goal_difference.gf,
        ) > (
            worse_goal_difference.points,
            worse_goal_difference.gd,
            worse_goal_difference.gf,
        )


# -- simulate_group_stage ------------------------------------------------------

class TestGroupStage:
    def setup_method(self):
        self.prob_table, self.goals_table = make_tables(TEAMS)
        self.rng = np.random.default_rng(0)

    def simulate(self):
        return simulate_group_stage(
            GROUP_STRUCTURE,
            self.prob_table,
            self.goals_table,
            self.rng,
        )

    def test_returns_all_twelve_groups(self):
        standings = self.simulate()
        assert set(standings.keys()) == set(GROUP_LABELS)

    def test_each_group_has_four_ranked_results(self):
        standings = self.simulate()
        for group, ranked in standings.items():
            assert len(ranked) == 4, f"Group {group} has {len(ranked)} teams"
            assert all(isinstance(result, GroupResult) for result in ranked)

    def test_each_group_contains_correct_teams(self):
        standings = self.simulate()
        for group, ranked in standings.items():
            expected = set(GROUP_STRUCTURE[group])
            actual = {result.team for result in ranked}
            assert actual == expected

    def test_no_team_appears_twice(self):
        standings = self.simulate()
        all_teams_out = [result.team for ranked in standings.values() for result in ranked]
        assert len(all_teams_out) == 48
        assert len(all_teams_out) == len(set(all_teams_out))

    def test_each_team_plays_three_group_matches(self):
        standings = self.simulate()
        for ranked in standings.values():
            assert {result.played for result in ranked} == {3}

    def test_completed_result_is_locked_into_group_table(self):
        completed = {
            frozenset((TEAMS[0], TEAMS[1])): (TEAMS[0], TEAMS[1], 2, 0)
        }
        scoreless_goals = {
            matchup: (0.0, 0.0) for matchup in self.goals_table
        }
        all_draws = {
            matchup: (0.0, 1.0, 0.0) for matchup in self.prob_table
        }
        standings = simulate_group_stage(
            GROUP_STRUCTURE,
            all_draws,
            scoreless_goals,
            np.random.default_rng(0),
            completed_results=completed,
        )
        by_team = {result.team: result for result in standings["A"]}
        assert by_team[TEAMS[0]].gf == 2
        assert by_team[TEAMS[0]].points == 5
        assert by_team[TEAMS[1]].ga == 2
        assert by_team[TEAMS[1]].points == 2


# -- simulate_knockout_stage ---------------------------------------------------

class TestKnockoutStage:
    def setup_method(self):
        self.prob_table, self.goals_table = make_tables(TEAMS)
        rng = np.random.default_rng(0)
        self.standings = simulate_group_stage(
            GROUP_STRUCTURE,
            self.prob_table,
            self.goals_table,
            rng,
        )
        self.result = simulate_knockout_stage(self.standings, self.prob_table, rng)

    def test_champion_is_a_tournament_team(self):
        assert self.result["champion"] in TEAMS

    def test_runner_up_is_different_from_champion(self):
        assert self.result["runner_up"] != self.result["champion"]

    def test_result_has_current_2026_knockout_keys(self):
        required = {
            "round_of_32",
            "round_of_16",
            "quarterfinals",
            "semifinals",
            "third_place",
            "runner_up",
            "champion",
        }
        assert required.issubset(set(self.result.keys()))

    def test_round_of_32_has_sixteen_matches(self):
        assert len(self.result["round_of_32"]) == 16

    def test_round_of_16_has_eight_matches(self):
        assert len(self.result["round_of_16"]) == 8

    def test_quarterfinals_have_four_matches(self):
        assert len(self.result["quarterfinals"]) == 4

    def test_semifinals_have_two_matches(self):
        assert len(self.result["semifinals"]) == 2

    def test_third_place_winner_is_a_semifinal_loser(self):
        semifinalists = {
            team
            for matchup in self.result["semifinals"]
            for team in matchup.split(" vs ")
        }
        finalists = set(self.result["semifinals"].values())
        semifinal_losers = semifinalists - finalists
        assert self.result["third_place"] in semifinal_losers


# -- run_monte_carlo -----------------------------------------------------------

class TestMonteCarlo:
    def setup_method(self):
        self.predictor = make_predictor(TEAMS)

    def test_all_48_teams_have_probabilities(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=200, seed=0)
        assert set(results["probabilities"]) == set(TEAMS)

    def test_champion_probs_sum_to_one(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=500, seed=1)
        total = sum(p["champion"] for p in results["probabilities"].values())
        assert abs(total - 1.0) < 0.02

    def test_stage_totals_match_48_team_2026_format(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=300, seed=2)
        probabilities = results["probabilities"]

        assert abs(sum(p["group_winner"] for p in probabilities.values()) - 12.0) < 0.05
        assert abs(sum(p["group_advance"] for p in probabilities.values()) - 32.0) < 0.05
        assert abs(sum(p["round_of_32"] for p in probabilities.values()) - 32.0) < 0.05
        assert abs(sum(p["round_of_16"] for p in probabilities.values()) - 16.0) < 0.05
        assert abs(sum(p["quarterfinal"] for p in probabilities.values()) - 8.0) < 0.05
        assert abs(sum(p["semifinal"] for p in probabilities.values()) - 4.0) < 0.05
        assert abs(sum(p["final"] for p in probabilities.values()) - 2.0) < 0.05
        assert abs(sum(p["champion"] for p in probabilities.values()) - 1.0) < 0.05

    def test_group_advance_greater_than_champion(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=500, seed=3)
        for probabilities in results["probabilities"].values():
            assert probabilities["group_advance"] >= probabilities["champion"] - 0.01

    def test_group_advance_greater_than_group_winner(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=500, seed=6)
        for probabilities in results["probabilities"].values():
            assert probabilities["group_advance"] >= probabilities["group_winner"]

    def test_top_contenders_length(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=200, seed=4)
        assert len(results["top_contenders"]) == 8

    def test_probabilities_are_valid(self):
        results = run_monte_carlo(GROUPS, self.predictor, n_simulations=300, seed=5)
        for team, probabilities in results["probabilities"].items():
            for stage, value in probabilities.items():
                assert 0.0 <= value <= 1.0, f"{team} {stage}={value} out of range"
