"""Tests for knockout_bracket: topology integrity + probability conservation."""
import numpy as np
import pytest

from knockout_bracket import (
    BRACKET, R32_SEEDING, ROUND_OF, simulate_bracket,
)


def _ratings():
    teams = sorted({t for pair in R32_SEEDING.values() for t in pair})
    return {t: i for i, t in enumerate(teams)}


def make_match_fn(seed_ratings):
    def match_fn(a, b, rng):
        pa = 1.0 / (1.0 + 10 ** ((seed_ratings[b] - seed_ratings[a]) / 10))
        return a if rng.random() < pa else b
    return match_fn


def test_thirty_two_distinct_teams():
    teams = {t for pair in R32_SEEDING.values() for t in pair}
    assert len(teams) == 32


def test_every_match_has_two_sources():
    # 73..104 inclusive = 32 knockout matches (incl. third-place play-off)
    assert sorted(BRACKET) == list(range(73, 105))
    for m, (a, b) in BRACKET.items():
        assert a and b


def test_dependencies_are_backward_only():
    # a match may only reference earlier match ids -> sorted play is valid
    for m, (a, b) in BRACKET.items():
        for kind, ref in (a, b):
            if kind in ("W", "L"):
                assert ref < m, f"match {m} references future match {ref}"


def test_champion_probabilities_sum_to_one():
    res = simulate_bracket(make_match_fn(_ratings()), n=3000, seed=1)
    total = sum(v["champion"] for v in res.values())
    # rounded to 2dp per team, so allow rounding slack across 32 teams
    assert abs(total - 1.0) < 0.05


def test_all_r32_teams_reach_r32_with_prob_one():
    res = simulate_bracket(make_match_fn(_ratings()), n=1000, seed=2)
    for team, v in res.items():
        assert v["reach"]["R32"] == 1.0


def test_reach_is_monotonically_non_increasing():
    res = simulate_bracket(make_match_fn(_ratings()), n=3000, seed=3)
    order = ["R32", "R16", "QF", "SF", "F"]
    for team, v in res.items():
        seq = [v["reach"][r] for r in order]
        for earlier, later in zip(seq, seq[1:]):
            assert later <= earlier + 1e-9, f"{team}: {seq}"


def test_determinism_same_seed_same_result():
    fn = make_match_fn(_ratings())
    a = simulate_bracket(fn, n=1500, seed=7)
    b = simulate_bracket(fn, n=1500, seed=7)
    assert a == b


def test_conditioning_pins_a_known_result():
    # Force South Africa through match 73; they must never be eliminated AT the
    # R32 stage -> reach R16 with probability 1.
    fn = make_match_fn(_ratings())
    res = simulate_bracket(fn, n=1000, seed=9, decided={73: "South Africa"})
    assert res["South Africa"]["reach"]["R16"] == 1.0
    # And Canada, pinned as the loser of 73, cannot reach R16.
    assert res["Canada"]["reach"]["R16"] == 0.0


def test_decided_rejects_team_not_in_matchup():
    fn = make_match_fn(_ratings())
    with pytest.raises(ValueError):
        simulate_bracket(fn, n=10, seed=0, decided={73: "Brazil"})
