import pandas as pd
import pytest

from knockout_bracket import R32_SEEDING, simulate_bracket
from knockout_results import build_decided, normalize_team_name, verify_seeding


def _fixture(match_id: int, team_a: str, team_b: str) -> dict:
    return {"match_id": match_id, "team_a": team_a, "team_b": team_b}


def _row(match_id: int, winner: str, stage: str = "Knockout") -> dict:
    team_a, team_b = R32_SEEDING[match_id]
    return {
        "date": "2026-07-01",
        "stage": stage,
        "group": "",
        "team_a": team_a,
        "team_b": team_b,
        "score_a": 1,
        "score_b": 0,
        "winner": winner,
        "status": "Final",
    }


def test_normalization_maps_known_alias_and_unknown_raises():
    assert normalize_team_name("Congo DR") == "DR Congo"

    with pytest.raises(ValueError):
        normalize_team_name("Atlantis")


def test_verify_seeding_passes_with_frozen_fixture_list_and_raises_on_mismatch():
    fixtures = [
        _fixture(match_id, team_a, team_b)
        for match_id, (team_a, team_b) in R32_SEEDING.items()
    ]
    verify_seeding(fixtures)

    bad = list(fixtures)
    bad[0] = _fixture(73, "South Africa", "Brazil")
    with pytest.raises(ValueError, match="match 73"):
        verify_seeding(bad)


def test_build_decided_maps_finished_r32_match_to_id_and_winner():
    results = pd.DataFrame([_row(73, "Canada")])

    assert build_decided(results) == {73: "Canada"}


def test_build_decided_maps_r16_only_after_feeder_results_are_present():
    r16_only = pd.DataFrame([
        {
            "date": "2026-07-05",
            "stage": "Round of 16",
            "group": "",
            "team_a": "Germany",
            "team_b": "France",
            "score_a": 1,
            "score_b": 2,
            "winner": "France",
            "status": "Final",
        }
    ])
    assert build_decided(r16_only) == {}

    with_feeders = pd.concat(
        [
            pd.DataFrame([_row(74, "Germany"), _row(77, "France")]),
            r16_only,
        ],
        ignore_index=True,
    )
    decided = build_decided(with_feeders)
    assert decided[74] == "Germany"
    assert decided[77] == "France"
    assert decided[89] == "France"


def test_build_decided_rejects_illegal_winner():
    results = pd.DataFrame([_row(73, "Brazil")])

    with pytest.raises(ValueError, match="Invalid knockout winner"):
        build_decided(results)


def test_pinning_known_result_makes_team_reach_next_round_with_probability_one():
    def match_fn(team_a, team_b, rng):
        return team_a if rng.random() < 0.5 else team_b

    results = simulate_bracket(match_fn, n=100, seed=1, decided={73: "South Africa"})

    assert results["South Africa"]["reach"]["R16"] == 1.0
