"""Tests for automated results ingestion: fetch config, validation, merging."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.results_updater import (
    ResultsFetchError,
    fetch_and_merge_results,
    fetch_latest_world_cup_results,
    merge_results,
    validate_results,
)


def _row(**overrides):
    row = {
        "date": "2026-06-11",
        "stage": "Group",
        "group": "A",
        "team_a": "Mexico",
        "team_b": "South Africa",
        "score_a": 2,
        "score_b": 0,
        "winner": "Mexico",
        "status": "Final",
    }
    row.update(overrides)
    return row


# -- fetch_latest_world_cup_results --------------------------------------------

def test_fetch_raises_when_provider_unset(monkeypatch):
    monkeypatch.delenv("RESULTS_PROVIDER", raising=False)
    with pytest.raises(ResultsFetchError):
        fetch_latest_world_cup_results()


def test_fetch_raises_for_unsupported_provider(monkeypatch):
    monkeypatch.setenv("RESULTS_PROVIDER", "not-a-real-provider")
    with pytest.raises(ResultsFetchError):
        fetch_latest_world_cup_results()


def test_fetch_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setenv("RESULTS_PROVIDER", "football-data")
    monkeypatch.delenv("RESULTS_API_KEY", raising=False)
    with pytest.raises(ResultsFetchError):
        fetch_latest_world_cup_results()


# -- validate_results -----------------------------------------------------------

def test_validate_results_accepts_final_and_scheduled_rows():
    df = pd.DataFrame([
        _row(),
        _row(team_a="South Korea", team_b="Czechia", score_a=1, score_b=1, winner="Draw"),
        _row(
            team_a="Mexico", team_b="Czechia", date="2026-06-17",
            score_a=None, score_b=None, winner=None, status="Scheduled",
        ),
    ])
    validated = validate_results(df)
    assert len(validated) == 3


def test_validate_results_rejects_non_integer_scores():
    df = pd.DataFrame([_row(score_a=2.5)])
    with pytest.raises(ValueError):
        validate_results(df)


def test_validate_results_rejects_unknown_team():
    df = pd.DataFrame([_row(team_a="Atlantis", winner="Atlantis")])
    with pytest.raises(ValueError):
        validate_results(df)


def test_validate_results_rejects_duplicate_fixture():
    df = pd.DataFrame([
        _row(),
        _row(date="2026-06-12", score_a=1, score_b=1, winner="Draw"),
    ])
    with pytest.raises(ValueError):
        validate_results(df)


# -- merge_results -----------------------------------------------------------------

def test_merge_adds_new_match():
    existing = pd.DataFrame([_row()])
    fetched = pd.DataFrame([
        _row(team_a="South Korea", team_b="Czechia", score_a=2, score_b=1, winner="South Korea")
    ])

    merged = merge_results(existing, fetched)

    pairs = {frozenset((r.team_a, r.team_b)) for r in merged.itertuples(index=False)}
    assert frozenset(("Mexico", "South Africa")) in pairs
    assert frozenset(("South Korea", "Czechia")) in pairs


def test_merge_upgrades_scheduled_to_final():
    existing = pd.DataFrame([
        _row(
            team_a="Mexico", team_b="Czechia", date="2026-06-17",
            score_a=None, score_b=None, winner=None, status="Scheduled",
        )
    ])
    fetched = pd.DataFrame([
        _row(team_a="Mexico", team_b="Czechia", date="2026-06-17", score_a=3, score_b=1, winner="Mexico")
    ])

    merged = merge_results(existing, fetched).set_index(["team_a", "team_b"])

    row = merged.loc[("Mexico", "Czechia")]
    assert row["status"] == "Final"
    assert row["score_a"] == 3
    assert row["score_b"] == 1


def test_merge_preserves_locked_in_final_result():
    existing = pd.DataFrame([_row(score_a=2, score_b=0, winner="Mexico")])
    fetched = pd.DataFrame([_row(score_a=5, score_b=5, winner="Draw")])

    merged = merge_results(existing, fetched).set_index(["team_a", "team_b"])

    row = merged.loc[("Mexico", "South Africa")]
    assert row["score_a"] == 2
    assert row["score_b"] == 0
    assert row["winner"] == "Mexico"


# -- fetch_and_merge_results: fallback --------------------------------------------

def test_fetch_and_merge_falls_back_when_unconfigured(monkeypatch):
    monkeypatch.delenv("RESULTS_PROVIDER", raising=False)
    existing = pd.DataFrame([_row()])

    result = fetch_and_merge_results(existing)

    assert len(result) == 1
    assert result.iloc[0]["team_a"] == "Mexico"
