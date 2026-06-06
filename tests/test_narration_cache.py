from __future__ import annotations

from cache.narration_cache import (
    DEFAULT_MODEL,
    cache_key,
    get_or_create_group_analysis,
)


def test_same_probabilities_produce_same_key():
    probabilities = {
        "United States": 0.855,
        "Turkey": 0.702,
        "Paraguay": 0.590,
    }

    assert (
        cache_key(group="D", probabilities=probabilities, model=DEFAULT_MODEL)
        == cache_key(group="D", probabilities=dict(reversed(probabilities.items())), model=DEFAULT_MODEL)
    )


def test_probabilities_equal_after_two_decimal_rounding_produce_same_key():
    first = {
        "United States": 0.854,
        "Turkey": 0.702,
        "Paraguay": 0.590,
    }
    second = {
        "United States": 0.853,
        "Turkey": 0.704,
        "Paraguay": 0.591,
    }

    assert (
        cache_key(group="D", probabilities=first, model=DEFAULT_MODEL)
        == cache_key(group="D", probabilities=second, model=DEFAULT_MODEL)
    )


def test_different_rounded_probabilities_produce_different_key():
    first = {
        "United States": 0.854,
        "Turkey": 0.702,
        "Paraguay": 0.590,
    }
    second = {
        "United States": 0.864,
        "Turkey": 0.702,
        "Paraguay": 0.590,
    }

    assert (
        cache_key(group="D", probabilities=first, model=DEFAULT_MODEL)
        != cache_key(group="D", probabilities=second, model=DEFAULT_MODEL)
    )


def test_cache_miss_without_api_key_degrades_without_live_call(tmp_path):
    def live_narration() -> str:
        raise AssertionError("live narration should not be called without an API key")

    narration, status = get_or_create_group_analysis(
        group="D",
        probabilities={"United States": 0.855, "Turkey": 0.702},
        has_api_key=False,
        live_narration=live_narration,
        path=tmp_path / "group_analysis_cache.json",
    )

    assert narration is None
    assert status == "unavailable"
