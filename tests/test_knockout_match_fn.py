import subprocess

import numpy as np

from knockout_engine import knockout_match_fn


def test_same_seed_produces_identical_winner_sequence():
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        knockout_match_fn("Brazil", "Germany", first_rng)
        for _ in range(500)
    ]
    second = [
        knockout_match_fn("Brazil", "Germany", second_rng)
        for _ in range(500)
    ]

    assert first == second


def test_return_is_always_one_of_the_inputs():
    rng = np.random.default_rng(7)

    winners = {
        knockout_match_fn("Argentina", "France", rng)
        for _ in range(100)
    }

    assert winners <= {"Argentina", "France"}


def test_equal_strength_matchup_always_returns_a_winner():
    rng = np.random.default_rng(11)

    winners = [
        knockout_match_fn("Equal A", "Equal B", rng)
        for _ in range(100)
    ]

    assert all(winner in {"Equal A", "Equal B"} for winner in winners)


def test_knockout_engine_does_not_call_random_random():
    result = subprocess.run(
        ["git", "grep", "-n", "random.random", "--", "knockout_engine.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
