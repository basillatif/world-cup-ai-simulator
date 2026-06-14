"""Fetch the latest World Cup results and merge them into results.csv.

Run manually or via the scheduled `update-results` GitHub Actions workflow.
Requires RESULTS_PROVIDER and RESULTS_API_KEY (and optionally
RESULTS_API_URL) to be set as environment variables.

Usage:
    python scripts/update_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.load_data import SAMPLE_DIR
from src.data.results_updater import (
    ResultsFetchError,
    fetch_latest_world_cup_results,
    merge_results,
    validate_results,
)
from src.simulation.live_tracker import normalize_results


def _scores_equal(a: float, b: float) -> bool:
    return (pd.isna(a) and pd.isna(b)) or a == b


def main() -> None:
    results_path = SAMPLE_DIR / "results.csv"
    existing = normalize_results(pd.read_csv(results_path, parse_dates=["date"]))

    try:
        fetched = fetch_latest_world_cup_results()
    except ResultsFetchError as exc:
        print(f"Skipping update — {exc}")
        return
    validate_results(fetched)

    merged = merge_results(existing, fetched)

    existing_by_pair = {
        frozenset((row.team_a, row.team_b)): row for row in existing.itertuples(index=False)
    }

    added, updated = [], []
    for row in merged.itertuples(index=False):
        prev = existing_by_pair.get(frozenset((row.team_a, row.team_b)))
        if prev is None:
            added.append(row)
        elif (
            prev.status != row.status
            or not _scores_equal(prev.score_a, row.score_a)
            or not _scores_equal(prev.score_b, row.score_b)
        ):
            updated.append(row)

    if not added and not updated:
        print("No changes — results.csv is already up to date.")
        return

    for column in ("score_a", "score_b"):
        merged[column] = merged[column].astype("Int64")
    merged.to_csv(results_path, index=False)

    for row in added:
        print(f"Added: {row.date.date()} {row.team_a} vs {row.team_b} ({row.status})")
    for row in updated:
        print(
            f"Updated: {row.date.date()} {row.team_a} {row.score_a}-{row.score_b} "
            f"{row.team_b} ({row.status})"
        )


if __name__ == "__main__":
    main()
