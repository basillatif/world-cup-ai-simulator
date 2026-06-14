"""Sync completed World Cup 2026 matches from football-data.org into results.csv.

Fetches FINISHED matches from the football-data.org API and appends any that
are not already present to data/sample/results.csv, using the schema expected
by the Tournament Results tab (see src/simulation/live_tracker.py).

Usage:
    python -m src.data.sync_results
    python -m src.data.sync_results --dry-run

Requires FOOTBALL_DATA_API_KEY in .streamlit/secrets.toml (or the
FOOTBALL_DATA_API_KEY environment variable).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.load_data import SAMPLE_DIR

API_BASE = "https://api.football-data.org/v4"
COMPETITION = "WC"

RESULTS_COLUMNS = ["date", "stage", "group", "team_a", "team_b", "score_a", "score_b", "winner", "status"]

# football-data.org team names that differ from the names used in
# data/sample/groups.csv, data/sample/teams.csv, and results.csv.
NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
}

STAGE_MAP = {
    "GROUP_STAGE": "Group",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-final",
    "SEMI_FINALS": "Semi-final",
    "THIRD_PLACE": "Third-place play-off",
    "FINAL": "Final",
}


def _get_api_key() -> str | None:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if key:
        return key
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    secrets_path = Path(__file__).parents[2] / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    return secrets.get("FOOTBALL_DATA_API_KEY")


def _team_name(team: dict[str, Any] | None) -> str:
    if not team:
        return "TBD"
    name = team.get("name") or team.get("shortName") or "TBD"
    return NAME_MAP.get(name, name)


def fetch_finished_matches(api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE}/competitions/{COMPETITION}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("matches", [])


def _group_for_team(groups_df: pd.DataFrame, team: str) -> str | None:
    row = groups_df[groups_df["team"] == team]
    return row.iloc[0]["group"] if not row.empty else None


def match_to_result_row(match: dict[str, Any], groups_df: pd.DataFrame) -> dict[str, Any]:
    team_a = _team_name(match.get("homeTeam"))
    team_b = _team_name(match.get("awayTeam"))
    full_time = match.get("score", {}).get("fullTime", {})
    score_a, score_b = full_time.get("home"), full_time.get("away")

    if score_a > score_b:
        winner = team_a
    elif score_b > score_a:
        winner = team_b
    else:
        winner = "Draw"

    return {
        "date": pd.to_datetime(match["utcDate"]).strftime("%Y-%m-%d"),
        "stage": STAGE_MAP.get(match.get("stage", "GROUP_STAGE"), "Group"),
        "group": _group_for_team(groups_df, team_a) or "",
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
        "status": "Final",
    }


def _is_duplicate(row: dict[str, Any], existing: pd.DataFrame) -> bool:
    """A match is a duplicate if the same two teams already have a result for
    the same stage. football-data.org reports kickoff times in UTC, which can
    fall on a different calendar date than the locally-recorded result, so
    dates are intentionally not part of the dedup key."""
    teams = {row["team_a"], row["team_b"]}
    for _, existing_row in existing.iterrows():
        if {existing_row["team_a"], existing_row["team_b"]} == teams and existing_row["stage"] == row["stage"]:
            return True
    return False


def sync_results(results_path: Path, groups_path: Path, api_key: str, dry_run: bool = False) -> list[dict[str, Any]]:
    existing = pd.read_csv(results_path, dtype=str)
    groups_df = pd.read_csv(groups_path)

    new_rows = []
    for match in fetch_finished_matches(api_key):
        row = match_to_result_row(match, groups_df)
        if not _is_duplicate(row, existing):
            new_rows.append(row)

    if new_rows and not dry_run:
        updated = pd.concat([existing, pd.DataFrame(new_rows)[RESULTS_COLUMNS].astype(str)], ignore_index=True)
        updated.to_csv(results_path, index=False)

    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print new results without writing to results.csv")
    args = parser.parse_args()

    api_key = _get_api_key()
    if not api_key:
        raise SystemExit(
            "No football-data.org API key found. Set FOOTBALL_DATA_API_KEY "
            "in .streamlit/secrets.toml or as an environment variable."
        )

    new_rows = sync_results(
        results_path=SAMPLE_DIR / "results.csv",
        groups_path=SAMPLE_DIR / "groups.csv",
        api_key=api_key,
        dry_run=args.dry_run,
    )

    if not new_rows:
        print("No new finished matches found.")
        return

    for row in new_rows:
        print(f"{row['date']}: {row['team_a']} {row['score_a']}-{row['score_b']} {row['team_b']} ({row['status']})")
    if args.dry_run:
        print(f"\n(dry run — {len(new_rows)} row(s) not written)")
    else:
        print(f"\nAppended {len(new_rows)} row(s) to {SAMPLE_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
