"""Automated daily ingestion of World Cup 2026 match results.

Fetches results from a configurable external provider, normalizes them into
the schema used by `data/sample/results.csv`, and merges them into an
existing results table without disturbing already-locked-in Final rows.

Configuration (environment variables, no hardcoded secrets):
    RESULTS_PROVIDER  e.g. "football-data"
    RESULTS_API_KEY   provider API key
    RESULTS_API_URL   optional base URL override (defaults per provider)
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

from src.data.load_data import load_groups
from src.data.sync_results import STAGE_MAP, _group_for_team, _team_name
from src.simulation.live_tracker import RESULT_COLUMNS, normalize_results

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"


class ResultsFetchError(RuntimeError):
    """Raised when the external results provider can't be reached or parsed."""


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_latest_world_cup_results() -> pd.DataFrame:
    """Fetch the latest match results/schedule from the configured provider.

    Raises ResultsFetchError if RESULTS_PROVIDER is unset/unsupported or the
    request fails. Callers should fall back to the existing results CSV.
    """
    provider = os.environ.get("RESULTS_PROVIDER", "").strip().lower()
    if not provider:
        raise ResultsFetchError("RESULTS_PROVIDER is not configured")
    if provider == "football-data":
        return _fetch_football_data()
    raise ResultsFetchError(f"Unsupported RESULTS_PROVIDER: {provider!r}")


def _fetch_football_data() -> pd.DataFrame:
    api_key = os.environ.get("RESULTS_API_KEY")
    if not api_key:
        raise ResultsFetchError("RESULTS_API_KEY is not configured")
    base_url = (os.environ.get("RESULTS_API_URL") or FOOTBALL_DATA_BASE_URL).rstrip("/")

    try:
        response = requests.get(
            f"{base_url}/competitions/WC/matches",
            headers={"X-Auth-Token": api_key},
            timeout=10,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])
    except (requests.RequestException, ValueError) as exc:
        raise ResultsFetchError(f"football-data request failed: {exc}") from exc

    if not matches:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    groups_df = load_groups()
    rows = [_football_data_match_to_row(match, groups_df) for match in matches]
    return normalize_results(pd.DataFrame(rows))


def _football_data_match_to_row(match: dict[str, Any], groups_df: pd.DataFrame) -> dict[str, Any]:
    team_a = _team_name(match.get("homeTeam"))
    team_b = _team_name(match.get("awayTeam"))
    full_time = match.get("score", {}).get("fullTime", {})
    score_a, score_b = full_time.get("home"), full_time.get("away")

    if match.get("status") == "FINISHED" and score_a is not None and score_b is not None:
        status = "Final"
        if score_a > score_b:
            winner = team_a
        elif score_b > score_a:
            winner = team_b
        else:
            winner = "Draw"
    else:
        status, score_a, score_b, winner = "Scheduled", None, None, None

    return {
        "date": pd.to_datetime(match["utcDate"]).strftime("%Y-%m-%d"),
        "stage": STAGE_MAP.get(match.get("stage", "GROUP_STAGE"), "Group"),
        "group": _group_for_team(groups_df, team_a) or "",
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Validate fetched results and return them normalized.

    - Final rows must have integer scores and a winner that is team_a,
      team_b, or "Draw".
    - Both teams in every row must exist in the app's group/team data.
    - No two rows within the same stage may be the same fixture.

    Raises ValueError on violation.
    """
    normalized = normalize_results(df)

    known_teams = set(load_groups()["team"])
    for row in normalized.itertuples(index=False):
        unknown = {row.team_a, row.team_b} - known_teams
        if unknown:
            raise ValueError(f"Unknown team(s) in results data: {sorted(unknown)}")

    seen_fixtures: set[tuple[str, frozenset[str]]] = set()
    for row in normalized.itertuples(index=False):
        fixture = (str(row.stage), frozenset((row.team_a, row.team_b)))
        if fixture in seen_fixtures:
            raise ValueError(f"Duplicate fixture: {row.team_a} vs {row.team_b} ({row.stage})")
        seen_fixtures.add(fixture)

    finals = normalized[normalized["status"].astype(str).str.casefold() == "final"]
    for row in finals.itertuples(index=False):
        if pd.isna(row.score_a) or pd.isna(row.score_b):
            raise ValueError(f"Final result missing scores: {row.team_a} vs {row.team_b}")
        if float(row.score_a) % 1 or float(row.score_b) % 1:
            raise ValueError(f"Scores must be integers: {row.team_a} vs {row.team_b}")
        if row.winner not in (row.team_a, row.team_b, "Draw"):
            raise ValueError(f"Invalid winner for {row.team_a} vs {row.team_b}: {row.winner!r}")
    return normalized


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_results(existing_df: pd.DataFrame, fetched_df: pd.DataFrame) -> pd.DataFrame:
    """Upsert fetched results into an existing results table.

    Rows are matched by team pair rather than date, since providers can
    report kickoff dates in a different timezone than the locally recorded
    date. Existing Final rows with scores are preserved as the source of
    truth; Scheduled rows are upgraded once the fetched source reports a
    Final result, and brand-new matches are appended.
    """
    existing = normalize_results(existing_df)
    fetched = normalize_results(fetched_df)

    rows = existing.to_dict("records")
    by_pair = {frozenset((r["team_a"], r["team_b"])): i for i, r in enumerate(rows)}

    for frow in fetched.to_dict("records"):
        pair = frozenset((frow["team_a"], frow["team_b"]))
        idx = by_pair.get(pair)
        if idx is None:
            by_pair[pair] = len(rows)
            rows.append(frow)
            continue

        erow = rows[idx]
        locked_in = (
            str(erow["status"]).casefold() == "final"
            and pd.notna(erow["score_a"])
            and pd.notna(erow["score_b"])
        )
        if not locked_in:
            rows[idx] = frow

    return normalize_results(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Combined fetch + merge with fallback
# ---------------------------------------------------------------------------

def fetch_and_merge_results(existing_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch live results and merge into existing_df.

    Falls back to existing_df unchanged (normalized) if the provider is
    unconfigured, unreachable, or returns invalid data — the app must not
    crash if the external source is unavailable.
    """
    try:
        fetched = fetch_latest_world_cup_results()
        validate_results(fetched)
    except (ResultsFetchError, ValueError):
        return normalize_results(existing_df)
    return merge_results(existing_df, fetched)
