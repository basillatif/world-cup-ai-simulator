"""Live World Cup 2026 scorecard backed by the football-data.org API.

This module is pure data display: it fetches the current live match (and,
failing that, the next scheduled match) from football-data.org and renders
it with Streamlit. It performs no Claude/LLM calls and generates no scores,
predictions, or probabilities — the LLM layer in this app is narrator-only
and is intentionally absent here.

API used: football-data.org v4
  GET https://api.football-data.org/v4/competitions/WC/matches?status=LIVE
  GET https://api.football-data.org/v4/competitions/WC/matches?status=IN_PLAY
  GET https://api.football-data.org/v4/competitions/WC/matches?status=SCHEDULED

Setup:
  Add your free API key (https://www.football-data.org/client/register) to
  .streamlit/secrets.toml:

      FOOTBALL_DATA_API_KEY = "your_key_here"
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st

API_BASE = "https://api.football-data.org/v4"
COMPETITION = "WC"


def _get_api_key() -> str | None:
    try:
        key = st.secrets.get("FOOTBALL_DATA_API_KEY")
    except Exception:
        key = None
    return str(key) if key else None


def _fetch_matches(api_key: str, status: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE}/competitions/{COMPETITION}/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": status},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("matches", [])


def fetch_live_match(api_key: str) -> dict[str, Any] | None:
    """Return the current live match, checking LIVE then IN_PLAY."""
    for status in ("LIVE", "IN_PLAY"):
        matches = _fetch_matches(api_key, status)
        if matches:
            return matches[0]
    return None


def fetch_next_scheduled_match(api_key: str) -> dict[str, Any] | None:
    """Return the nearest future scheduled match, if any."""
    matches = _fetch_matches(api_key, "SCHEDULED")
    if not matches:
        return None
    return min(matches, key=lambda m: m["utcDate"])


def _team_name(team: dict[str, Any] | None) -> str:
    if not team:
        return "TBD"
    return team.get("shortName") or team.get("name") or "TBD"


def _render_match_header(match: dict[str, Any]) -> None:
    home = _team_name(match.get("homeTeam"))
    away = _team_name(match.get("awayTeam"))
    st.markdown(
        f"<h2 style='text-align: center;'>{home} vs {away}</h2>",
        unsafe_allow_html=True,
    )

    group = match.get("group")
    if group:
        st.caption(f"Group {group.replace('GROUP_', '')}" if group.startswith("GROUP_") else group)


def _render_score(match: dict[str, Any]) -> None:
    score = match.get("score", {})
    full_time = score.get("fullTime", {})
    home_score = full_time.get("home")
    away_score = full_time.get("away")

    home = _team_name(match.get("homeTeam"))
    away = _team_name(match.get("awayTeam"))

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.metric(home, home_score if home_score is not None else "-")
    with col2:
        st.markdown("<h1 style='text-align: center;'>—</h1>", unsafe_allow_html=True)
    with col3:
        st.metric(away, away_score if away_score is not None else "-")


def _render_status(match: dict[str, Any]) -> None:
    status = match.get("status", "UNKNOWN")
    minute = match.get("minute")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", status)
    with col2:
        st.metric("Minute", f"{minute}'" if minute is not None else "-")


def _render_next_match(api_key: str) -> None:
    try:
        next_match = fetch_next_scheduled_match(api_key)
    except requests.RequestException as e:
        st.error(f"Could not fetch the next scheduled match: {e}")
        return

    if not next_match:
        st.info("No upcoming matches found.")
        return

    home = _team_name(next_match.get("homeTeam"))
    away = _team_name(next_match.get("awayTeam"))
    kickoff = next_match.get("utcDate")
    st.markdown("**Next scheduled match:**")
    st.markdown(f"### {home} vs {away}")
    if kickoff:
        kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        st.caption(f"Kickoff: {kickoff_dt.strftime('%Y-%m-%d %H:%M UTC')}")

    group = next_match.get("group")
    if group:
        st.caption(f"Group {group.replace('GROUP_', '')}" if group.startswith("GROUP_") else group)


def render_live_scorecard() -> None:
    """Render the Live Scorecard page."""
    st.header("🟢 Live Scorecard")

    api_key = _get_api_key()
    if not api_key:
        st.error(
            "No football-data.org API key configured. Add one to "
            "`.streamlit/secrets.toml`:\n\n"
            "```toml\n"
            'FOOTBALL_DATA_API_KEY = "your_key_here"\n'
            "```\n\n"
            "Get a free key at https://www.football-data.org/client/register"
        )
        return

    refresh_seconds = st.sidebar.select_slider(
        "Live Scorecard refresh interval (seconds)",
        options=[30, 60, 120],
        value=60,
    )

    try:
        live_match = fetch_live_match(api_key)
    except requests.RequestException as e:
        st.error(f"Could not fetch live match data: {e}")
        return

    if live_match:
        _render_match_header(live_match)
        _render_score(live_match)
        _render_status(live_match)
    else:
        st.info("No match currently live. Check back during match windows.")
        _render_next_match(api_key)

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    countdown_placeholder = st.empty()
    for remaining in range(refresh_seconds, 0, -1):
        countdown_placeholder.caption(f"Refreshing in {remaining}s…")
        time.sleep(1)
    st.rerun()
