"""Live-result conditioning helpers for the frozen knockout bracket."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from knockout_bracket import BRACKET, R32_SEEDING, Source, _resolve
from src.data.sync_results import NAME_MAP


R32_MATCH_IDS = range(73, 89)
FROZEN_TEAMS = {team for pair in R32_SEEDING.values() for team in pair}


def _canonical_key(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def build_team_aliases(live_names: Iterable[str]) -> dict[str, str]:
    aliases = {team: team for team in FROZEN_TEAMS}
    by_key = {_canonical_key(team): team for team in FROZEN_TEAMS}

    for provider_name, app_name in NAME_MAP.items():
        if app_name in FROZEN_TEAMS:
            aliases[provider_name] = app_name

    for raw_name in live_names:
        if raw_name in aliases:
            continue
        mapped = NAME_MAP.get(raw_name)
        if mapped in FROZEN_TEAMS:
            aliases[raw_name] = mapped
            continue
        keyed = by_key.get(_canonical_key(raw_name))
        if keyed:
            aliases[raw_name] = keyed
            continue
        raise ValueError(f"Unresolved live team name: {raw_name!r}")

    return aliases


def normalize_team_name(name: str, aliases: Mapping[str, str] | None = None) -> str:
    lookup = aliases or build_team_aliases([name])
    try:
        return lookup[name]
    except KeyError as exc:
        raise ValueError(f"Unresolved live team name: {name!r}") from exc


def _fixture_match_id(fixture: Mapping[str, Any]) -> int:
    for key in ("match_id", "matchId", "matchday", "id"):
        value = fixture.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Live R32 fixture has no match id: {fixture!r}")


def _fixture_pair(fixture: Mapping[str, Any]) -> tuple[str, str]:
    if "team_a" in fixture and "team_b" in fixture:
        return str(fixture["team_a"]), str(fixture["team_b"])
    if "homeTeam" in fixture and "awayTeam" in fixture:
        home = fixture["homeTeam"]
        away = fixture["awayTeam"]
        home_name = home.get("name") if isinstance(home, Mapping) else home
        away_name = away.get("name") if isinstance(away, Mapping) else away
        return str(home_name), str(away_name)
    raise ValueError(f"Live R32 fixture has no teams: {fixture!r}")


def verify_seeding(live_r32_fixtures: Iterable[Mapping[str, Any]]) -> None:
    fixtures = list(live_r32_fixtures)
    live_names = []
    for fixture in fixtures:
        live_names.extend(_fixture_pair(fixture))
    aliases = build_team_aliases(live_names)
    by_id = {_fixture_match_id(fixture): fixture for fixture in fixtures}

    missing = [match_id for match_id in R32_MATCH_IDS if match_id not in by_id]
    if missing:
        raise ValueError(f"Missing live R32 fixture(s): {missing}")

    for match_id in R32_MATCH_IDS:
        frozen_pair = R32_SEEDING[match_id]
        live_pair = tuple(normalize_team_name(team, aliases) for team in _fixture_pair(by_id[match_id]))
        if set(frozen_pair) != set(live_pair):
            raise ValueError(
                f"R32 seeding mismatch for match {match_id}: "
                f"frozen={frozen_pair!r} live={live_pair!r}"
            )


def _row_pair(row: pd.Series, aliases: Mapping[str, str]) -> frozenset[str]:
    return frozenset(
        (
            normalize_team_name(str(row["team_a"]), aliases),
            normalize_team_name(str(row["team_b"]), aliases),
        )
    )


def _source_available(source: Source, winners: dict[int, str], losers: dict[int, str]) -> bool:
    kind, ref = source
    if kind == "seed":
        return True
    if kind == "W":
        return ref in winners
    if kind == "L":
        return ref in losers
    return False


def build_decided(results: pd.DataFrame) -> dict[int, str]:
    finals = results[results["status"].astype(str).str.casefold() == "final"].copy()
    if "stage" in finals:
        finals = finals[finals["stage"].astype(str).str.casefold() != "group"]
    if finals.empty:
        return {}

    live_names = finals["team_a"].astype(str).tolist() + finals["team_b"].astype(str).tolist()
    live_names.extend(str(winner) for winner in finals["winner"].dropna().tolist() if winner != "Draw")
    aliases = build_team_aliases(live_names)

    rows_by_pair = {_row_pair(row, aliases): row for _, row in finals.iterrows()}
    decided: dict[int, str] = {}
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}

    for match_id in sorted(BRACKET):
        source_a, source_b = BRACKET[match_id]
        if not (
            _source_available(source_a, winners, losers)
            and _source_available(source_b, winners, losers)
        ):
            continue

        team_a = _resolve(source_a, R32_SEEDING, winners, losers, 0)
        team_b = _resolve(source_b, R32_SEEDING, winners, losers, 1)
        row = rows_by_pair.get(frozenset((team_a, team_b)))
        if row is None:
            continue

        winner = normalize_team_name(str(row["winner"]), aliases)
        if winner not in (team_a, team_b):
            raise ValueError(
                f"Invalid knockout winner for match {match_id}: "
                f"{winner!r} not in {team_a!r} v {team_b!r}"
            )
        decided[match_id] = winner
        winners[match_id] = winner
        losers[match_id] = team_b if winner == team_a else team_a

    return decided
