"""Data-quality checks for the confirmed 2026 World Cup field."""

from __future__ import annotations

from src.data.load_data import load_groups, load_matches, load_teams


CONFIRMED_2026_GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

RATING_COLUMNS = {
    "elo_rating",
    "fifa_rank",
    "avg_goals_scored",
    "avg_goals_conceded",
    "recent_form",
}


def test_teams_match_confirmed_2026_field():
    teams = load_teams()
    expected = {
        team
        for group_teams in CONFIRMED_2026_GROUPS.values()
        for team in group_teams
    }

    assert len(teams) == 48
    assert set(teams["team"]) == expected
    assert teams["team"].is_unique


def test_groups_match_confirmed_2026_draw():
    groups = load_groups()
    actual = {
        group: group_df["team"].tolist()
        for group, group_df in groups.groupby("group", sort=True)
    }

    assert actual == CONFIRMED_2026_GROUPS
    assert len(actual) == 12
    assert all(len(group_teams) == 4 for group_teams in actual.values())
    assert groups["team"].is_unique


def test_every_team_has_strength_inputs():
    teams = load_teams()

    assert not teams[list(RATING_COLUMNS)].isna().any().any()
    assert (teams["elo_rating"] > 0).all()
    assert (teams["fifa_rank"] > 0).all()
    assert (teams["avg_goals_scored"] >= 0).all()
    assert (teams["avg_goals_conceded"] >= 0).all()


def test_sample_matches_only_reference_confirmed_teams():
    matches = load_matches()
    expected = {
        team
        for group_teams in CONFIRMED_2026_GROUPS.values()
        for team in group_teams
    }
    match_teams = set(matches["home_team"]) | set(matches["away_team"])

    assert match_teams <= expected
    assert expected <= match_teams
    assert set(matches["tournament"]) == {"Illustrative Sample Data"}
