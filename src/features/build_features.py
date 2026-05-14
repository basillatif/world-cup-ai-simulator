"""Feature engineering: head-to-head records, rolling form, matchup deltas."""

import pandas as pd
import numpy as np


def compute_head_to_head(matches_df: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """Return win/draw/loss counts and goal stats between two teams."""
    mask = (
        ((matches_df["home_team"] == team_a) & (matches_df["away_team"] == team_b)) |
        ((matches_df["home_team"] == team_b) & (matches_df["away_team"] == team_a))
    )
    h2h = matches_df[mask].copy()

    if h2h.empty:
        return {"games": 0, "a_wins": 0, "b_wins": 0, "draws": 0,
                "a_goals": 0.0, "b_goals": 0.0}

    a_goals, b_goals, a_wins, b_wins, draws = 0, 0, 0, 0, 0
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            ag, bg = row["home_goals"], row["away_goals"]
        else:
            ag, bg = row["away_goals"], row["home_goals"]
        a_goals += ag
        b_goals += bg
        if ag > bg:
            a_wins += 1
        elif bg > ag:
            b_wins += 1
        else:
            draws += 1

    n = len(h2h)
    return {
        "games": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_goals": round(a_goals / n, 2),
        "b_goals": round(b_goals / n, 2),
    }


def compute_recent_form(matches_df: pd.DataFrame, team: str, n: int = 5) -> dict:
    """Points-per-game and goal difference over last n matches."""
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    recent = matches_df[mask].tail(n)

    if recent.empty:
        return {"ppg": 0.0, "gd": 0.0, "games": 0}

    points, gd = 0, 0
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            scored, conceded = row["home_goals"], row["away_goals"]
        else:
            scored, conceded = row["away_goals"], row["home_goals"]
        gd += scored - conceded
        if scored > conceded:
            points += 3
        elif scored == conceded:
            points += 1

    n_actual = len(recent)
    return {
        "ppg": round(points / n_actual, 2),
        "gd": gd,
        "games": n_actual,
    }


def build_matchup_features(
    teams_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral: bool = True,
) -> dict:
    """Assemble feature vector for a single matchup."""
    home = teams_df[teams_df["team"] == home_team].iloc[0]
    away = teams_df[teams_df["team"] == away_team].iloc[0]

    h2h = compute_head_to_head(matches_df, home_team, away_team)
    home_form = compute_recent_form(matches_df, home_team)
    away_form = compute_recent_form(matches_df, away_team)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "elo_diff": home["elo_rating"] - away["elo_rating"],
        "rank_diff": away["fifa_rank"] - home["fifa_rank"],  # positive = home is better ranked
        "home_avg_scored": home["avg_goals_scored"],
        "home_avg_conceded": home["avg_goals_conceded"],
        "away_avg_scored": away["avg_goals_scored"],
        "away_avg_conceded": away["avg_goals_conceded"],
        "home_recent_form": home["recent_form"],
        "away_recent_form": away["recent_form"],
        "squad_value_ratio": home["squad_value_m"] / max(away["squad_value_m"], 1),
        "home_form_ppg": home_form["ppg"],
        "away_form_ppg": away_form["ppg"],
        "h2h_home_win_rate": h2h["a_wins"] / max(h2h["games"], 1),
        "h2h_games": h2h["games"],
    }


def build_all_group_matchups(
    teams_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    groups_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return features for every intra-group fixture."""
    rows = []
    for group, gdf in groups_df.groupby("group"):
        group_teams = gdf["team"].tolist()
        for i, t1 in enumerate(group_teams):
            for t2 in group_teams[i + 1:]:
                feats = build_matchup_features(teams_df, matches_df, t1, t2, neutral=True)
                feats["group"] = group
                rows.append(feats)
    return pd.DataFrame(rows)
