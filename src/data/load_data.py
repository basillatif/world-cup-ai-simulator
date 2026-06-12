"""Data loading and validation utilities."""

from pathlib import Path
import pandas as pd

from src.simulation.live_tracker import normalize_results

DATA_DIR = Path(__file__).parents[2] / "data"
SAMPLE_DIR = DATA_DIR / "sample"


def load_teams(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else SAMPLE_DIR / "teams.csv"
    df = pd.read_csv(path)
    _validate_teams(df)
    return df


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else SAMPLE_DIR / "matches.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.lower() == "true"
    _validate_matches(df)
    return df.sort_values("date").reset_index(drop=True)


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """Load actual and scheduled tournament results for the live tracker."""
    path = Path(path) if path else SAMPLE_DIR / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return normalize_results(df)


def load_groups(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else SAMPLE_DIR / "groups.csv"
    df = pd.read_csv(path)
    _validate_groups(df)
    return df


def _validate_teams(df: pd.DataFrame) -> None:
    required = {"team", "elo_rating", "fifa_rank", "avg_goals_scored",
                "avg_goals_conceded", "recent_form", "squad_value_m", "world_cup_titles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"teams.csv missing columns: {missing}")
    if df["team"].duplicated().any():
        raise ValueError("Duplicate team names in teams.csv")


def _validate_matches(df: pd.DataFrame) -> None:
    required = {"date", "home_team", "away_team", "home_goals", "away_goals",
                "tournament", "neutral"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"matches.csv missing columns: {missing}")


def _validate_groups(df: pd.DataFrame) -> None:
    required = {"group", "team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"groups.csv missing columns: {missing}")
    counts = df.groupby("group")["team"].count()
    if (counts != 4).any():
        raise ValueError(f"Each group must have exactly 4 teams. Got:\n{counts}")


def get_all_teams_in_groups(groups_df: pd.DataFrame) -> list[str]:
    return groups_df["team"].tolist()


def get_team_stats(teams_df: pd.DataFrame, team: str) -> dict:
    row = teams_df[teams_df["team"] == team]
    if row.empty:
        raise KeyError(f"Team not found: {team}")
    return row.iloc[0].to_dict()
