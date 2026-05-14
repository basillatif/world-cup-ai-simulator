"""Claude-powered analyst agent — explains model outputs, never replaces them."""

from __future__ import annotations

import os

import anthropic

from src.genai.prompt_templates import (
    SYSTEM_PROMPT,
    group_summary_prompt,
    match_preview_prompt,
    tournament_outlook_prompt,
    upset_alert_prompt,
)
from src.features.build_features import compute_head_to_head, compute_recent_form


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the analyst agent."
        )
    return anthropic.Anthropic(api_key=api_key)


def _call_claude(user_prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    client = _client()
    message = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


class AnalystAgent:
    """
    Wraps the Claude API and exposes high-level analysis methods.

    Design principle: every method receives pre-computed model outputs.
    Claude's job is narration and explanation, not prediction.
    """

    def __init__(self, teams_df, matches_df):
        self.teams_df = teams_df
        self.matches_df = matches_df

    def _get_team_stats(self, team: str) -> dict:
        row = self.teams_df[self.teams_df["team"] == team]
        return row.iloc[0].to_dict() if not row.empty else {}

    def match_preview(
        self,
        home_team: str,
        away_team: str,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_xg: float,
        away_xg: float,
        elo_diff: float,
    ) -> str:
        h2h = compute_head_to_head(self.matches_df, home_team, away_team)
        home_form = compute_recent_form(self.matches_df, home_team)
        away_form = compute_recent_form(self.matches_df, away_team)
        home_stats = self._get_team_stats(home_team)
        away_stats = self._get_team_stats(away_team)

        prompt = match_preview_prompt(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            home_xg=home_xg,
            away_xg=away_xg,
            elo_diff=elo_diff,
            h2h=h2h,
            home_form=home_form,
            away_form=away_form,
            home_stats=home_stats,
            away_stats=away_stats,
        )
        return _call_claude(prompt)

    def tournament_outlook(
        self,
        team: str,
        probs: dict,
        group: str,
        group_opponents: list[str],
    ) -> str:
        team_stats = self._get_team_stats(team)
        prompt = tournament_outlook_prompt(
            team=team,
            probs=probs,
            group=group,
            group_opponents=group_opponents,
            team_stats=team_stats,
        )
        return _call_claude(prompt)

    def group_summary(
        self,
        group: str,
        teams: list[str],
        advance_probs: dict[str, float],
    ) -> str:
        team_stats_map = {t: self._get_team_stats(t) for t in teams}
        prompt = group_summary_prompt(
            group=group,
            teams=teams,
            advance_probs=advance_probs,
            team_stats_map=team_stats_map,
        )
        return _call_claude(prompt)

    def upset_alert(
        self,
        underdog: str,
        favourite: str,
        underdog_win_prob: float,
        elo_diff: float,
    ) -> str:
        underdog_stats = self._get_team_stats(underdog)
        favourite_stats = self._get_team_stats(favourite)
        prompt = upset_alert_prompt(
            underdog=underdog,
            favourite=favourite,
            underdog_win_prob=underdog_win_prob,
            elo_diff=elo_diff,
            underdog_stats=underdog_stats,
            favourite_stats=favourite_stats,
        )
        return _call_claude(prompt, max_tokens=512)
