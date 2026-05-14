"""Structured prompt templates for the Claude analyst layer."""

from __future__ import annotations


SYSTEM_PROMPT = """You are an expert football analyst with deep knowledge of international football,
World Cup history, and statistical analysis. You have been given outputs from a Monte Carlo
tournament simulator and a Poisson/ELO prediction model.

Your role is NOT to generate the predictions yourself — the statistical model has already done that.
Your role is to EXPLAIN the model's predictions in compelling, insightful language that a fan
or analyst would find valuable. Ground your explanations in the numbers provided. Be concise
but substantive. Avoid generic statements; refer to specific statistics."""


def match_preview_prompt(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    home_xg: float,
    away_xg: float,
    elo_diff: float,
    h2h: dict,
    home_form: dict,
    away_form: dict,
    home_stats: dict,
    away_stats: dict,
) -> str:
    return f"""The statistical model has generated the following pre-match analysis for
{home_team} vs {away_team} (neutral venue):

**Model Output:**
- {home_team} win probability: {home_win_prob:.1%}
- Draw probability: {draw_prob:.1%}
- {away_team} win probability: {away_win_prob:.1%}
- Expected goals: {home_team} {home_xg:.2f} | {away_team} {away_xg:.2f}
- ELO rating differential (home minus away): {elo_diff:+.0f} points

**Team Profiles:**
{home_team}: ELO {home_stats.get('elo_rating', 'N/A')}, FIFA Rank #{home_stats.get('fifa_rank', 'N/A')},
  Avg scored: {home_stats.get('avg_goals_scored', 'N/A')}, Avg conceded: {home_stats.get('avg_goals_conceded', 'N/A')},
  Squad value: €{home_stats.get('squad_value_m', 'N/A')}M, World Cup titles: {home_stats.get('world_cup_titles', 0)}

{away_team}: ELO {away_stats.get('elo_rating', 'N/A')}, FIFA Rank #{away_stats.get('fifa_rank', 'N/A')},
  Avg scored: {away_stats.get('avg_goals_scored', 'N/A')}, Avg conceded: {away_stats.get('avg_goals_conceded', 'N/A')},
  Squad value: €{away_stats.get('squad_value_m', 'N/A')}M, World Cup titles: {away_stats.get('world_cup_titles', 0)}

**Head-to-Head (last {h2h.get('games', 0)} meetings):**
{home_team} wins: {h2h.get('a_wins', 0)}, Draws: {h2h.get('draws', 0)}, {away_team} wins: {h2h.get('b_wins', 0)}
Average goals: {home_team} {h2h.get('a_goals', 0.0):.1f} | {away_team} {h2h.get('b_goals', 0.0):.1f}

**Recent Form (last 5 matches):**
{home_team}: {home_form.get('ppg', 0.0):.2f} pts/game, GD {home_form.get('gd', 0):+d}
{away_team}: {away_form.get('ppg', 0.0):.2f} pts/game, GD {away_form.get('gd', 0):+d}

Based on these model outputs, write a 3-4 paragraph match preview. Explain WHY the model
favours the predicted outcome. Highlight which statistics are driving the prediction.
Mention any notable risks or upset potential the model's probabilities imply.
Do not invent facts beyond what is given above."""


def tournament_outlook_prompt(
    team: str,
    probs: dict,
    group: str,
    group_opponents: list[str],
    team_stats: dict,
) -> str:
    return f"""The Monte Carlo simulator (10,000 iterations) produced the following
tournament probabilities for {team}:

**Simulated Probabilities:**
- Advance from Group {group}: {probs.get('group_advance', 0):.1%}
- Reach Round of 16: {probs.get('round_of_16', 0):.1%}
- Reach Quarterfinals: {probs.get('quarterfinal', 0):.1%}
- Reach Semifinals: {probs.get('semifinal', 0):.1%}
- Reach Final: {probs.get('final', 0):.1%}
- Win the tournament: {probs.get('champion', 0):.1%}

**Team Profile:**
ELO rating: {team_stats.get('elo_rating', 'N/A')} | FIFA Rank: #{team_stats.get('fifa_rank', 'N/A')}
Avg goals scored: {team_stats.get('avg_goals_scored', 'N/A')} | Avg goals conceded: {team_stats.get('avg_goals_conceded', 'N/A')}
Recent form score: {team_stats.get('recent_form', 'N/A')} | Squad value: €{team_stats.get('squad_value_m', 'N/A')}M
World Cup titles: {team_stats.get('world_cup_titles', 0)}

**Group {group} opponents:** {', '.join(group_opponents)}

Write a 2-3 paragraph tournament outlook for {team}. Explain what the probability
distribution tells us about their ceiling and floor at this tournament.
Discuss where the model sees their path forward or the likely bottleneck.
Contextualise their win probability vs. historical base rates for a team of their profile."""


def group_summary_prompt(
    group: str,
    teams: list[str],
    advance_probs: dict[str, float],
    team_stats_map: dict[str, dict],
) -> str:
    teams_block = "\n".join(
        f"  {t}: advance prob {advance_probs.get(t, 0):.1%}, "
        f"ELO {team_stats_map[t].get('elo_rating', 'N/A')}, "
        f"FIFA #{team_stats_map[t].get('fifa_rank', 'N/A')}"
        for t in teams
    )
    return f"""Group {group} analysis from the Monte Carlo simulator:

{teams_block}

Write a 2-paragraph group analysis. Identify the favourites, the most dangerous
dark horse, and which matchup will likely decide who tops the group.
Keep it grounded in the numbers above."""


def upset_alert_prompt(
    underdog: str,
    favourite: str,
    underdog_win_prob: float,
    elo_diff: float,
    underdog_stats: dict,
    favourite_stats: dict,
) -> str:
    return f"""The model flags a potential upset scenario:

{favourite} (ELO {favourite_stats.get('elo_rating', 'N/A')}, ranked #{favourite_stats.get('fifa_rank', 'N/A')})
is favoured, but {underdog} (ELO {underdog_stats.get('elo_rating', 'N/A')}, ranked #{underdog_stats.get('fifa_rank', 'N/A')})
has a {underdog_win_prob:.1%} win probability according to the model.

ELO gap: {abs(elo_diff):.0f} points in {favourite}'s favour.
{underdog} recent form: {underdog_stats.get('recent_form', 'N/A')} | avg goals scored: {underdog_stats.get('avg_goals_scored', 'N/A')}
{favourite} avg goals conceded: {favourite_stats.get('avg_goals_conceded', 'N/A')}

In 2-3 sentences, explain what a {underdog_win_prob:.1%} upset probability actually means
in context and what conditions the model implies would need to occur for the upset to happen."""
