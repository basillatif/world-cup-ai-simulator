"""Reusable presentation helpers for the Streamlit UI.

This module is presentation-only: it injects CSS and renders small HTML
snippets via ``st.markdown(..., unsafe_allow_html=True)``. It contains no
data loading, simulation, or prediction logic — those stay in
``src/data``, ``src/models``, and ``src/simulation``.

Note on markdown/HTML strings: every snippet passed to ``st.markdown(...,
unsafe_allow_html=True)`` must start with ``<`` at column 0 (no leading
blank line or indentation) and must not contain blank lines. Markdown
treats a line indented by 4+ spaces as a code block, and an HTML block
ends at the first blank line — either mistake causes the raw HTML to be
shown as literal text instead of being rendered.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st


# ── FIFA-inspired palette ──────────────────────────────────────────────────

COLOR_BG = "#0A1F3D"
COLOR_BLUE = "#0057B8"
COLOR_GOLD = "#FFD700"
COLOR_TEXT = "#F5F7FA"
COLOR_MUTED = "#A9B7CC"


# ── Country flag emoji lookup ───────────────────────────────────────────────

_ENGLAND_FLAG = "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F"
_SCOTLAND_FLAG = "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F"

FLAGS: dict[str, str] = {
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷",
    "Canada": "🇨🇦",
    "Cape Verde": "🇨🇻",
    "Colombia": "🇨🇴",
    "Croatia": "🇭🇷",
    "Curacao": "🇨🇼",
    "Czechia": "🇨🇿",
    "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": _ENGLAND_FLAG,
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Haiti": "🇭🇹",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Ivory Coast": "🇨🇮",
    "Japan": "🇯🇵",
    "Jordan": "🇯🇴",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": _SCOTLAND_FLAG,
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}


def get_flag(team: str) -> str:
    """Return the flag emoji for a team, or a neutral flag if unknown."""
    return FLAGS.get(team, "🏳️")


# ── Theme ────────────────────────────────────────────────────────────────────

def apply_custom_theme() -> None:
    """Inject global CSS for the FIFA-style navy dashboard theme.

    Call this once, immediately after ``st.set_page_config``.
    """
    st.markdown(
        f"""<style>
        :root {{
            --wc-bg: {COLOR_BG};
            --wc-border: rgba(255, 255, 255, 0.12);
            --wc-blue: {COLOR_BLUE};
            --wc-blue-light: #3B82F6;
            --wc-gold: {COLOR_GOLD};
            --wc-text: {COLOR_TEXT};
            --wc-muted: {COLOR_MUTED};
        }}
        .stApp {{
            background:
                radial-gradient(circle at 10% 0%, rgba(0, 87, 184, 0.28), transparent 45%),
                radial-gradient(circle at 90% 10%, rgba(255, 215, 0, 0.10), transparent 40%),
                var(--wc-bg);
            color: var(--wc-text);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #123B66 0%, var(--wc-bg) 100%);
            border-right: 1px solid var(--wc-border);
        }}
        [data-testid="stHeader"] {{
            background: rgba(10, 31, 61, 0.0);
        }}
        h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
            color: var(--wc-text);
        }}
        .wc-hero {{
            background: linear-gradient(135deg, rgba(0, 87, 184, 0.55) 0%, rgba(0, 40, 95, 0.65) 55%, rgba(10, 31, 61, 0.92) 100%);
            border: 1px solid var(--wc-border);
            border-radius: 20px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
            animation: wc-fade-in 0.6s ease-out;
        }}
        .wc-hero-title {{
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--wc-text);
            margin: 0;
        }}
        .wc-hero-subtitle {{
            font-size: 1.05rem;
            color: rgba(245, 247, 250, 0.88);
            margin-top: 0.4rem;
        }}
        .wc-hero-meta {{
            font-size: 0.78rem;
            color: var(--wc-gold);
            margin-top: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .wc-card {{
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border: 1px solid var(--wc-border);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
            animation: wc-fade-in 0.45s ease-out;
        }}
        .wc-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
            border-color: rgba(255, 215, 0, 0.35);
        }}
        .wc-metric-card {{
            text-align: center;
            padding: 1.1rem 0.75rem;
        }}
        .wc-metric-icon {{ font-size: 1.6rem; margin-bottom: 0.35rem; }}
        .wc-metric-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--wc-text);
            line-height: 1.1;
        }}
        .wc-metric-label {{
            font-size: 0.8rem;
            color: var(--wc-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.25rem;
        }}
        .wc-metric-help {{
            font-size: 0.72rem;
            color: var(--wc-muted);
            margin-top: 0.35rem;
        }}
        .wc-champion-card {{
            text-align: center;
            padding: 1.75rem 1rem;
            background: linear-gradient(160deg, rgba(255, 215, 0, 0.18), rgba(255, 215, 0, 0.03));
            border: 1px solid rgba(255, 215, 0, 0.45);
        }}
        .wc-champion-label {{
            font-size: 0.85rem;
            font-weight: 800;
            color: var(--wc-gold);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.5rem;
        }}
        .wc-champion-flag {{ font-size: 3.5rem; line-height: 1; margin-bottom: 0.25rem; }}
        .wc-champion-team {{
            font-size: 2.4rem;
            font-weight: 800;
            color: var(--wc-text);
            margin-bottom: 0.35rem;
        }}
        .wc-champion-prob {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--wc-gold);
        }}
        .wc-match-card {{ padding: 0.85rem 1.1rem; }}
        .wc-match-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            flex-wrap: wrap;
            gap: 0.4rem;
        }}
        .wc-match-meta {{ font-size: 0.75rem; color: var(--wc-muted); }}
        .wc-match-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}
        .wc-team {{
            flex: 1;
            min-width: 110px;
            font-size: 1rem;
            font-weight: 600;
            color: var(--wc-text);
        }}
        .wc-team:last-child {{ text-align: right; }}
        .wc-team-winner {{ color: var(--wc-gold); font-weight: 800; }}
        .wc-flag {{ margin: 0 0.4rem; font-size: 1.2rem; }}
        .wc-score {{
            font-size: 1.3rem;
            font-weight: 800;
            color: var(--wc-text);
            background: rgba(0, 87, 184, 0.30);
            border-radius: 10px;
            padding: 0.2rem 0.85rem;
            min-width: 90px;
            text-align: center;
        }}
        .wc-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .wc-badge-final {{ background: rgba(255, 255, 255, 0.10); color: var(--wc-muted); }}
        .wc-badge-live {{ background: rgba(239, 68, 68, 0.20); color: #FCA5A5; animation: wc-pulse 1.6s infinite; }}
        .wc-badge-upcoming {{ background: rgba(59, 130, 246, 0.20); color: #93C5FD; }}
        .wc-group-card {{ padding: 0.9rem 1rem 0.6rem; }}
        .wc-group-header {{
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--wc-gold);
            margin-bottom: 0.5rem;
            letter-spacing: 0.02em;
        }}
        .wc-group-row {{
            display: grid;
            grid-template-columns: 2.4fr 0.6fr 1fr 0.6fr 0.6fr;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.5rem;
            border-radius: 8px;
            font-size: 0.85rem;
        }}
        .wc-group-row-header {{
            color: var(--wc-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
            padding-bottom: 0.3rem;
        }}
        .wc-group-row-alt {{ background: rgba(255, 255, 255, 0.04); }}
        .wc-group-row-qualified {{
            background: rgba(255, 215, 0, 0.10);
            border: 1px solid rgba(255, 215, 0, 0.35);
        }}
        .wc-group-team {{
            font-weight: 600;
            color: var(--wc-text);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }}
        .wc-group-stat {{ text-align: center; color: var(--wc-text); }}
        .wc-group-points {{ font-weight: 800; color: var(--wc-gold); }}
        .wc-qualify-badge {{
            background: var(--wc-blue);
            color: white;
            font-size: 0.65rem;
            font-weight: 800;
            border-radius: 5px;
            padding: 0.05rem 0.35rem;
            margin-right: 0.2rem;
        }}
        .wc-probbar {{ margin-bottom: 0.6rem; }}
        .wc-probbar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--wc-text);
            margin-bottom: 0.2rem;
        }}
        .wc-probbar-value {{ font-weight: 700; color: var(--wc-gold); }}
        .wc-probbar-track {{
            height: 10px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.10);
            overflow: hidden;
        }}
        .wc-bar-fill-blue, .wc-bar-fill-gold {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s ease;
        }}
        .wc-bar-fill-blue {{ background: linear-gradient(90deg, var(--wc-blue), var(--wc-blue-light)); }}
        .wc-bar-fill-gold {{ background: linear-gradient(90deg, #FFB800, var(--wc-gold)); }}
        [data-testid="stMetric"] {{
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
            border: 1px solid var(--wc-border);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        }}
        [data-testid="stMetricValue"] {{ color: var(--wc-text); }}
        [data-testid="stMetricLabel"] {{ color: var(--wc-muted); }}
        @keyframes wc-fade-in {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes wc-pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.55; }}
        }}
        .wc-pred-card {{ padding: 1rem 1.25rem; }}
        .wc-pred-meta {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.75rem;
            color: var(--wc-muted);
            margin-bottom: 0.65rem;
            flex-wrap: wrap;
        }}
        .wc-pred-teams {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .wc-pred-team {{
            flex: 1;
            font-size: 1rem;
            font-weight: 700;
            color: var(--wc-text);
        }}
        .wc-pred-team-b {{ text-align: right; }}
        .wc-pred-vs {{
            font-size: 0.8rem;
            color: var(--wc-muted);
            font-weight: 600;
            flex-shrink: 0;
            padding: 0 0.25rem;
        }}
        .wc-pred-probs {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.4rem;
            margin-bottom: 0.7rem;
        }}
        .wc-pred-prob-cell {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--wc-border);
            border-radius: 10px;
            text-align: center;
            padding: 0.45rem 0.3rem;
        }}
        .wc-pred-prob-cell-highlight {{
            background: rgba(0, 87, 184, 0.20);
            border-color: rgba(59, 130, 246, 0.45);
        }}
        .wc-pred-prob-label {{
            font-size: 0.68rem;
            color: var(--wc-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
        }}
        .wc-pred-prob-value {{
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--wc-text);
        }}
        .wc-pred-footer {{
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            flex-wrap: wrap;
        }}
        .wc-pred-scoreline {{
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--wc-gold);
            background: rgba(255, 215, 0, 0.12);
            border: 1px solid rgba(255, 215, 0, 0.30);
            border-radius: 6px;
            padding: 0.15rem 0.55rem;
            white-space: nowrap;
        }}
        .wc-pred-confidence-high {{
            font-size: 0.72rem; font-weight: 700;
            background: rgba(20, 131, 59, 0.20); color: #86EFAC;
            border: 1px solid rgba(20, 131, 59, 0.35);
            border-radius: 6px; padding: 0.15rem 0.55rem; white-space: nowrap;
        }}
        .wc-pred-confidence-medium {{
            font-size: 0.72rem; font-weight: 700;
            background: rgba(234, 179, 8, 0.20); color: #FDE047;
            border: 1px solid rgba(234, 179, 8, 0.35);
            border-radius: 6px; padding: 0.15rem 0.55rem; white-space: nowrap;
        }}
        .wc-pred-confidence-low {{
            font-size: 0.72rem; font-weight: 700;
            background: rgba(148, 163, 184, 0.15); color: var(--wc-muted);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 6px; padding: 0.15rem 0.55rem; white-space: nowrap;
        }}
        .wc-pred-explanation {{
            font-size: 0.78rem;
            color: var(--wc-muted);
            line-height: 1.4;
            flex: 1;
            min-width: 120px;
        }}
        @media (max-width: 640px) {{
            .wc-hero {{ padding: 1.25rem 1.25rem; }}
            .wc-hero-title {{ font-size: 1.6rem; }}
            .wc-group-row {{ grid-template-columns: 1.8fr 0.5fr 0.9fr 0.5fr 0.5fr; font-size: 0.75rem; }}
            .wc-match-row {{ flex-direction: column; align-items: stretch; text-align: center; }}
            .wc-team:last-child {{ text-align: center; }}
            .wc-score {{ margin: 0.3rem auto; }}
        }}
        </style>""",
        unsafe_allow_html=True,
    )


# ── Hero header ──────────────────────────────────────────────────────────────

def render_hero_header(subtitle: str, last_updated: str | None = None) -> None:
    """Render the top-of-page hero banner."""
    timestamp = last_updated or datetime.now().strftime("%b %d, %Y · %H:%M")
    st.markdown(
        f"""<div class="wc-hero">
<div class="wc-hero-title">⚽ FIFA World Cup 2026 Predictor</div>
<div class="wc-hero-subtitle">{subtitle}</div>
<div class="wc-hero-meta">Last updated: {timestamp}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Metric card ──────────────────────────────────────────────────────────────

def render_metric_card(label: str, value: str, icon: str = "", help_text: str | None = None) -> None:
    """Render a single KPI card. Place inside a column for a grid layout."""
    help_html = f'<div class="wc-metric-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""<div class="wc-card wc-metric-card">
<div class="wc-metric-icon">{icon}</div>
<div class="wc-metric-value">{value}</div>
<div class="wc-metric-label">{label}</div>{help_html}
</div>""",
        unsafe_allow_html=True,
    )


# ── Champion card ────────────────────────────────────────────────────────────

def render_champion_card(team: str, probability: float) -> None:
    """Render a large, gold-highlighted hero card for the top simulation pick."""
    pct = max(0.0, min(1.0, float(probability))) * 100
    st.markdown(
        f"""<div class="wc-card wc-champion-card">
<div class="wc-champion-label">🏆 Predicted Champion</div>
<div class="wc-champion-flag">{get_flag(team)}</div>
<div class="wc-champion-team">{team}</div>
<div class="wc-champion-prob">{pct:.1f}% win probability</div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Status badge ─────────────────────────────────────────────────────────────

def render_status_badge(status: str) -> str:
    """Return an inline HTML status badge: Final / Live / Upcoming."""
    normalized = (status or "").strip().casefold()
    if normalized == "final":
        return '<span class="wc-badge wc-badge-final">⚪ Final</span>'
    if normalized in {"live", "in_play", "in play", "paused"}:
        return '<span class="wc-badge wc-badge-live">🔴 Live</span>'
    return '<span class="wc-badge wc-badge-upcoming">🔵 Upcoming</span>'


# ── Match card ───────────────────────────────────────────────────────────────

def render_match_card(
    team_a: str,
    team_b: str,
    score_a: float | int | None,
    score_b: float | int | None,
    status: str = "Scheduled",
    date: str | None = None,
    group: str | None = None,
) -> None:
    """Render a single match as a scoreboard card."""
    has_score = (
        score_a is not None
        and score_b is not None
        and not (pd.isna(score_a) or pd.isna(score_b))
    )
    score_a_disp = str(int(score_a)) if has_score else "–"
    score_b_disp = str(int(score_b)) if has_score else "–"

    a_winner = has_score and score_a > score_b
    b_winner = has_score and score_b > score_a

    a_class = "wc-team wc-team-winner" if a_winner else "wc-team"
    b_class = "wc-team wc-team-winner" if b_winner else "wc-team"

    meta_bits = [bit for bit in (f"Group {group}" if group else None, date) if bit]
    meta_html = f'<div class="wc-match-meta">{" · ".join(meta_bits)}</div>' if meta_bits else ""

    st.markdown(
        f"""<div class="wc-card wc-match-card">
<div class="wc-match-top">{render_status_badge(status)}{meta_html}</div>
<div class="wc-match-row">
<div class="{a_class}"><span class="wc-flag">{get_flag(team_a)}</span>{team_a}</div>
<div class="wc-score">{score_a_disp} – {score_b_disp}</div>
<div class="{b_class}">{team_b}<span class="wc-flag">{get_flag(team_b)}</span></div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Group standings card ─────────────────────────────────────────────────────

def render_group_card(group_name: str, standings_df: pd.DataFrame, highlight_n: int = 2) -> None:
    """Render a group standings table as a compact card.

    Expects ``standings_df`` with columns: team, played, wins, draws, losses,
    goals_for, goals_against, goal_diff, points (already sorted by rank).
    The top ``highlight_n`` rows are highlighted as qualifiers.
    """
    rows_html = ""
    for i, row in enumerate(standings_df.itertuples(index=False)):
        is_top = i < highlight_n
        row_classes = ["wc-group-row"]
        if is_top:
            row_classes.append("wc-group-row-qualified")
        elif i % 2 == 1:
            row_classes.append("wc-group-row-alt")
        badge = '<span class="wc-qualify-badge">Q</span>' if is_top else ""

        rows_html += (
            f'<div class="{" ".join(row_classes)}">'
            f'<div class="wc-group-team">{badge}<span class="wc-flag">{get_flag(row.team)}</span>{row.team}</div>'
            f'<div class="wc-group-stat">{row.played}</div>'
            f'<div class="wc-group-stat">{row.wins}-{row.draws}-{row.losses}</div>'
            f'<div class="wc-group-stat">{int(row.goal_diff):+d}</div>'
            f'<div class="wc-group-stat wc-group-points">{row.points}</div>'
            f'</div>'
        )

    st.markdown(
        f"""<div class="wc-card wc-group-card">
<div class="wc-group-header">Group {group_name}</div>
<div class="wc-group-row wc-group-row-header">
<div class="wc-group-team">Team</div>
<div class="wc-group-stat">P</div>
<div class="wc-group-stat">W-D-L</div>
<div class="wc-group-stat">GD</div>
<div class="wc-group-stat wc-group-points">Pts</div>
</div>{rows_html}
</div>""",
        unsafe_allow_html=True,
    )


# ── Match prediction card ────────────────────────────────────────────────────

def render_prediction_card(
    team_a: str,
    team_b: str,
    team_a_win: float,
    draw: float,
    team_b_win: float,
    scoreline: str,
    confidence: str,
    explanation: str,
    date: str | None = None,
    group: str | None = None,
) -> None:
    """Render a single upcoming-match prediction as a styled card.

    Parameters
    ----------
    team_a / team_b : Team names (used for flag lookup).
    team_a_win / draw / team_b_win : Win probabilities (0-1, should sum to 1).
    scoreline : Predicted score string, e.g. ``"1–0"``.
    confidence : ``"High"``, ``"Medium"``, or ``"Low"``.
    explanation : One-sentence model explanation to show below the card.
    date / group : Optional metadata shown in the card header.
    """
    # Highlight the cell for the outcome with the highest probability
    a_is_fav = team_a_win >= draw and team_a_win >= team_b_win
    d_is_fav = draw > team_a_win and draw >= team_b_win
    b_is_fav = not a_is_fav and not d_is_fav

    a_cell = "wc-pred-prob-cell wc-pred-prob-cell-highlight" if a_is_fav else "wc-pred-prob-cell"
    d_cell = "wc-pred-prob-cell wc-pred-prob-cell-highlight" if d_is_fav else "wc-pred-prob-cell"
    b_cell = "wc-pred-prob-cell wc-pred-prob-cell-highlight" if b_is_fav else "wc-pred-prob-cell"

    conf_class = {
        "High": "wc-pred-confidence-high",
        "Medium": "wc-pred-confidence-medium",
    }.get(confidence, "wc-pred-confidence-low")
    conf_icon = {"High": "●", "Medium": "◑"}.get(confidence, "○")

    meta_bits = [
        f"Group {group}" if group else None,
        date if date else None,
    ]
    meta_html = (
        '<div class="wc-pred-meta">'
        + "".join(
            f'<span>{b}</span><span style="color:rgba(255,255,255,0.2)">·</span>'
            for b in meta_bits
            if b
        ).rstrip('<span style="color:rgba(255,255,255,0.2)">·</span>')
        + "</div>"
        if any(meta_bits)
        else ""
    )

    st.markdown(
        f"""<div class="wc-card wc-pred-card">
{meta_html}<div class="wc-pred-teams">
<div class="wc-pred-team"><span class="wc-flag">{get_flag(team_a)}</span>{team_a}</div>
<div class="wc-pred-vs">vs</div>
<div class="wc-pred-team wc-pred-team-b">{team_b}<span class="wc-flag">{get_flag(team_b)}</span></div>
</div>
<div class="wc-pred-probs">
<div class="{a_cell}"><div class="wc-pred-prob-label">{team_a}</div><div class="wc-pred-prob-value">{team_a_win:.0%}</div></div>
<div class="{d_cell}"><div class="wc-pred-prob-label">Draw</div><div class="wc-pred-prob-value">{draw:.0%}</div></div>
<div class="{b_cell}"><div class="wc-pred-prob-label">{team_b}</div><div class="wc-pred-prob-value">{team_b_win:.0%}</div></div>
</div>
<div class="wc-pred-footer">
<span class="wc-pred-scoreline">⚽ {scoreline}</span>
<span class="{conf_class}">{conf_icon} {confidence} confidence</span>
<span class="wc-pred-explanation">{explanation}</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Probability bar ──────────────────────────────────────────────────────────

def render_probability_bar(label: str, value: float, accent: str = "blue") -> None:
    """Render a horizontal probability bar (0.0-1.0) with a percentage label."""
    pct = max(0.0, min(1.0, float(value))) * 100
    fill_class = "wc-bar-fill-gold" if accent == "gold" else "wc-bar-fill-blue"
    st.markdown(
        f"""<div class="wc-probbar">
<div class="wc-probbar-label">
<span>{label}</span>
<span class="wc-probbar-value">{pct:.1f}%</span>
</div>
<div class="wc-probbar-track">
<div class="{fill_class}" style="width: {pct:.2f}%;"></div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
