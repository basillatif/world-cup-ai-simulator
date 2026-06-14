"""Streamlit UI for the World Cup AI Simulator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running with `streamlit run src/app/streamlit_app.py` from project root
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import streamlit as st

from cache.narration_cache import (
    LIVE_NARRATION_UNAVAILABLE,
    cache_key,
    canonical_probabilities,
    get_or_create_group_analysis,
)
from src.data.load_data import load_groups, load_matches, load_teams
from src.genai.analyst_agent import MODEL as CLAUDE_MODEL
from src.genai.analyst_agent import AnalystAgent
from src.models.elo import build_elo_from_seed
from src.models.match_predictor import MatchPredictor
from src.models.poisson_model import build_poisson_from_teams
from src.simulation.live_scorecard import render_live_scorecard
from src.simulation.live_tracker import (
    calculate_group_standings,
    compare_probabilities,
    normalize_results,
)
from src.simulation.prediction_store import save_predictions
from src.simulation.tournament_simulator import run_monte_carlo


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="World Cup AI Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ World Cup AI Simulator")
st.caption(
    "Monte Carlo tournament engine + Claude analyst layer. "
    "**GenAI explains the model. It does not replace it.**"
)


# ── Data & model loading (cached) ─────────────────────────────────────────────

@st.cache_data
def load_all_data():
    teams = load_teams()
    matches = load_matches()
    results_path = Path(__file__).parents[2] / "data" / "sample" / "results.csv"
    results = normalize_results(pd.read_csv(results_path, parse_dates=["date"]))
    groups = load_groups()
    return teams, matches, results, groups


@st.cache_resource
def build_models():
    # Zero-argument cache — avoids Streamlit pickling DataFrames on every
    # rerun just to compute the cache key, which was the source of slowness.
    _teams, _matches, _results, _ = load_all_data()
    completed = _results[_results["status"].str.casefold() == "final"].rename(
        columns={
            "team_a": "home_team",
            "team_b": "away_team",
            "score_a": "home_goals",
            "score_b": "away_goals",
        }
    )
    completed["tournament"] = "FIFA World Cup 2026"
    completed["neutral"] = True
    model_matches = pd.concat(
        [_matches, completed[_matches.columns]], ignore_index=True
    )
    elo = build_elo_from_seed(_teams, model_matches)
    poisson = build_poisson_from_teams(_teams)
    if len(model_matches) > 20:
        try:
            poisson.fit(model_matches)
        except Exception:
            pass  # fall back to team-seeded model
    return MatchPredictor(elo=elo, poisson=poisson)


with st.spinner("Loading data and building models…"):
    teams_df, matches_df, results_df, groups_df = load_all_data()
    predictor = build_models()
completed_history_df = results_df[
    results_df["status"].str.casefold() == "final"
].rename(
    columns={
        "team_a": "home_team",
        "team_b": "away_team",
        "score_a": "home_goals",
        "score_b": "away_goals",
    }
)
completed_history_df["tournament"] = "FIFA World Cup 2026"
completed_history_df["neutral"] = True
match_history_df = pd.concat(
    [matches_df, completed_history_df[matches_df.columns]], ignore_index=True
).sort_values("date")
all_teams = sorted(teams_df["team"].tolist())


# ── Sidebar navigation ────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigate",
    [
        "Group Analysis",
        "Live Scorecard",
        "Live Tournament Tracker",
        "Tournament Results",
        "Tournament Simulator",
        "Team Deep-Dive",
    ],
)


def configured_anthropic_api_key() -> str | None:
    api_key_input = st.sidebar.text_input(
        "Anthropic API Key (for AI analysis)",
        type="password",
        help=(
            "Used only for Claude commentary. You can also set "
            "ANTHROPIC_API_KEY or add it to .streamlit/secrets.toml."
        ),
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        return api_key_input

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = str(api_key)
        return str(api_key)
    return None


anthropic_api_key = configured_anthropic_api_key()


def get_analyst() -> AnalystAgent | None:
    if anthropic_api_key:
        return AnalystAgent(teams_df=teams_df, matches_df=match_history_df)
    return None


def probability_table(probabilities: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Format the simulator's probability output for Streamlit tables."""
    rows = []
    for team, values in sorted(
        probabilities.items(), key=lambda item: item[1]["champion"], reverse=True
    ):
        rows.append(
            {
                "Team": team,
                "Advance Group": values["group_advance"],
                "Reach R16": values["round_of_16"],
                "Reach QF": values["quarterfinal"],
                "Reach SF": values["semifinal"],
                "Reach Final": values["final"],
                "Win WC": values["champion"],
            }
        )
    return pd.DataFrame(rows)


def render_probability_table(probabilities: dict[str, dict[str, float]]) -> None:
    """Render tournament probabilities using a consistent table style."""
    table = probability_table(probabilities)
    st.dataframe(
        table.style.format({column: "{:.1%}" for column in table.columns if column != "Team"}),
        use_container_width=True,
        hide_index=True,
    )


def render_live_tournament_tracker() -> None:
    """Render actual results, standings, live probabilities, and movers."""
    st.header("Live Tournament Tracker")
    st.caption(
        "Final scores are locked into every live simulation; only remaining matches are simulated."
    )

    actual_tab, standings_tab, probabilities_tab, movers_tab = st.tabs(
        ["Actual Results", "Live Standings", "Updated Probabilities", "Biggest Movers"]
    )

    with actual_tab:
        display = results_df.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("The table is file-backed today and ready for a future editor or upload flow.")

    with standings_tab:
        standings = calculate_group_standings(results_df)
        if standings.empty:
            st.info("No final group results are available yet.")
        else:
            for group in standings["group"].unique():
                st.subheader(f"Group {group}")
                group_table = standings[standings["group"] == group].drop(columns="group")
                st.dataframe(group_table, use_container_width=True, hide_index=True)

    with probabilities_tab:
        st.write(
            "Run matched baseline and live simulations to isolate the effect of completed results."
        )
        col1, col2 = st.columns(2)
        live_sims = col1.number_input(
            "Simulations", min_value=500, max_value=20_000, value=5_000, step=500
        )
        live_seed = col2.number_input(
            "Comparison seed", min_value=0, value=42, key="live_tracker_seed"
        )
        if st.button("Recalculate Live Probabilities", type="primary"):
            with st.spinner(f"Running two sets of {int(live_sims):,} simulations..."):
                baseline = run_monte_carlo(
                    groups_df=groups_df,
                    predictor=predictor,
                    n_simulations=int(live_sims),
                    seed=int(live_seed),
                )
                updated = run_monte_carlo(
                    groups_df=groups_df,
                    predictor=predictor,
                    n_simulations=int(live_sims),
                    seed=int(live_seed),
                    completed_results_df=results_df,
                )
            st.session_state["live_tracker_results"] = {
                "baseline": baseline,
                "updated": updated,
            }

        live_results = st.session_state.get("live_tracker_results")
        if live_results:
            render_probability_table(live_results["updated"]["probabilities"])
        else:
            st.info("Recalculate to view probabilities based on the latest final results.")

    with movers_tab:
        live_results = st.session_state.get("live_tracker_results")
        if not live_results:
            st.info("Recalculate live probabilities first to see the biggest movers.")
        else:
            movers = compare_probabilities(
                live_results["baseline"]["probabilities"],
                live_results["updated"]["probabilities"],
            )
            percent_columns = [column for column in movers.columns if column != "team"]

            def highlight_change(value: float) -> str:
                if value > 0:
                    return "color: #14833b; font-weight: 600"
                if value < 0:
                    return "color: #c62828; font-weight: 600"
                return ""

            styled = movers.style.format(
                {column: "{:+.1%}" if column.endswith("change") else "{:.1%}" for column in percent_columns}
            ).map(highlight_change, subset=["advance_prob_change", "title_prob_change"])
            st.dataframe(styled, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def generate_group_analysis_live(
    narration_cache_key: str,
    group: str,
    teams: tuple[str, ...],
    rounded_advance_probs: tuple[tuple[str, float], ...],
) -> str:
    _ = narration_cache_key
    analyst = AnalystAgent(teams_df=teams_df, matches_df=match_history_df)
    return analyst.group_summary(
        group=group,
        teams=list(teams),
        advance_probs=dict(rounded_advance_probs),
    )


# ── Page: Group Analysis ──────────────────────────────────────────────────────

if page == "Group Analysis":
    st.header("Group Stage Analysis")

    selected_group = st.selectbox("Select Group", sorted(groups_df["group"].unique()))
    group_teams = groups_df[groups_df["group"] == selected_group]["team"].tolist()

    st.subheader(f"Group {selected_group} Teams")
    group_stats = teams_df[teams_df["team"].isin(group_teams)][
        ["team", "elo_rating", "fifa_rank", "avg_goals_scored", "avg_goals_conceded",
         "recent_form", "squad_value_m"]
    ].sort_values("elo_rating", ascending=False).rename(
        columns={"squad_value_m": "Squad Value (USD M)"}
    )
    st.dataframe(group_stats, use_container_width=True, hide_index=True)

    st.subheader("Intra-Group Match Predictions")
    rows = []
    for i, t1 in enumerate(group_teams):
        for t2 in group_teams[i + 1:]:
            completed = results_df[
                (results_df["status"].str.casefold() == "final")
                & (
                    ((results_df["team_a"] == t1) & (results_df["team_b"] == t2))
                    | ((results_df["team_a"] == t2) & (results_df["team_b"] == t1))
                )
            ]
            if not completed.empty:
                result = completed.iloc[-1]
                rows.append({
                    "Match": f"{t1} vs {t2}",
                    "Status": "Final",
                    "Result": (
                        f"{result['team_a']} {int(result['score_a'])}–"
                        f"{int(result['score_b'])} {result['team_b']}"
                    ),
                })
                continue

            p = predictor.predict_probs(t1, t2, neutral=True)
            xg = predictor.expected_goals_display(t1, t2, neutral=True)
            rows.append({
                "Match": f"{t1} vs {t2}",
                "Status": "Upcoming",
                f"{t1} Win": f"{p['home_win']:.1%}",
                "Draw": f"{p['draw']:.1%}",
                f"{t2} Win": f"{p['away_win']:.1%}",
                f"{t1} xG": xg["home_xg"],
                f"{t2} xG": xg["away_xg"],
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Run a quick simulation for group advance probs
    if st.button("Estimate Group Advance Probabilities (1,000 sims)"):
        with st.spinner("Simulating..."):
            quick = run_monte_carlo(groups_df=groups_df, predictor=predictor, n_simulations=1_000, seed=0)
        advance = {t: quick["probabilities"][t]["group_advance"] for t in group_teams}
        st.session_state["group_advance_estimate"] = {
            "group": selected_group,
            "teams": group_teams,
            "advance": advance,
        }
        st.session_state.pop("group_analysis_narration", None)

    estimate = st.session_state.get("group_advance_estimate")
    if estimate and estimate["group"] == selected_group:
        advance = estimate["advance"]
        estimate_teams = estimate["teams"]
        for t, p in sorted(advance.items(), key=lambda x: x[1], reverse=True):
            st.metric(t, f"{p:.1%}")

        if st.button("Claude Group Analysis"):
            rounded_advance = canonical_probabilities(advance)
            rounded_items = tuple(rounded_advance.items())
            narration_cache_key = cache_key(
                group=selected_group,
                probabilities=advance,
                model=CLAUDE_MODEL,
            )

            def live_narration() -> str:
                return generate_group_analysis_live(
                    narration_cache_key,
                    selected_group,
                    tuple(estimate_teams),
                    rounded_items,
                )

            with st.spinner("Generating group analysis..."):
                try:
                    commentary, status = get_or_create_group_analysis(
                        group=selected_group,
                        probabilities=advance,
                        has_api_key=bool(anthropic_api_key),
                        live_narration=live_narration,
                        model=CLAUDE_MODEL,
                    )
                except Exception as e:
                    st.error(f"Claude API error: {e}")
                else:
                    st.session_state["group_analysis_narration"] = {
                        "group": selected_group,
                        "narration": commentary,
                        "status": status,
                    }

        narration_state = st.session_state.get("group_analysis_narration")
        if narration_state and narration_state["group"] == selected_group:
            if narration_state["status"] == "hit":
                st.caption("Claude Group Analysis served from cache.")
            elif narration_state["status"] == "unavailable":
                st.warning(LIVE_NARRATION_UNAVAILABLE)
            if narration_state["narration"]:
                st.markdown(narration_state["narration"])


# ── Page: Live Scorecard ──────────────────────────────────────────────────────

elif page == "Live Scorecard":
    render_live_scorecard()


# ── Page: Live Tournament Tracker ─────────────────────────────────────────────

elif page == "Live Tournament Tracker":
    render_live_tournament_tracker()


# ── Page: Tournament Results ──────────────────────────────────────────────────

elif page == "Tournament Results":
    st.header("Tournament Results")
    st.caption("Completed matches are locked into all new tournament simulations.")

    display_results = results_df.copy()
    display_results["Date"] = display_results["date"].dt.strftime("%b %d, %Y")
    display_results["Match"] = display_results["team_a"] + " vs " + display_results["team_b"]
    display_results["Score"] = (
        display_results["score_a"].astype("Int64").astype(str)
        + "–"
        + display_results["score_b"].astype("Int64").astype(str)
    )
    display_results = display_results.rename(columns={"group": "Group"})
    st.dataframe(
        display_results[["Date", "Group", "Match", "Score"]],
        use_container_width=True,
        hide_index=True,
    )


# ── Page: Tournament Simulator ────────────────────────────────────────────────

elif page == "Tournament Simulator":
    st.header("Monte Carlo Tournament Simulator")

    n_sims = st.slider("Number of simulations", 1_000, 20_000, 5_000, step=1_000,
                        help="~0.7s per 1 000 sims. 5 000 gives stable results in ~3.5s.")
    seed = st.number_input("Random seed", value=42, min_value=0)

    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            results = run_monte_carlo(
                groups_df=groups_df,
                predictor=predictor,
                n_simulations=n_sims,
                seed=int(seed),
                completed_results_df=results_df,
            )
            saved_predictions = save_predictions(results, seed=int(seed))
        st.session_state["sim_results"] = results
        st.session_state["saved_predictions"] = saved_predictions
        st.success(f"Done! {n_sims:,} tournaments simulated.")
        st.caption(
            "Saved locally to "
            f"`pre-wc-predictions/{saved_predictions['json'].name}` and "
            f"`pre-wc-predictions/{saved_predictions['csv'].name}`."
        )

    if "sim_results" in st.session_state:
        results = st.session_state["sim_results"]
        probs = results["probabilities"]

        st.subheader("Championship Probabilities")
        render_probability_table(probs)

        # Top 5 bar chart
        st.subheader("Top 5 Contenders — Win Probability")
        top8 = results["top_contenders"]
        chart_df = pd.DataFrame(
            {"Team": [t for t, _ in top8], "Win Probability": [p["champion"] for _, p in top8]}
        ).set_index("Team")
        chart_df = chart_df.sort_values("Win Probability", ascending=False).head(5)
        st.bar_chart(chart_df)


# ── Page: Team Deep-Dive ──────────────────────────────────────────────────────

elif page == "Team Deep-Dive":
    st.header("Team Deep-Dive")

    selected_team = st.selectbox("Select Team", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
    team_stats = teams_df[teams_df["team"] == selected_team].iloc[0]
    team_group_row = groups_df[groups_df["team"] == selected_team]
    team_group = team_group_row["group"].iloc[0] if not team_group_row.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ELO Rating", int(team_stats["elo_rating"]))
    col2.metric("FIFA Rank", f"#{int(team_stats['fifa_rank'])}")
    col3.metric("Squad Value", f"${team_stats['squad_value_m']}M")
    col4.metric("World Cup Titles", int(team_stats["world_cup_titles"]))

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Goals Scored", team_stats["avg_goals_scored"])
    col6.metric("Avg Goals Conceded", team_stats["avg_goals_conceded"])
    col7.metric("Recent Form", team_stats["recent_form"])

    # Recent results from match history
    mask = ((match_history_df["home_team"] == selected_team)
            | (match_history_df["away_team"] == selected_team))
    recent_matches = match_history_df[mask].tail(8).sort_values("date", ascending=False)
    if not recent_matches.empty:
        st.subheader("Recent Results")
        display_rows = []
        for _, row in recent_matches.iterrows():
            if row["home_team"] == selected_team:
                display_rows.append({
                    "Date": row["date"].strftime("%Y-%m-%d"),
                    "Opponent": row["away_team"],
                    "Score": f"{int(row['home_goals'])}–{int(row['away_goals'])}",
                    "Venue": "Home",
                    "Tournament": row["tournament"],
                })
            else:
                display_rows.append({
                    "Date": row["date"].strftime("%Y-%m-%d"),
                    "Opponent": row["home_team"],
                    "Score": f"{int(row['away_goals'])}–{int(row['home_goals'])}",
                    "Venue": "Away",
                    "Tournament": row["tournament"],
                })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    # Tournament outlook
    if "sim_results" in st.session_state and st.button("Get Claude Tournament Outlook"):
        analyst = get_analyst()
        if not analyst:
            st.warning("Set ANTHROPIC_API_KEY to enable this.")
        else:
            team_probs = st.session_state["sim_results"]["probabilities"].get(selected_team, {})
            group_opponents = [
                t for t in groups_df[groups_df["group"] == team_group]["team"].tolist()
                if t != selected_team
            ]
            with st.spinner("Generating outlook..."):
                try:
                    outlook = analyst.tournament_outlook(
                        team=selected_team,
                        probs=team_probs,
                        group=team_group,
                        group_opponents=group_opponents,
                    )
                    st.markdown(outlook)
                except Exception as e:
                    st.error(f"Claude API error: {e}")
    elif "sim_results" not in st.session_state:
        st.info("Run the Tournament Simulator first to unlock the AI outlook.")
