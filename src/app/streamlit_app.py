"""Default Streamlit landing page: Road to the Final."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running with `streamlit run src/app/streamlit_app.py` from project root.
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import streamlit as st

from knockout_bracket import R32_SEEDING, simulate_bracket
from knockout_engine import knockout_match_fn
from knockout_results import build_decided, verify_seeding
from src.app.ui_components import apply_custom_theme, get_flag, render_probability_bar
from src.data.load_data import load_results
from src.data.results_updater import (
    ResultsFetchError,
    fetch_and_merge_results,
    fetch_live_round_of_32_fixtures,
)


FINISHED: dict[int, str] = {}


def prepare_bracket_data(
    n: int = 10_000,
    decided: dict[int, str] = FINISHED,
) -> pd.DataFrame:
    results = simulate_bracket(knockout_match_fn, n=n, decided=decided)
    rows = []
    for team, values in results.items():
        rows.append(
            {
                "Team": team,
                "Champion %": values["champion"],
                "Reach SF %": values["reach"]["SF"],
                "Reach Final %": values["reach"]["F"],
                "Reach QF %": values["reach"]["QF"],
                "Reach R16 %": values["reach"]["R16"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Champion %", "Reach SF %", "Team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def r32_matchups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Match": match_id, "Team A": team_a, "Team B": team_b}
            for match_id, (team_a, team_b) in sorted(R32_SEEDING.items())
        ]
    )


def load_conditioned_decisions() -> tuple[dict[int, str], str | None]:
    latest_results = fetch_and_merge_results(load_results())
    try:
        live_r32_fixtures = fetch_live_round_of_32_fixtures()
        verify_seeding(live_r32_fixtures)
    except (ResultsFetchError, ValueError) as exc:
        return FINISHED.copy(), str(exc)
    return build_decided(latest_results), None


def render() -> None:
    st.set_page_config(
        page_title="Road to the Final",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_theme()

    st.title("Knockout Bracket")
    st.caption("Road to the Final")

    decided, seeding_error = load_conditioned_decisions()
    if seeding_error:
        st.error(
            "Live seeding integrity check failed. Running the frozen bracket "
            f"without live-result conditioning. {seeding_error}"
        )

    with st.spinner("Simulating the knockout bracket..."):
        bracket = prepare_bracket_data(decided=decided)

    leader = bracket.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Champion", f"{get_flag(leader['Team'])} {leader['Team']}")
    col2.metric("Champion", f"{leader['Champion %']:.0%}")
    col3.metric("Reach SF", f"{leader['Reach SF %']:.0%}")

    st.subheader("Title Race")
    for _, row in bracket.head(12).iterrows():
        label = f"{get_flag(row['Team'])} {row['Team']}"
        render_probability_bar(label, row["Champion %"], accent="gold")

    st.subheader("Model Probabilities")
    display = bracket.copy()
    percent_columns = [column for column in display.columns if column != "Team"]
    st.dataframe(
        display.style.format({column: "{:.0%}" for column in percent_columns}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Round of 32")
    st.dataframe(r32_matchups(), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
