"""Shared Streamlit navigation labels."""

from __future__ import annotations

import streamlit as st


ARCHIVE_PAGES = [
    ("Live Scorecard", "pages/20_Live_Scorecard.py"),
    ("Live Tournament Tracker", "pages/21_Live_Tournament_Tracker.py"),
    ("Game Predictions", "pages/22_Game_Predictions.py"),
    ("Tournament Results", "pages/23_Tournament_Results.py"),
    ("Team Deep-Dive", "pages/24_Team_Deep-Dive.py"),
    ("Group Analysis", "pages/25_Group_Analysis.py"),
]


def render_sidebar_navigation() -> None:
    st.sidebar.page_link("streamlit_app.py", label="Knockout Bracket", icon="🏆")
    for label, path in ARCHIVE_PAGES:
        st.sidebar.page_link(path, label=label)
