"""Shared Streamlit navigation labels."""

from __future__ import annotations

import streamlit as st


ARCHIVE_PAGES = [
    ("Archive · Live Scorecard", "src/app/pages/20_Archive_·_Live_Scorecard.py"),
    ("Archive · Live Tournament Tracker", "src/app/pages/21_Archive_·_Live_Tournament_Tracker.py"),
    ("Archive · Game Predictions", "src/app/pages/22_Archive_·_Game_Predictions.py"),
    ("Archive · Tournament Results", "src/app/pages/23_Archive_·_Tournament_Results.py"),
    ("Archive · Team Deep-Dive", "src/app/pages/24_Archive_·_Team_Deep-Dive.py"),
    ("Archive · Group Analysis", "src/app/pages/25_Archive_·_Group_Analysis.py"),
]


def render_sidebar_navigation() -> None:
    st.sidebar.page_link("src/app/streamlit_app.py", label="Knockout Bracket", icon="🏆")
    for label, path in ARCHIVE_PAGES:
        st.sidebar.page_link(path, label=label)
