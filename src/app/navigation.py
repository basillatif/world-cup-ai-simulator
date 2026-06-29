"""Shared Streamlit navigation labels."""

from __future__ import annotations

import streamlit as st


ARCHIVE_PAGES = [
    ("Archive · Live Scorecard", "pages/20_Archive_·_Live_Scorecard.py"),
    ("Archive · Live Tournament Tracker", "pages/21_Archive_·_Live_Tournament_Tracker.py"),
    ("Archive · Game Predictions", "pages/22_Archive_·_Game_Predictions.py"),
    ("Archive · Tournament Results", "pages/23_Archive_·_Tournament_Results.py"),
    ("Archive · Team Deep-Dive", "pages/24_Archive_·_Team_Deep-Dive.py"),
    ("Archive · Group Analysis", "pages/25_Archive_·_Group_Analysis.py"),
]


def render_sidebar_navigation() -> None:
    st.sidebar.page_link("streamlit_app.py", label="Knockout Bracket", icon="🏆")
    for label, path in ARCHIVE_PAGES:
        st.sidebar.page_link(path, label=label)
