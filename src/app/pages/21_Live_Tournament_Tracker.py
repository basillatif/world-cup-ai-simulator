from __future__ import annotations

import os

import streamlit as st


st.set_page_config(page_title="Live Tournament Tracker", page_icon="⚽", layout="wide")
os.environ["WORLD_CUP_ARCHIVE_PAGE"] = "Live Tournament Tracker"

import src.app.archive_app  # noqa: E402,F401
