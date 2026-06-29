from __future__ import annotations

import os

import streamlit as st


st.set_page_config(page_title="Archive · Game Predictions", page_icon="⚽", layout="wide")
st.caption("Group stage — completed, retained for the prediction record.")
os.environ["WORLD_CUP_ARCHIVE_PAGE"] = "Game Predictions"

import src.app.archive_app  # noqa: E402,F401
