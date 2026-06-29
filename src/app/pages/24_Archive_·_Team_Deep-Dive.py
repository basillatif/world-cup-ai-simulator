from __future__ import annotations

import os

import streamlit as st


st.set_page_config(page_title="Archive · Team Deep-Dive", page_icon="⚽", layout="wide")
st.caption("Group stage — completed, retained for the prediction record.")
os.environ["WORLD_CUP_ARCHIVE_PAGE"] = "Team Deep-Dive"

import src.app.archive_app  # noqa: E402,F401
