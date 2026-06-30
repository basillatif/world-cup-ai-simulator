from __future__ import annotations

import os

import streamlit as st


st.set_page_config(page_title="Group Analysis", page_icon="⚽", layout="wide")
os.environ["WORLD_CUP_ARCHIVE_PAGE"] = "Group Analysis"

import src.app.archive_app  # noqa: E402,F401
