"""MixLab dashboard entry point. Run with: uv run streamlit run app/main.py."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
CODE_DIRS = (APP_DIR, APP_DIR / "views", APP_DIR.parent / "src" / "mixlab")
STALE_PREFIXES = ("common", "charts", "mixlab")


def refresh_stale_modules() -> None:
    """Drop in-memory copies of this project's modules when the code on disk has changed.

    A hosted app keeps running after a new version is pulled. Streamlit re-reads this entry
    script on every run, but modules it imported earlier stay in memory, so a page that needs
    a newly added function fails with an ImportError until the process restarts. Comparing
    the newest file timestamp with the one seen last run lets the app heal itself.
    """
    stamp = max(path.stat().st_mtime for folder in CODE_DIRS for path in folder.glob("*.py"))
    previous = getattr(sys, "_mixlab_code_stamp", None)
    # ``previous is None`` covers a process that was already running before this check existed;
    # on a genuine cold start nothing is loaded yet, so purging is a no-op.
    if previous != stamp:
        for name in list(sys.modules):
            if name.split(".")[0] in STALE_PREFIXES:
                del sys.modules[name]
        st.cache_data.clear()
        st.cache_resource.clear()
    sys._mixlab_code_stamp = stamp  # type: ignore[attr-defined]


refresh_stale_modules()
st.set_page_config(page_title="MixLab", page_icon=":material/monitoring:", layout="wide")

PAGES = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/channels.py", title="Channel performance", icon=":material/bar_chart:"),
    st.Page("views/optimizer.py", title="Budget optimizer", icon=":material/tune:"),
    st.Page("views/scenarios.py", title="Scenario planner", icon=":material/compare_arrows:"),
    st.Page("views/ask.py", title="Ask MixLab", icon=":material/forum:"),
    st.Page("views/regions.py", title="Regions", icon=":material/map:"),
    st.Page("views/test_and_learn.py", title="Test and learn", icon=":material/science:"),
    st.Page("views/health.py", title="Model health", icon=":material/verified:"),
    st.Page("views/upload.py", title="Upload data", icon=":material/upload_file:"),
    st.Page("views/how_it_works.py", title="How it works", icon=":material/schema:"),
]

st.navigation(PAGES).run()
