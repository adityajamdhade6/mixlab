"""MixLab dashboard entry point. Run with: uv run streamlit run app/main.py."""

import streamlit as st

st.set_page_config(page_title="MixLab", page_icon=":material/monitoring:", layout="wide")

PAGES = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/channels.py", title="Channel performance", icon=":material/bar_chart:"),
    st.Page("views/optimizer.py", title="Budget optimizer", icon=":material/tune:"),
    st.Page("views/scenarios.py", title="Scenario planner", icon=":material/compare_arrows:"),
    st.Page("views/ask.py", title="Ask MixLab", icon=":material/forum:"),
    st.Page("views/health.py", title="Model health", icon=":material/verified:"),
    st.Page("views/upload.py", title="Upload data", icon=":material/upload_file:"),
]

st.navigation(PAGES).run()
