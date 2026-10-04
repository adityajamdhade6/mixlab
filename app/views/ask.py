"""Ask MixLab: chat with the model's results. Claude calls the real analysis functions."""

import streamlit as st
from common import ai_context, has_api_key, setup

from mixlab import config
from mixlab.ai_explainer import TONE_PRESETS, ExplainerError, ask

TONE_LABELS = {"cmo": "CMO", "analyst": "Analyst", "founder": "Founder"}
SUGGESTIONS = [
    "Why should I cut Meta?",
    "What happens if we pause TV for a quarter?",
    "Which channel has the most room to grow?",
]

brand, results = setup(
    "Ask MixLab",
    "Ask a question in plain English. Answers come from the fitted model, not from guesswork.",
)
history_key = f"chat_{brand}"
history: list[dict[str, str]] = st.session_state.setdefault(history_key, [])

if not has_api_key():
    st.warning(
        "Ask MixLab needs a Claude API key, and none is set for this copy of the app, so "
        "questions cannot be answered here. Everything else works without it. To try the chat, "
        f"run the app locally with a key in `.env` ([setup steps]({config.REPO_URL}#quickstart))."
    )

picker, _ = st.columns([1, 3])
tone = picker.selectbox("Answer as if for", list(TONE_PRESETS), format_func=TONE_LABELS.get)

question = st.chat_input("Ask about channels, budgets or what-if plans", disabled=not has_api_key())
if not history:
    st.markdown("**Try one of these**")
    for suggestion in SUGGESTIONS:
        if st.button(suggestion, icon=":material/arrow_forward:", disabled=not has_api_key()):
            question = suggestion

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("note"):
            st.caption(message["note"])

if question:
    history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the model…"):
                answer = ask(ai_context(brand, with_model=True), question, tone)
        except ExplainerError as error:
            st.error(str(error))
            history.pop()
        else:
            tools = ", ".join(dict.fromkeys(call.name for call in answer.tool_calls))
            note = (f"Looked up: {tools}. " if tools else "") + answer.number_check.summary()
            st.markdown(answer.text)
            st.caption(note)
            history.append({"role": "assistant", "text": answer.text, "note": note})

if history and st.button("Clear conversation"):
    st.session_state[history_key] = []
    st.rerun()
