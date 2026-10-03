"""Ask MixLab: chat with the model's results. Claude calls the real analysis functions."""

import streamlit as st
from common import ai_context, setup

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

picker, _ = st.columns([1, 3])
tone = picker.selectbox("Answer as if for", list(TONE_PRESETS), format_func=TONE_LABELS.get)

question = st.chat_input("Ask about channels, budgets or what-if plans")
if not history:
    st.markdown("**Try one of these**")
    for suggestion in SUGGESTIONS:
        if st.button(suggestion, icon=":material/arrow_forward:"):
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
