"""Thin Streamlit chat UI for the churn data analyst agent.

Run with: ``streamlit run src/app/streamlit_app.py`` after setting either
``GROQ_API_KEY`` or ``OPENROUTER_API_KEY`` (and optionally LLM_PROVIDER).
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

# ``streamlit run src/app/streamlit_app.py`` executes this file as a script,
# so ensure the project root (rather than only ``src/app``) is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_VERSION = 2
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import ChatCompletionProvider, DataAnalystAgent, ProviderError


def load_project_env(env_file: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without overriding existing environment values."""
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def create_agent() -> DataAnalystAgent:
    """Build the real agent once per Streamlit session."""
    load_project_env()
    return DataAnalystAgent(ChatCompletionProvider.from_env())


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # Lets the rest of the project run without the UI extra installed.
        raise RuntimeError("Install Streamlit to run the chat UI") from exc

    st.set_page_config(page_title="Churn Data Analyst", page_icon="📊")
    st.title("Churn Data Analyst")
    st.caption("Answers are computed from local data and model tools before they are shown.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about churn data or customer risk")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Computing answer…"):
            try:
                if st.session_state.get("agent_version") != AGENT_VERSION:
                    st.session_state.agent = create_agent()
                    st.session_state.agent_version = AGENT_VERSION
                answer = st.session_state.agent.answer(prompt)
            except (ProviderError, ValueError, RuntimeError) as exc:
                answer = f"Unable to answer this request: {exc}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    run()
