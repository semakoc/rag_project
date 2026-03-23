"""
Streamlit interface for study participants (IRB Protocol 26.0670).
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is importable when launched via `streamlit run`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.llm import generate_response, safety_check
from src.logging_utils import log_interaction
from src.pipeline import build_pipeline
from src.router import route_query

load_dotenv()

st.set_page_config(
    page_title="AI-Mediated Dialogue Study",
    page_icon="🤝",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipelines():
    """Cache pipelines so they don't reload on every interaction."""
    root = Path(__file__).resolve().parent.parent
    pro_life = build_pipeline(str(root / "data" / "pro_life"), "pro_life")
    pro_choice = build_pipeline(str(root / "data" / "pro_choice"), "pro_choice")
    return pro_life, pro_choice


def _init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_group" not in st.session_state:
        st.session_state.user_group = "pro_life"
    if "last_context" not in st.session_state:
        st.session_state.last_context = []


def main() -> None:
    _init_session()

    with st.sidebar:
        st.title("Study information")
        st.caption("IRB Protocol 26.0670 · Boston College")
        st.markdown(
            "**AI-mediated dialogue for intergroup understanding** — "
            "explore the other group's materials in a supportive way."
        )
        group_label = st.selectbox(
            "Your participation group",
            (
                "I am a pro-life participant",
                "I am a pro-choice participant",
            ),
        )
        st.session_state.user_group = (
            "pro_life" if group_label.startswith("I am a pro-life") else "pro_choice"
        )
        if st.button("Reset session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_context = []
            st.rerun()

        with st.expander("About this study"):
            st.write(
                "This app helps you read and discuss materials from students "
                "who hold a different view, using an AI assistant trained on their "
                "documents. The goal is understanding and empathy, not winning an argument."
            )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 720px; }
        div[data-testid="stChatMessage"] { background-color: #f6f7fb; border-radius: 12px; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Guided conversation")
    st.write(
        "Welcome. Choose your group in the sidebar, then share a question or reflection. "
        "The assistant draws on the *other* group's knowledge base and responds with warmth and curiosity."
    )

    try:
        pro_life_pipeline, pro_choice_pipeline = load_pipelines()
    except Exception as exc:
        st.error(
            f"Could not build the knowledge bases ({exc}). "
            "Add PDF or TXT files under data/pro_life/ and data/pro_choice/, then refresh."
        )
        return

    if (
        pro_life_pipeline["chunk_count"] == 0
        and pro_choice_pipeline["chunk_count"] == 0
    ):
        st.warning(
            "Both data folders are empty. Add at least one PDF or TXT file to each folder to enable retrieval."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("context"):
                with st.expander("View retrieved context"):
                    for i, ch in enumerate(msg["context"], 1):
                        st.markdown(f"**[{i}]** {ch}")

    prompt = st.chat_input("Type your message", disabled=False)

    if prompt:
        if not safety_check(prompt):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I want to keep this space respectful. "
                        "Could you rephrase that in calmer language?"
                    ),
                    "context": [],
                }
            )
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                routed = route_query(
                    st.session_state.user_group,
                    prompt,
                    pro_life_pipeline,
                    pro_choice_pipeline,
                )
                st.session_state.last_context = routed["context_chunks"]
                reply = generate_response(
                    prompt,
                    routed["context_chunks"],
                    routed["source_group"],
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
                log_interaction(
                    st.session_state.session_id,
                    st.session_state.user_group,
                    prompt,
                    reply,
                    routed["context_chunks"],
                )
            except Exception as exc:
                reply = f"Something went wrong: {exc}"
                st.session_state.last_context = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "context": list(st.session_state.last_context),
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
