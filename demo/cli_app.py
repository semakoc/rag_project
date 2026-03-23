"""
Terminal demo for researchers (IRB Protocol 26.0670).
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.llm import generate_response, safety_check
from src.logging_utils import log_interaction
from src.pipeline import build_pipeline
from src.router import route_query

load_dotenv()

_LAST_CONTEXT: list[str] = []


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_pipelines():
    root = _project_root()
    pl = build_pipeline(str(root / "data" / "pro_life"), "pro_life")
    pc = build_pipeline(str(root / "data" / "pro_choice"), "pro_choice")
    return pl, pc


def main() -> None:
    global _LAST_CONTEXT
    print("=== AI-Mediated Dialogue Research System ===")
    print("IRB Protocol 26.0670 | Boston College\n")
    print(
        "This tool helps you explore another group's materials with empathy.\n"
        "It is not built for debate or scoring points.\n"
    )

    try:
        pro_life_pipeline, pro_choice_pipeline = _load_pipelines()
    except Exception as exc:
        print(f"Could not load pipelines: {exc}")
        sys.exit(1)

    print("Select your group:")
    print("  [1] Pro-Life")
    print("  [2] Pro-Choice")
    choice = input("\n> ").strip()
    if choice == "1":
        user_group = "pro_life"
        label = "pro-choice perspective"
    elif choice == "2":
        user_group = "pro_choice"
        label = "pro-life perspective"
    else:
        print("Please choose 1 or 2.")
        return

    session_id = str(uuid.uuid4())
    print(f"\nYou are now chatting with the {label} chatbot.")
    print("Type 'quit' to exit, 'reset' to start a new session.")
    print("Type 'show context' to see retrieved passages.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            print("(empty input — type a question or 'quit'.)\n")
            continue
        if user.lower() == "quit":
            print("Goodbye.")
            break
        if user.lower() == "reset":
            session_id = str(uuid.uuid4())
            print("Started a new anonymous session.\n")
            continue
        if user.lower() == "show context":
            if not _LAST_CONTEXT:
                print("No context yet — ask a question first.\n")
            else:
                for i, ch in enumerate(_LAST_CONTEXT, 1):
                    print(f"[{i}] {ch}\n")
            continue

        if not safety_check(user):
            print(
                "Bot: I want to keep this space respectful. "
                "Could you rephrase that in calmer language?\n"
            )
            continue

        print("thinking...")
        try:
            routed = route_query(
                user_group,
                user,
                pro_life_pipeline,
                pro_choice_pipeline,
            )
            _LAST_CONTEXT = routed["context_chunks"]
            reply = generate_response(
                user,
                routed["context_chunks"],
                routed["source_group"],
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            log_interaction(
                session_id,
                user_group,
                user,
                reply,
                routed["context_chunks"],
            )
            print(f"Bot: {reply}\n")
        except Exception as exc:
            print(f"Bot: Something went wrong ({exc}). Please try again.\n")
        print("---")


if __name__ == "__main__":
    main()
