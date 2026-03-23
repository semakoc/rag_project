"""
main.py — Quick demo of the dual RAG system.

Usage:
    python main.py

Builds both pipelines from data/ folders, then runs a single
example interaction to verify the system works end to end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.llm import generate_response
from src.pipeline import build_pipeline
from src.router import route_query

load_dotenv()


def main() -> None:
    root = Path(__file__).resolve().parent
    pro_life_dir = root / "data" / "pro_life"
    pro_choice_dir = root / "data" / "pro_choice"

    if not pro_life_dir.is_dir() or not pro_choice_dir.is_dir():
        print(
            "Expected folders data/pro_life/ and data/pro_choice/ next to main.py.\n"
            "Create them and add PDF or TXT sources, then run again."
        )
        sys.exit(1)

    print("Building pipelines (this may take a minute on first run)...")
    pro_life_pipeline = build_pipeline(str(pro_life_dir), "pro_life")
    pro_choice_pipeline = build_pipeline(str(pro_choice_dir), "pro_choice")

    print("\n--- Pipeline stats ---")
    for label, pipe in ("pro_life", pro_life_pipeline), ("pro_choice", pro_choice_pipeline):
        print(
            f"  {label}: documents={pipe['doc_count']}, chunks={pipe['chunk_count']}"
        )

    print("\n--- Example: pro-life participant asking the pro-choice corpus ---")
    r1 = route_query(
        "pro_life",
        "Why do people emphasize bodily autonomy in this debate?",
        pro_life_pipeline,
        pro_choice_pipeline,
    )
    resp1 = generate_response(
        "Why do people emphasize bodily autonomy in this debate?",
        r1["context_chunks"],
        r1["source_group"],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"Source group: {r1['source_group']}")
    print(f"Response:\n{resp1}\n")

    print("--- Example: pro-choice participant asking the pro-life corpus ---")
    r2 = route_query(
        "pro_choice",
        "How is human life described in these materials?",
        pro_life_pipeline,
        pro_choice_pipeline,
    )
    resp2 = generate_response(
        "How is human life described in these materials?",
        r2["context_chunks"],
        r2["source_group"],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"Source group: {r2['source_group']}")
    print(f"Response:\n{resp2}")


if __name__ == "__main__":
    main()
