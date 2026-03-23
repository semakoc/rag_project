"""
Route participant queries to the opposing group's knowledge base.
"""

from __future__ import annotations

from src.pipeline import query_pipeline

VALID_GROUPS = frozenset({"pro_life", "pro_choice"})


def validate_group(user_group: str) -> bool:
    """
    Check whether a group label is one the study recognizes.

    Returns True for pro_life and pro_choice, and False for anything else.
    """
    return user_group in VALID_GROUPS


def route_query(
    user_group: str,
    query: str,
    pro_life_pipeline: dict,
    pro_choice_pipeline: dict,
) -> dict:
    """
    Send the question to the other side's materials so the participant hears
    that perspective instead of their own group's library.

    Pro-life participants see retrieved text from the pro-choice corpus and
    vice versa. The returned dict includes the chunks and which group they
    came from.
    """
    if user_group not in VALID_GROUPS:
        raise ValueError(f"Invalid user_group: {user_group!r}. Use pro_life or pro_choice.")
    if user_group == "pro_life":
        target = pro_choice_pipeline
        source_group = "pro_choice"
    else:
        target = pro_life_pipeline
        source_group = "pro_life"
    context_chunks = query_pipeline(target, query)
    return {"context_chunks": context_chunks, "source_group": source_group}
