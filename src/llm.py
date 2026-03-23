"""
Build prompts and call the language model for empathetic mediator replies.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_UNSAFE_PATTERNS = [
    re.compile(r"\b(kill yourself|kys)\b", re.I),
    re.compile(r"\b(n[i1]gg[ae]r|faggot|rape\s+you)\b", re.I),
]


def build_system_prompt(group_name: str) -> str:
    """
    Create the system message that defines the assistant as a caring guide.

    The knowledge base comes from group_name (pro_life or pro_choice). The
    model must stay warm, curious, and non-adversarial so participants can
    explore another viewpoint without feeling attacked.
    """
    label = "pro-life" if group_name == "pro_life" else "pro-choice"
    return f"""You are a thoughtful facilitator helping someone understand how {label} perspectives are often framed and experienced, based only on the reference passages provided.

Your priorities:
- Speak with warmth, patience, and respect. Never belittle the person asking.
- Do not debate, score points, or imply that one side wins. This is understanding, not argument.
- Reflect the values, concerns, and lived experiences suggested by the passages, using plain language.
- Invite curiosity: name tensions honestly, note where people sincerely disagree, and mention shared care for dignity or fairness when the text supports it.
- If the user sounds upset or provocative, stay calm and gently refocus on understanding rather than blame.

Stay grounded in the supplied context. If the context is thin, say so briefly and still offer a good-faith summary of what the materials emphasize."""


def build_user_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Combine the participant's question with up to five context snippets.

    The context is labeled so the model can quote or paraphrase it clearly.
    """
    limited = context_chunks[:5]
    parts = [f"Participant question:\n{query.strip()}", "", "Reference passages:"]
    for i, chunk in enumerate(limited, start=1):
        parts.append(f"[{i}] {chunk.strip()}")
    return "\n\n".join(parts)


def _fallback_response(query: str, context_chunks: list[str]) -> str:
    """Show retrieved passages clearly when no API key is available."""
    lines = [
        "Here is a concise summary based on the retrieved materials "
        "(LLM generation is unavailable without an API key):",
        "",
    ]
    for i, ch in enumerate(context_chunks[:5], start=1):
        lines.append(f"{i}. {ch.strip()}")
    lines.append("")
    lines.append(
        "For a full conversational response, set OPENAI_API_KEY in your environment."
    )
    return "\n".join(lines)


def generate_response(
    query: str,
    context_chunks: list[str],
    group_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Ask OpenAI for a reply, or fall back to a readable context summary.

    If no API key is passed, the OPENAI_API_KEY environment variable is used.
    When no key exists, the function still returns helpful text built from the
    retrieved chunks so demos work offline.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return _fallback_response(query, context_chunks)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        system = build_system_prompt(group_name)
        user = build_user_prompt(query, context_chunks)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        choice = completion.choices[0].message.content
        if not choice:
            return _fallback_response(query, context_chunks)
        return choice.strip()
    except Exception as exc:
        return (
            f"Could not reach the language model ({exc}). "
            "Showing retrieved context instead:\n\n" + _fallback_response(query, context_chunks)
        )


def safety_check(text: str) -> bool:
    """
    Quick screen for obviously unsafe user input before calling the model.

    This is a light pre-filter; the main safeguards live in the system
    prompt. Returns False when crude slurs or severe harassment appear.
    """
    if not text or not text.strip():
        return True
    lowered = text.lower()
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(lowered):
            return False
    if "http://" in lowered or "https://" in lowered:
        return True
    return True
