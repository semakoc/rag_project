"""
Append-only interaction logging with IRB-friendly metadata only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "interactions.txt")


def log_interaction(
    session_id: str,
    group: str,
    query: str,
    response: str,
    context_chunks: list[str],
) -> None:
    """
    Append one JSON line describing an interaction without storing raw text.

    We log lengths and counts so researchers can study usage patterns while
    keeping what participants typed and read out of the log file. Storing
    verbatim queries or responses would raise confidentiality risk under
    Protocol 26.0670, so those strings are intentionally omitted.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "group": group,
        "query_length": len(query or ""),
        "response_length": len(response or ""),
        "num_context_chunks": len(context_chunks or []),
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        raise OSError(f"Could not write interaction log to {LOG_FILE}") from exc
