import json

from src import logging_utils


def test_log_interaction_writes_metadata_only(tmp_path, monkeypatch):
    log_file = tmp_path / "interactions.txt"
    monkeypatch.setattr(logging_utils, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logging_utils, "LOG_FILE", str(log_file))

    logging_utils.log_interaction(
        session_id="test-session-id",
        group="pro_life",
        query="secret question text",
        response="secret answer text",
        context_chunks=["a", "b"],
    )

    line = log_file.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["session_id"] == "test-session-id"
    assert data["group"] == "pro_life"
    assert data["query_length"] == len("secret question text")
    assert data["response_length"] == len("secret answer text")
    assert data["num_context_chunks"] == 2
    assert "query" not in data
    assert "secret" not in line
