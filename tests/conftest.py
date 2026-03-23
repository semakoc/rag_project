import os
from pathlib import Path

import pytest

# Use a project-local Hugging Face cache so tests do not require ~/.cache writes.
_hf = Path(__file__).resolve().parent.parent / ".hf_cache"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))


@pytest.fixture
def sample_text():
    return """This is a sample document about values and community.
    People hold many different perspectives on important issues.
    Understanding each other requires patience and empathy.
    Page 1
    We must listen carefully to those we disagree with."""


@pytest.fixture
def short_text():
    return "Short text for testing."
