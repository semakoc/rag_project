from src.llm import (
    build_system_prompt,
    build_user_prompt,
    generate_response,
    safety_check,
)


def test_build_system_prompt_contains_guidance():
    text = build_system_prompt("pro_choice")
    assert "pro-choice" in text.lower() or "pro_choice" in text.lower()
    assert "respect" in text.lower() or "warm" in text.lower()


def test_build_user_prompt_limits_five_chunks():
    chunks = [f"chunk {i} text here." for i in range(10)]
    prompt = build_user_prompt("why?", chunks)
    assert prompt.count("[1]") == 1
    assert "[6]" not in prompt


def test_generate_response_no_key_returns_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = generate_response(
        "hello",
        ["context one", "context two"],
        "pro_life",
        api_key=None,
    )
    assert "context" in out.lower() or "retrieved" in out.lower() or "1." in out


def test_safety_check_allows_normal():
    assert safety_check("I want to understand the other perspective.") is True


def test_safety_check_blocks_harassment_phrase():
    assert safety_check("you should kys immediately") is False
