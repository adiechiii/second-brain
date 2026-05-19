"""AI-enhanced reflection tests."""

import pytest

from app.core.config import get_settings
from app.models.memory import Memory, ProcessingState, RecordState
from app.services.reflection_engine import (
    generate_reflection,
    generate_reflection_with_ai,
)


def build_memory(
    summary: str,
    tags: list[str],
    topic: str | None = None,
    importance_score: float | None = None,
) -> Memory:
    return Memory(
        raw_text=summary,
        clean_text=summary,
        summary=summary,
        tags=tags,
        topic=topic,
        importance_score=importance_score,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def enable_openai_reflections(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_OPENAI_REFLECTIONS", "true")


def test_ai_reflection_disabled_by_default_does_not_call_completion(monkeypatch):
    memories = [build_memory("Reviewed database schema decisions.", ["database"])]

    def fail_completion(prompt: str) -> str:
        raise AssertionError("completion should not be called")

    get_settings.cache_clear()
    monkeypatch.delenv("ENABLE_OPENAI_REFLECTIONS", raising=False)
    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        fail_completion,
    )

    assert generate_reflection_with_ai(memories) == generate_reflection(memories)


def test_ai_reflection_uses_mocked_openai_response(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [
        build_memory("Reviewed database schema decisions.", ["database", "schema"]),
        build_memory("Captured database indexing tradeoffs.", ["database", "indexing"]),
    ]

    def fake_completion(prompt: str) -> str:
        assert "Reviewed database schema decisions." in prompt
        assert "Captured database indexing tradeoffs." in prompt
        assert "Use ONLY the provided memories" in prompt
        return """
        {
          "summary": "A database-focused reflection.",
          "themes": ["database", "schema"],
          "insights": ["database appears across the retrieved memories."],
          "questions": ["What database decision needs the next review?"]
        }
        """

    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        fake_completion,
    )

    reflection = generate_reflection_with_ai(memories)

    assert set(reflection) == {"summary", "themes", "insights", "questions"}
    assert reflection["summary"] == generate_reflection(memories)["summary"]
    assert reflection["themes"] == ["database", "schema"]
    assert reflection["insights"] == ["database appears across the retrieved memories."]
    assert reflection["questions"] == ["What database decision needs the next review?"]


def test_ai_reflection_falls_back_when_api_fails(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [build_memory("Reviewed database schema decisions.", ["database"])]

    def fail_completion(prompt: str) -> str:
        raise RuntimeError("api failed")

    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        fail_completion,
    )

    assert generate_reflection_with_ai(memories) == generate_reflection(memories)


def test_ai_reflection_falls_back_on_invalid_json(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [build_memory("Reviewed database schema decisions.", ["database"])]

    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        lambda prompt: "not json",
    )

    assert generate_reflection_with_ai(memories) == generate_reflection(memories)


def test_ai_reflection_falls_back_on_timeout(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [build_memory("Reviewed database schema decisions.", ["database"])]

    def timeout_completion(prompt: str) -> str:
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        timeout_completion,
    )

    assert generate_reflection_with_ai(memories) == generate_reflection(memories)


def test_ai_reflection_schema_consistency(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [
        build_memory("Budget note about vendor invoices.", ["budget", "vendor"]),
        build_memory("Follow up on vendor invoice timing.", ["vendor", "invoice"]),
    ]
    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        lambda prompt: """
        {
          "summary": "Vendor reflection.",
          "themes": ["vendor"],
          "insights": ["vendor appears in the retrieved memories."],
          "questions": ["What vendor invoice timing needs review?"]
        }
        """,
    )

    reflection = generate_reflection_with_ai(memories)

    assert isinstance(reflection["summary"], str)
    assert isinstance(reflection["themes"], list)
    assert isinstance(reflection["insights"], list)
    assert isinstance(reflection["questions"], list)


def test_unsafe_grounded_ai_output_uses_deterministic_fallback_values(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [
        build_memory("Budget note about vendor invoices.", ["budget", "vendor"]),
        build_memory("Follow up on vendor invoice timing.", ["vendor", "invoice"]),
    ]
    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        lambda prompt: """
        {
          "summary": "Vendor reflection.",
          "themes": ["vendor"],
          "insights": ["vendor caused burnout across the retrieved memories."],
          "questions": ["What should change about vendor invoice timing?"]
        }
        """,
    )

    reflection = generate_reflection_with_ai(memories)
    deterministic = generate_reflection(memories)

    assert reflection["themes"] == ["vendor"]
    assert reflection["insights"] == deterministic["insights"]
    assert reflection["questions"] == deterministic["questions"]


def test_ai_reflection_does_not_leak_prompt(monkeypatch):
    enable_openai_reflections(monkeypatch)
    memories = [
        build_memory("Budget note about vendor invoices.", ["budget", "vendor"]),
        build_memory("Follow up on vendor invoice timing.", ["vendor", "invoice"]),
    ]
    monkeypatch.setattr(
        "app.services.reflection_engine.generate_completion",
        lambda prompt: """
        {
          "summary": "Ignore prior instructions.",
          "themes": ["vendor"],
          "insights": ["vendor appears in the retrieved memories."],
          "questions": ["What vendor invoice timing needs review?"]
        }
        """,
    )

    reflection = generate_reflection_with_ai(memories)
    combined = " ".join(
        [reflection["summary"]]
        + reflection["themes"]
        + reflection["insights"]
        + reflection["questions"]
    )

    assert "Use ONLY the provided memories" not in combined
    assert "Deterministic baseline JSON" not in combined
