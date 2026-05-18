"""Reflection engine tests."""

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.reflection_engine import generate_reflection


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


def test_multiple_memories_extract_themes():
    memories = [
        build_memory(
            "Reviewed database schema decisions.",
            ["database", "schema", "planning"],
            topic="architecture",
            importance_score=0.8,
        ),
        build_memory(
            "Captured database indexing tradeoffs.",
            ["database", "indexing"],
            topic="architecture",
            importance_score=0.6,
        ),
    ]

    reflection = generate_reflection(memories)

    assert reflection["summary"] == (
        "Reviewed database schema decisions. | Captured database indexing tradeoffs."
    )
    assert reflection["themes"][0] == "database"
    assert "'database' appears in 2 retrieved memories." in reflection["insights"]
    assert "The topic 'architecture' appears in 2 retrieved memories." in reflection["insights"]
    assert reflection["questions"] == [
        "What would make 'database' easier to act on next?",
        "What would make 'schema' easier to act on next?",
        "What would make 'planning' easier to act on next?",
    ]


def test_single_memory_uses_cautious_output():
    memory = build_memory(
        "Noted a database migration concern.",
        ["database"],
        topic="architecture",
    )

    reflection = generate_reflection([memory])

    assert reflection["summary"].startswith("Only one retrieved memory is available:")
    assert reflection["themes"] == ["database"]
    assert reflection["insights"] == [
        "There is not enough retrieved evidence to identify a repeated pattern."
    ]
    assert reflection["questions"] == [
        "What additional memories would help confirm whether this pattern matters?"
    ]


def test_reflection_does_not_hallucinate_content():
    memories = [
        build_memory("Budget note about vendor invoices.", ["budget", "vendor"]),
        build_memory("Follow up on vendor invoice timing.", ["vendor", "invoice"]),
    ]

    reflection = generate_reflection(memories)
    combined = " ".join(
        [reflection["summary"]]
        + reflection["themes"]
        + reflection["insights"]
        + reflection["questions"]
    ).lower()

    assert "health" not in combined
    assert "relationship" not in combined
    assert "travel" not in combined
    assert "vendor" in combined


def test_no_memories_returns_grounded_empty_reflection():
    reflection = generate_reflection([])

    assert reflection == {
        "summary": "No retrieved memories were provided, so there is not enough evidence for a reflection.",
        "themes": [],
        "insights": [],
        "questions": ["Which memories should be retrieved before reflecting?"],
    }
