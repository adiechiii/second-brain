"""Reflection engine tests."""

from datetime import datetime, timezone

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.reflection_engine import generate_reflection


def build_memory(
    summary: str,
    tags,
    topic=None,
    importance_score: float | None = None,
    updated_at=None,
    created_at=None,
) -> Memory:
    return Memory(
        raw_text=summary,
        clean_text=summary,
        summary=summary,
        tags=tags,
        topic=topic,
        importance_score=importance_score,
        updated_at=updated_at,
        created_at=created_at,
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
        "Across 2 retrieved memories, the strongest evidence points to: "
        "Reviewed database schema decisions. | Captured database indexing tradeoffs."
    )
    assert reflection["themes"] == [
        "architecture",
        "database",
        "indexing",
        "planning",
        "schema",
    ]
    assert "'architecture' is supported by 2 retrieved memories." in reflection["insights"]
    assert "'database' is supported by 2 retrieved memories." in reflection["insights"]
    assert reflection["questions"] == [
        "What decision, action, or open question keeps recurring around 'architecture'?",
        "What decision, action, or open question keeps recurring around 'database'?",
        "What extra memory would confirm whether 'indexing' is important or just a one-off lead?",
    ]


def test_single_memory_uses_cautious_output():
    memory = build_memory(
        "Noted a database migration concern.",
        ["database"],
        topic="architecture",
    )

    reflection = generate_reflection([memory])

    assert reflection["summary"].startswith("Only one retrieved memory is available:")
    assert reflection["themes"] == ["architecture", "database"]
    assert reflection["insights"] == [
        "There is not enough retrieved evidence to identify a repeated pattern."
    ]
    assert reflection["questions"] == [
        "What follow-up memory would show whether this is isolated or recurring?"
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


def test_multi_memory_summary_has_evidence_count_prefix():
    memories = [
        build_memory("Reviewed database schema decisions.", ["database"]),
        build_memory("Captured database indexing tradeoffs.", ["database"]),
        build_memory("Documented service boundary notes.", ["services"]),
    ]

    reflection = generate_reflection(memories)

    assert reflection["summary"].startswith(
        "Across 3 retrieved memories, the strongest evidence points to: "
    )


def test_duplicate_summaries_are_included_once():
    memories = [
        build_memory("Reviewed database schema decisions.", ["database"]),
        build_memory(" reviewed   database schema decisions. ", ["database"]),
        build_memory("Captured database indexing tradeoffs.", ["database"]),
    ]

    reflection = generate_reflection(memories)

    assert reflection["summary"].count("Reviewed database schema decisions.") == 1
    assert "Captured database indexing tradeoffs." in reflection["summary"]


def test_summary_selection_prefers_theme_coverage_over_unrelated_importance():
    memories = [
        build_memory("Unrelated administrative note.", [], importance_score=1.0),
        build_memory("Reviewed database schema decisions.", ["database"], importance_score=0.1),
        build_memory("Captured database indexing tradeoffs.", ["database"], importance_score=0.1),
        build_memory("Documented database rollback notes.", ["database"], importance_score=0.1),
    ]

    reflection = generate_reflection(memories)

    assert "Unrelated administrative note." not in reflection["summary"]
    assert "Reviewed database schema decisions." in reflection["summary"]
    assert "Captured database indexing tradeoffs." in reflection["summary"]
    assert "Documented database rollback notes." in reflection["summary"]


def test_summary_selection_uses_updated_at_recency_as_tie_breaker():
    older = datetime(2026, 1, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 5, 1, tzinfo=timezone.utc)
    memories = [
        build_memory("Older database note.", ["database"], importance_score=0.5, updated_at=older),
        build_memory("Newer database note.", ["database"], importance_score=0.5, updated_at=newer),
    ]

    reflection = generate_reflection(memories)

    assert reflection["summary"].index("Newer database note.") < reflection["summary"].index(
        "Older database note."
    )


def test_summary_selection_falls_back_to_created_at_for_recency():
    older = datetime(2026, 1, 1)
    newer = datetime(2026, 5, 1)
    memories = [
        build_memory("Older database note.", ["database"], importance_score=0.5, created_at=older),
        build_memory("Newer database note.", ["database"], importance_score=0.5, created_at=newer),
    ]

    reflection = generate_reflection(memories)

    assert reflection["summary"].index("Newer database note.") < reflection["summary"].index(
        "Older database note."
    )


def test_topic_only_memories_produce_themes():
    memories = [
        build_memory("Reviewed schema boundaries.", [], topic="Architecture"),
        build_memory("Captured service dependencies.", [], topic="Architecture"),
    ]

    reflection = generate_reflection(memories)

    assert reflection["themes"] == ["architecture"]
    assert reflection["insights"] == [
        "'architecture' is supported by 2 retrieved memories."
    ]


def test_tags_and_topics_both_contribute_to_theme_counts():
    memories = [
        build_memory("Reviewed database schema decisions.", ["Database"], topic="Architecture"),
        build_memory("Captured database indexing tradeoffs.", ["database"], topic="Operations"),
        build_memory("Documented service boundary notes.", ["services"], topic="architecture"),
    ]

    reflection = generate_reflection(memories)

    assert reflection["themes"][:2] == ["architecture", "database"]
    assert "'architecture' is supported by 2 retrieved memories." in reflection["insights"]
    assert "'database' is supported by 2 retrieved memories." in reflection["insights"]


def test_insights_include_evidence_counts_for_single_theme_leads():
    memories = [
        build_memory("Reviewed database schema decisions.", ["database"]),
        build_memory("Captured indexing tradeoffs.", ["indexing"]),
    ]

    reflection = generate_reflection(memories)

    assert (
        "'database' appears in one retrieved memory, so treat it as a lead rather than a pattern."
        in reflection["insights"]
    )


def test_empty_or_non_string_tags_and_topics_are_ignored():
    memories = [
        build_memory("Reviewed database schema decisions.", ["database", "", None, 7], topic=12),
        build_memory("Captured database indexing tradeoffs.", ["database"], topic=""),
    ]

    reflection = generate_reflection(memories)

    assert reflection["themes"] == ["database"]
    assert reflection["insights"] == ["'database' is supported by 2 retrieved memories."]


def test_reflection_output_schema_is_unchanged():
    reflection = generate_reflection([
        build_memory("Reviewed database schema decisions.", ["database"]),
    ])

    assert set(reflection) == {"summary", "themes", "insights", "questions"}
    assert isinstance(reflection["summary"], str)
    assert isinstance(reflection["themes"], list)
    assert isinstance(reflection["insights"], list)
    assert isinstance(reflection["questions"], list)


def test_no_memories_returns_grounded_empty_reflection():
    reflection = generate_reflection([])

    assert reflection == {
        "summary": "No retrieved memories were provided, so there is not enough evidence for a reflection.",
        "themes": [],
        "insights": [],
        "questions": ["Which memories should be retrieved before reflecting?"],
    }
