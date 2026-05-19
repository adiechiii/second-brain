"""Memory search service tests."""

from datetime import datetime, timedelta, timezone

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_search import search_memories


def build_memory(
    clean_text: str | None,
    importance_score: float | None = None,
    summary: str | None = None,
    tags=None,
    topic: str | None = None,
    updated_at=None,
    created_at=None,
) -> Memory:
    return Memory(
        raw_text=clean_text or "",
        clean_text=clean_text,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
        importance_score=importance_score,
        summary=summary,
        tags=tags,
        topic=topic,
        updated_at=updated_at,
        created_at=created_at,
    )


def test_search_returns_relevant_memory():
    database_memory = build_memory(
        "PostgreSQL database indexing and schema notes",
        importance_score=0.2,
    )
    garden_memory = build_memory(
        "Garden watering schedule and soil notes",
        importance_score=1.0,
    )

    results = search_memories("database schema", [garden_memory, database_memory])

    assert results == [database_memory]


def test_search_sorts_by_similarity_then_importance():
    high_importance_partial = build_memory(
        "database maintenance",
        importance_score=0.9,
    )
    low_importance_full = build_memory(
        "database schema planning",
        importance_score=0.1,
    )
    medium_importance_full = build_memory(
        "schema database review",
        importance_score=0.5,
    )

    results = search_memories(
        "database schema",
        [high_importance_partial, low_importance_full, medium_importance_full],
    )

    assert results == [
        medium_importance_full,
        low_importance_full,
        high_importance_partial,
    ]


def test_empty_search_returns_empty_list():
    memory = build_memory("database schema notes", importance_score=1.0)

    assert search_memories("   ", [memory]) == []


def test_summary_match_returns_relevant_memory_when_clean_text_does_not_match():
    summary_match = build_memory(
        "Quarterly planning notes",
        summary="PostgreSQL database migration checklist",
    )
    unrelated = build_memory("Garden watering schedule")

    results = search_memories("database", [unrelated, summary_match])

    assert results == [summary_match]


def test_tag_match_returns_relevant_memory_when_clean_text_does_not_match():
    tag_match = build_memory("Quarterly planning notes", tags=["database"])
    unrelated = build_memory("Garden watering schedule")

    results = search_memories("database", [unrelated, tag_match])

    assert results == [tag_match]


def test_topic_match_returns_relevant_memory_when_clean_text_does_not_match():
    topic_match = build_memory("Quarterly planning notes", topic="database")
    unrelated = build_memory("Garden watering schedule")

    results = search_memories("database", [unrelated, topic_match])

    assert results == [topic_match]


def test_zero_overlap_memory_is_excluded():
    relevant = build_memory("database schema notes")
    unrelated = build_memory("Garden watering schedule")

    results = search_memories("database", [unrelated, relevant])

    assert results == [relevant]


def test_importance_improves_fallback_ranking_when_stronger_memory_is_later():
    low_importance = build_memory(
        "database maintenance notes",
        importance_score=0.1,
    )
    high_importance = build_memory(
        "database maintenance notes",
        importance_score=0.9,
    )

    results = search_memories("database", [low_importance, high_importance])

    assert results == [high_importance, low_importance]


def test_newer_updated_at_improves_fallback_ranking_when_other_signals_match():
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    old = build_memory(
        "database maintenance notes",
        updated_at=now - timedelta(days=120),
    )
    new = build_memory(
        "database maintenance notes",
        updated_at=now - timedelta(days=1),
    )

    results = search_memories("database", [old, new])

    assert results == [new, old]


def test_missing_optional_fields_do_not_crash():
    memory = build_memory(
        None,
        summary=None,
        tags="database",
        topic=None,
        importance_score=None,
        updated_at=None,
        created_at=None,
    )

    assert search_memories("database", [memory]) == []
