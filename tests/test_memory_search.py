"""Memory search service tests."""

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_search import search_memories


def build_memory(clean_text: str, importance_score: float | None = None) -> Memory:
    return Memory(
        raw_text=clean_text,
        clean_text=clean_text,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
        importance_score=importance_score,
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
