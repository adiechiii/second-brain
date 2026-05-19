from datetime import date, datetime, timezone

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_compression import (
    MemoryCompressionError,
    build_daily_summary,
    build_weekly_summary,
)


def _memory(
    *,
    summary: str | None = "Source summary",
    clean_text: str | None = "Clean source text",
    raw_text: str = "Raw source text",
    tags: object = None,
    topic: object = None,
    importance_score: object = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> Memory:
    return Memory(
        raw_text=raw_text,
        clean_text=clean_text,
        summary=summary,
        tags=tags,
        topic=topic,
        importance_score=importance_score,
        created_at=created_at,
        updated_at=updated_at,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def test_builds_daily_summary_memory_without_saving() -> None:
    memory = _memory(
        summary="Reviewed database schema.",
        tags=["database"],
        topic="architecture",
        importance_score=0.4,
    )

    result = build_daily_summary([memory], date(2026, 5, 19))

    assert result.raw_text == result.clean_text == result.summary
    assert result.summary == (
        "Across 1 source memories for 2026-05-19, "
        "strongest evidence points to: Reviewed database schema."
    )
    assert result.processing_state == ProcessingState.COMPLETED
    assert result.record_state == RecordState.ACTIVE
    assert result.embedding is None
    assert result.classification_confidence is None
    assert result.tone_confidence is None
    assert result.topic_confidence is None


def test_builds_weekly_summary_memory_with_monday_week_start() -> None:
    result = build_weekly_summary(
        [_memory(summary="Captured retrieval notes.")],
        date(2026, 5, 18),
    )

    assert result.summary.startswith(
        "Across 1 source memories for 2026-05-18..2026-05-24"
    )
    assert result.topic == "weekly-summary"
    assert "weekly-summary" in result.tags
    assert "period:2026-05-18..2026-05-24" in result.tags


def test_weekly_summary_rejects_non_monday_week_start() -> None:
    with pytest.raises(MemoryCompressionError):
        build_weekly_summary([_memory()], date(2026, 5, 19))


def test_original_memories_are_not_mutated() -> None:
    memory = _memory(
        summary="Original summary",
        clean_text="Original clean text",
        raw_text="Original raw text",
        tags=["database", "retrieval"],
        topic="architecture",
        importance_score=0.7,
    )
    original = (
        memory.summary,
        memory.clean_text,
        memory.raw_text,
        list(memory.tags),
        memory.topic,
        memory.importance_score,
        memory.processing_state,
        memory.record_state,
        memory.embedding,
    )

    build_daily_summary([memory], date(2026, 5, 19))

    assert (
        memory.summary,
        memory.clean_text,
        memory.raw_text,
        memory.tags,
        memory.topic,
        memory.importance_score,
        memory.processing_state,
        memory.record_state,
        memory.embedding,
    ) == original


def test_compression_tagged_source_memories_are_excluded() -> None:
    compression = _memory(summary="Prior compression", tags=["compression"])
    source = _memory(summary="Fresh source note", tags=["database"])

    result = build_daily_summary([compression, source], date(2026, 5, 19))

    assert "Across 1 source memories" in result.summary
    assert "Fresh source note" in result.summary
    assert "Prior compression" not in result.summary


def test_duplicate_evidence_summaries_are_included_once() -> None:
    result = build_daily_summary(
        [
            _memory(summary="Reviewed database schema."),
            _memory(summary=" reviewed   database schema. "),
            _memory(summary="Captured indexing tradeoffs."),
        ],
        date(2026, 5, 19),
    )

    assert result.summary.count("Reviewed database schema.") == 1
    assert "Captured indexing tradeoffs." in result.summary


def test_important_memory_appears_before_less_important_memory() -> None:
    result = build_daily_summary(
        [
            _memory(summary="Low importance note", importance_score=0.1),
            _memory(summary="High importance note", importance_score=0.9),
        ],
        date(2026, 5, 19),
    )

    assert result.summary.index("High importance note") < result.summary.index(
        "Low importance note"
    )


def test_recency_breaks_ties_when_importance_is_equal() -> None:
    result = build_daily_summary(
        [
            _memory(
                summary="Older note",
                importance_score=0.5,
                updated_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            ),
            _memory(
                summary="Newer note",
                importance_score=0.5,
                updated_at=datetime(2026, 5, 18),
            ),
        ],
        date(2026, 5, 19),
    )

    assert result.summary.index("Newer note") < result.summary.index("Older note")


def test_output_has_marker_tags_and_topic() -> None:
    result = build_daily_summary(
        [_memory(summary="Tagged note", tags=["database"], topic="architecture")],
        date(2026, 5, 19),
    )

    assert result.tags[:3] == [
        "compression",
        "daily-summary",
        "period:2026-05-19",
    ]
    assert "database" in result.tags
    assert "architecture" in result.tags
    assert result.topic == "daily-summary"


def test_empty_input_raises_memory_compression_error() -> None:
    with pytest.raises(MemoryCompressionError):
        build_daily_summary([], date(2026, 5, 19))


def test_all_empty_evidence_raises_memory_compression_error() -> None:
    memory = _memory(
        summary=None,
        clean_text=None,
        raw_text="Raw text should not be used for compression evidence.",
    )

    with pytest.raises(MemoryCompressionError):
        build_daily_summary([memory], date(2026, 5, 19))


def test_invalid_tags_and_topics_do_not_crash() -> None:
    result = build_daily_summary(
        [
            _memory(
                summary="Valid evidence",
                tags=[None, "", "database", 7],
                topic=123,
            ),
            _memory(
                summary="More evidence",
                tags="not-a-list",
                topic="Architecture",
            ),
        ],
        date(2026, 5, 19),
    )

    assert "database" in result.tags
    assert "architecture" in result.tags
    assert result.tags.count("compression") == 1


def test_importance_score_is_clamped_to_valid_range() -> None:
    high_result = build_daily_summary(
        [_memory(summary="Very important", importance_score=3.0)],
        date(2026, 5, 19),
    )
    low_result = build_daily_summary(
        [_memory(summary="Negative importance", importance_score=-1.0)],
        date(2026, 5, 19),
    )

    assert high_result.importance_score == 1.0
    assert low_result.importance_score == 0.0


def test_clean_text_is_used_when_summary_is_missing() -> None:
    result = build_daily_summary(
        [_memory(summary=None, clean_text="Clean fallback evidence")],
        date(2026, 5, 19),
    )

    assert "Clean fallback evidence" in result.summary
