from datetime import datetime, timezone
from uuid import UUID

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.idea_evolution import build_idea_threads


def _memory(
    id_value: str,
    *,
    summary: str | None = "Source summary",
    clean_text: str | None = "Clean text",
    raw_text: str = "Raw text",
    tags: object = None,
    topic: object = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> Memory:
    return Memory(
        id=UUID(id_value),
        raw_text=raw_text,
        clean_text=clean_text,
        summary=summary,
        tags=tags,
        topic=topic,
        created_at=created_at,
        updated_at=updated_at,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def test_builds_thread_from_shared_tag_across_memories() -> None:
    memories = [
        _memory("00000000-0000-0000-0000-000000000001", tags=["Database"]),
        _memory("00000000-0000-0000-0000-000000000002", tags=[" database "]),
    ]

    threads = build_idea_threads(memories)

    assert len(threads) == 1
    assert threads[0].theme == "database"
    assert threads[0].evidence_count == 2


def test_builds_thread_from_shared_topic_across_memories() -> None:
    memories = [
        _memory("00000000-0000-0000-0000-000000000001", topic="Retrieval"),
        _memory("00000000-0000-0000-0000-000000000002", topic=" retrieval "),
    ]

    threads = build_idea_threads(memories)

    assert len(threads) == 1
    assert threads[0].theme == "retrieval"


def test_one_off_themes_are_excluded_by_default() -> None:
    memories = [
        _memory("00000000-0000-0000-0000-000000000001", tags=["database"]),
        _memory("00000000-0000-0000-0000-000000000002", tags=["reflection"]),
    ]

    assert build_idea_threads(memories) == []


def test_invalid_tags_and_topics_are_ignored() -> None:
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            tags=[None, "", 7, "Database"],
            topic=object(),
        ),
        _memory(
            "00000000-0000-0000-0000-000000000002",
            tags="database",
            topic="Database",
        ),
    ]

    threads = build_idea_threads(memories)

    assert [thread.theme for thread in threads] == ["database"]


def test_compression_memories_are_excluded() -> None:
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            tags=["compression", "database"],
        ),
        _memory("00000000-0000-0000-0000-000000000002", tags=["database"]),
    ]

    assert build_idea_threads(memories) == []


def test_compression_marker_tags_and_period_tags_are_ignored() -> None:
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            tags=["daily-summary", "period:2026-05-19", "database"],
        ),
        _memory(
            "00000000-0000-0000-0000-000000000002",
            tags=["weekly-summary", "period:2026-05-18..2026-05-24", "database"],
        ),
    ]

    threads = build_idea_threads(memories, min_evidence_count=1)

    assert [thread.theme for thread in threads] == ["database"]


def test_memory_ids_are_ordered_chronologically() -> None:
    older_id = "00000000-0000-0000-0000-000000000001"
    newer_id = "00000000-0000-0000-0000-000000000002"
    memories = [
        _memory(
            newer_id,
            tags=["database"],
            created_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        ),
        _memory(
            older_id,
            tags=["database"],
            created_at=datetime(2026, 5, 19, tzinfo=timezone.utc),
        ),
    ]

    thread = build_idea_threads(memories)[0]

    assert thread.memory_ids == [UUID(older_id), UUID(newer_id)]


def test_first_seen_and_last_seen_are_computed() -> None:
    first = datetime(2026, 5, 19, tzinfo=timezone.utc)
    last = datetime(2026, 5, 20)
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            tags=["database"],
            created_at=first,
        ),
        _memory(
            "00000000-0000-0000-0000-000000000002",
            tags=["database"],
            created_at=last,
        ),
    ]

    thread = build_idea_threads(memories)[0]

    assert thread.first_seen == first
    assert thread.last_seen == last


def test_summaries_use_summary_first_then_clean_text() -> None:
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            summary="Summary evidence",
            clean_text="Clean evidence",
            tags=["database"],
        ),
        _memory(
            "00000000-0000-0000-0000-000000000002",
            summary=None,
            clean_text="Clean fallback",
            tags=["database"],
        ),
    ]

    thread = build_idea_threads(memories)[0]

    assert thread.summaries == ["Summary evidence", "Clean fallback"]


def test_raw_text_is_not_used_as_evidence() -> None:
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            summary=None,
            clean_text=None,
            raw_text="Raw evidence should not appear",
            tags=["database"],
        ),
        _memory(
            "00000000-0000-0000-0000-000000000002",
            summary=None,
            clean_text=None,
            raw_text="More raw evidence should not appear",
            tags=["database"],
        ),
    ]

    thread = build_idea_threads(memories)[0]

    assert thread.summaries == []


def test_missing_dates_do_not_crash() -> None:
    missing_date_id = "00000000-0000-0000-0000-000000000001"
    dated_id = "00000000-0000-0000-0000-000000000002"
    memories = [
        _memory(missing_date_id, tags=["database"]),
        _memory(
            dated_id,
            tags=["database"],
            created_at=datetime(2026, 5, 19, tzinfo=timezone.utc),
        ),
    ]

    thread = build_idea_threads(memories)[0]

    assert thread.memory_ids == [UUID(dated_id), UUID(missing_date_id)]
    assert thread.first_seen == datetime(2026, 5, 19, tzinfo=timezone.utc)
    assert thread.last_seen == datetime(2026, 5, 19, tzinfo=timezone.utc)


def test_thread_sorting_is_deterministic_for_equal_counts_and_dates() -> None:
    created_at = datetime(2026, 5, 19, tzinfo=timezone.utc)
    memories = [
        _memory(
            "00000000-0000-0000-0000-000000000001",
            tags=["beta", "alpha"],
            created_at=created_at,
        ),
        _memory(
            "00000000-0000-0000-0000-000000000002",
            tags=["beta", "alpha"],
            created_at=created_at,
        ),
    ]

    threads = build_idea_threads(memories)

    assert [thread.theme for thread in threads] == ["alpha", "beta"]


def test_source_memories_are_not_mutated() -> None:
    memory = _memory(
        "00000000-0000-0000-0000-000000000001",
        summary="Original summary",
        clean_text="Original clean",
        raw_text="Original raw",
        tags=["database"],
        topic="architecture",
    )
    original = (
        memory.summary,
        memory.clean_text,
        memory.raw_text,
        list(memory.tags),
        memory.topic,
        memory.created_at,
        memory.updated_at,
    )

    build_idea_threads([memory], min_evidence_count=1)

    assert (
        memory.summary,
        memory.clean_text,
        memory.raw_text,
        memory.tags,
        memory.topic,
        memory.created_at,
        memory.updated_at,
    ) == original


def test_custom_min_evidence_count_one_includes_one_off_themes() -> None:
    memories = [
        _memory("00000000-0000-0000-0000-000000000001", tags=["database"]),
    ]

    threads = build_idea_threads(memories, min_evidence_count=1)

    assert [thread.theme for thread in threads] == ["database"]


def test_max_summaries_limits_summaries() -> None:
    memories = [
        _memory(
            f"00000000-0000-0000-0000-00000000000{index}",
            summary=f"Summary {index}",
            tags=["database"],
        )
        for index in range(1, 4)
    ]

    thread = build_idea_threads(memories, max_summaries=2)[0]

    assert thread.summaries == ["Summary 1", "Summary 2"]
