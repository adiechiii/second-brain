from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_compression import MemoryCompressionError
from app.services.memory_compression_service import MemoryCompressionService


def _memory(
    summary: str = "Source summary",
    *,
    tags: list[str] | None = None,
    topic: str | None = None,
    created_at: datetime | None = None,
) -> Memory:
    return Memory(
        id=uuid4(),
        raw_text="Source raw text",
        clean_text="Source clean text",
        summary=summary,
        tags=tags,
        topic=topic,
        created_at=created_at,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


class FakeCompressionRepository:
    def __init__(
        self,
        *,
        sources: list[Memory] | None = None,
        existing: Memory | None = None,
    ):
        self.sources = sources or []
        self.existing = existing
        self.find_calls = []
        self.list_range = None
        self.saved_memory = None

    def find_compression_summary(
        self,
        summary_type: str,
        period_tag: str,
    ) -> Memory | None:
        self.find_calls.append((summary_type, period_tag))
        return self.existing

    def list_active_by_created_at_range(
        self,
        start_at: datetime,
        end_at: datetime,
    ) -> list[Memory]:
        self.list_range = (start_at, end_at)
        return self.sources

    def save(self, memory: Memory) -> Memory:
        memory.id = uuid4()
        self.saved_memory = memory
        return memory


def test_daily_compression_lists_utc_day_range_and_saves_summary() -> None:
    repository = FakeCompressionRepository(
        sources=[_memory("Daily source", tags=["database"])],
    )
    service = MemoryCompressionService(repository)

    result = service.create_daily_summary(date(2026, 5, 19))

    assert repository.find_calls == [
        ("daily-summary", "period:2026-05-19"),
    ]
    assert repository.list_range == (
        datetime(2026, 5, 19, tzinfo=timezone.utc),
        datetime(2026, 5, 20, tzinfo=timezone.utc),
    )
    assert repository.saved_memory is result
    assert result.topic == "daily-summary"
    assert "Daily source" in result.summary


def test_weekly_compression_lists_monday_to_next_monday_range_and_saves() -> None:
    repository = FakeCompressionRepository(
        sources=[_memory("Weekly source", tags=["retrieval"])],
    )
    service = MemoryCompressionService(repository)

    result = service.create_weekly_summary(date(2026, 5, 18))

    assert repository.find_calls == [
        ("weekly-summary", "period:2026-05-18..2026-05-24"),
    ]
    assert repository.list_range == (
        datetime(2026, 5, 18, tzinfo=timezone.utc),
        datetime(2026, 5, 25, tzinfo=timezone.utc),
    )
    assert repository.saved_memory is result
    assert result.topic == "weekly-summary"
    assert "Weekly source" in result.summary


def test_duplicate_daily_returns_existing_without_listing_or_saving() -> None:
    existing = _memory("Existing daily compression", tags=["compression"])
    repository = FakeCompressionRepository(existing=existing)
    service = MemoryCompressionService(repository)

    result = service.create_daily_summary(date(2026, 5, 19))

    assert result is existing
    assert repository.find_calls == [
        ("daily-summary", "period:2026-05-19"),
    ]
    assert repository.list_range is None
    assert repository.saved_memory is None


def test_duplicate_weekly_returns_existing_without_saving() -> None:
    existing = _memory("Existing weekly compression", tags=["compression"])
    repository = FakeCompressionRepository(existing=existing)
    service = MemoryCompressionService(repository)

    result = service.create_weekly_summary(date(2026, 5, 18))

    assert result is existing
    assert repository.find_calls == [
        ("weekly-summary", "period:2026-05-18..2026-05-24"),
    ]
    assert repository.list_range is None
    assert repository.saved_memory is None


def test_non_monday_weekly_raises_memory_compression_error() -> None:
    repository = FakeCompressionRepository(sources=[_memory()])
    service = MemoryCompressionService(repository)

    with pytest.raises(MemoryCompressionError):
        service.create_weekly_summary(date(2026, 5, 19))

    assert repository.find_calls == []
    assert repository.list_range is None
    assert repository.saved_memory is None


def test_source_memories_are_not_mutated() -> None:
    source = _memory(
        "Original source",
        tags=["database"],
        topic="architecture",
        created_at=datetime(2026, 5, 19, tzinfo=timezone.utc),
    )
    original = (
        source.raw_text,
        source.clean_text,
        source.summary,
        list(source.tags),
        source.topic,
        source.created_at,
        source.processing_state,
        source.record_state,
        source.embedding,
    )
    repository = FakeCompressionRepository(sources=[source])
    service = MemoryCompressionService(repository)

    service.create_daily_summary(date(2026, 5, 19))

    assert (
        source.raw_text,
        source.clean_text,
        source.summary,
        source.tags,
        source.topic,
        source.created_at,
        source.processing_state,
        source.record_state,
        source.embedding,
    ) == original
