"""Deterministic read-only idea thread construction."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from app.models.memory import Memory

IGNORED_THEME_TAGS = {"compression", "daily-summary", "weekly-summary"}


@dataclass(frozen=True)
class IdeaThread:
    theme: str
    memory_ids: list[UUID]
    first_seen: datetime | None
    last_seen: datetime | None
    evidence_count: int
    summaries: list[str]


def _normalized_tag(tag: object) -> str | None:
    if not isinstance(tag, str):
        return None
    normalized = tag.strip().lower()
    if not normalized:
        return None
    if normalized in IGNORED_THEME_TAGS or normalized.startswith("period:"):
        return None
    return normalized


def _normalized_topic(memory: Memory) -> str | None:
    if not isinstance(memory.topic, str):
        return None
    normalized = memory.topic.strip().lower()
    return normalized or None


def _normalized_tags(memory: Memory) -> set[str]:
    if not isinstance(memory.tags, list):
        return set()
    return {
        normalized
        for tag in memory.tags
        if (normalized := _normalized_tag(tag)) is not None
    }


def _is_compression_memory(memory: Memory) -> bool:
    if not isinstance(memory.tags, list):
        return False
    return any(
        isinstance(tag, str) and tag.strip().lower() == "compression"
        for tag in memory.tags
    )


def _memory_themes(memory: Memory) -> set[str]:
    themes = set(_normalized_tags(memory))
    if topic := _normalized_topic(memory):
        themes.add(topic)
    return themes


def _memory_evidence(memory: Memory) -> str:
    return (memory.summary or memory.clean_text or "").strip()


def _memory_datetime(memory: Memory) -> datetime | None:
    timestamp = memory.created_at or memory.updated_at
    return timestamp if isinstance(timestamp, datetime) else None


def _datetime_sort_value(timestamp: datetime | None) -> tuple[int, float]:
    if not isinstance(timestamp, datetime):
        return (1, 0.0)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (0, timestamp.timestamp())


def _thread_first_seen(memories: list[Memory]) -> datetime | None:
    timestamps = [
        timestamp
        for memory in memories
        if (timestamp := _memory_datetime(memory)) is not None
    ]
    if not timestamps:
        return None
    return min(timestamps, key=lambda timestamp: _datetime_sort_value(timestamp))


def _thread_last_seen(memories: list[Memory]) -> datetime | None:
    timestamps = [
        timestamp
        for memory in memories
        if (timestamp := _memory_datetime(memory)) is not None
    ]
    if not timestamps:
        return None
    return max(timestamps, key=lambda timestamp: _datetime_sort_value(timestamp))


def _memory_sort_key(memory: Memory) -> tuple[int, float, str]:
    missing, timestamp = _datetime_sort_value(_memory_datetime(memory))
    memory_id = str(memory.id) if memory.id is not None else ""
    return (missing, timestamp, memory_id)


def _thread_sort_key(thread: IdeaThread) -> tuple[int, int, float, str]:
    missing, timestamp = _datetime_sort_value(thread.last_seen)
    return (-thread.evidence_count, missing, -timestamp, thread.theme)


def _build_thread(
    theme: str,
    memories: list[Memory],
    max_summaries: int,
) -> IdeaThread:
    ordered_memories = sorted(memories, key=_memory_sort_key)
    summaries: list[str] = []
    for memory in ordered_memories:
        evidence = _memory_evidence(memory)
        if evidence:
            summaries.append(evidence)
        if len(summaries) == max_summaries:
            break

    return IdeaThread(
        theme=theme,
        memory_ids=[memory.id for memory in ordered_memories],
        first_seen=_thread_first_seen(ordered_memories),
        last_seen=_thread_last_seen(ordered_memories),
        evidence_count=len(ordered_memories),
        summaries=summaries,
    )


def build_idea_threads(
    memories: list[Memory],
    *,
    min_evidence_count: int = 2,
    max_summaries: int = 5,
) -> list[IdeaThread]:
    if not memories or min_evidence_count <= 0:
        return []

    grouped: dict[str, list[Memory]] = defaultdict(list)
    for memory in memories:
        if _is_compression_memory(memory):
            continue
        for theme in _memory_themes(memory):
            grouped[theme].append(memory)

    threads = [
        _build_thread(theme, theme_memories, max_summaries)
        for theme, theme_memories in grouped.items()
        if len(theme_memories) >= min_evidence_count
    ]
    return sorted(threads, key=_thread_sort_key)
