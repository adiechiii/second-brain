"""Deterministic memory compression builders.

This module creates unsaved summary memories only. It does not write to the
database, schedule work, call external services, or mutate source memories.
"""

from collections import Counter
from datetime import date, datetime, timedelta, timezone
import re

from app.models.memory import Memory, ProcessingState, RecordState

MAX_EVIDENCE_SNIPPETS = 5
MAX_SOURCE_THEMES = 5


class MemoryCompressionError(ValueError):
    """Raised when a compression summary cannot be built."""


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _source_tags(memory: Memory) -> list[str]:
    if not isinstance(memory.tags, list):
        return []
    return [
        tag.strip().lower()
        for tag in memory.tags
        if isinstance(tag, str) and tag.strip() and tag.strip().lower() != "compression"
    ]


def _source_topic(memory: Memory) -> str | None:
    if not isinstance(memory.topic, str):
        return None
    topic = memory.topic.strip().lower()
    return topic or None


def _is_compression_memory(memory: Memory) -> bool:
    if not isinstance(memory.tags, list):
        return False
    return any(
        isinstance(tag, str) and tag.strip().lower() == "compression"
        for tag in memory.tags
    )


def _evidence_text(memory: Memory) -> str:
    return (memory.summary or memory.clean_text or "").strip()


def _safe_importance(memory: Memory) -> float:
    try:
        importance = float(memory.importance_score or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(importance, 1.0))


def _recency_sort_value(memory: Memory) -> float:
    timestamp = memory.updated_at or memory.created_at
    if not isinstance(timestamp, datetime):
        return float("-inf")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.timestamp()


def _source_memories(memories: list[Memory]) -> list[Memory]:
    sources = [memory for memory in memories if not _is_compression_memory(memory)]
    if not sources:
        raise MemoryCompressionError("source memories must not be empty")
    return sources


def _evidence_snippets(memories: list[Memory]) -> list[str]:
    ranked = sorted(
        memories,
        key=lambda memory: (
            -_safe_importance(memory),
            -_recency_sort_value(memory),
            _normalized_text(_evidence_text(memory)),
        ),
    )
    snippets: list[str] = []
    seen: set[str] = set()
    for memory in ranked:
        evidence = _evidence_text(memory)
        normalized = _normalized_text(evidence)
        if not evidence or normalized in seen:
            continue
        snippets.append(evidence)
        seen.add(normalized)
        if len(snippets) == MAX_EVIDENCE_SNIPPETS:
            break

    if not snippets:
        raise MemoryCompressionError("source memories do not contain summary evidence")
    return snippets


def _top_source_themes(memories: list[Memory]) -> list[str]:
    counts: Counter[str] = Counter()
    for memory in memories:
        counts.update(set(_source_tags(memory)))
        if topic := _source_topic(memory):
            counts[topic] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [theme for theme, _ in ranked[:MAX_SOURCE_THEMES]]


def _max_importance(memories: list[Memory]) -> float:
    return max((_safe_importance(memory) for memory in memories), default=0.0)


def _build_summary_memory(
    memories: list[Memory],
    period_label: str,
    summary_type: str,
) -> Memory:
    sources = _source_memories(memories)
    snippets = _evidence_snippets(sources)
    summary = (
        f"Across {len(sources)} source memories for {period_label}, "
        "strongest evidence points to: "
        + " | ".join(snippets)
    )
    tags = [
        "compression",
        summary_type,
        f"period:{period_label}",
        *_top_source_themes(sources),
    ]

    return Memory(
        raw_text=summary,
        clean_text=summary,
        summary=summary,
        tags=tags,
        topic=summary_type,
        importance_score=_max_importance(sources),
        processing_state=ProcessingState.COMPLETED,
        record_state=RecordState.ACTIVE,
    )


def build_daily_summary(memories: list[Memory], day: date) -> Memory:
    period_label = day.isoformat()
    return _build_summary_memory(memories, period_label, "daily-summary")


def build_weekly_summary(memories: list[Memory], week_start: date) -> Memory:
    if week_start.weekday() != 0:
        raise MemoryCompressionError("week_start must be a Monday")

    week_end = week_start + timedelta(days=6)
    period_label = f"{week_start.isoformat()}..{week_end.isoformat()}"
    return _build_summary_memory(memories, period_label, "weekly-summary")
