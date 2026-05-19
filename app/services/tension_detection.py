from __future__ import annotations

from collections import defaultdict
from typing import Any


TENSION_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "key": "tension:focus_vs_avoidance",
        "left": "focus",
        "right": "avoidance_delay",
        "label": "You show tension between wanting focus and delaying or avoiding action.",
    },
    {
        "key": "tension:risk_reduction_vs_faster_feedback",
        "left": "risk_reduction",
        "right": "faster_feedback",
        "label": "You show tension between reducing risk and seeking faster feedback.",
    },
    {
        "key": "tension:overcommitment_vs_focus",
        "left": "overcommitment",
        "right": "focus",
        "label": "You show tension between being overcommitted and wanting protected focus.",
    },
    {
        "key": "tension:low_clarity_vs_avoidance",
        "left": "low_clarity",
        "right": "avoidance_delay",
        "label": "You show tension between needing clarity and delaying action when clarity is low.",
    },
)


LOOP_CATEGORY_FIELDS = ("trigger", "action", "result")


def _safe_count(source: dict[str, Any]) -> int | None:
    count = source.get("count")
    if isinstance(count, bool):
        return None
    if not isinstance(count, int):
        return None
    return count


def _safe_string(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _safe_memory_ids(source: dict[str, Any]) -> list[str]:
    value = source.get("memory_ids")
    if not isinstance(value, list):
        return []
    return [str(memory_id) for memory_id in value]


def _source_evidence(
    *,
    side: str,
    source_type: str,
    source_key: str,
    count: int,
    memory_ids: list[str],
) -> dict[str, Any]:
    return {
        "side": side,
        "source_type": source_type,
        "source_key": source_key,
        "count": count,
        "memory_ids": memory_ids,
    }


def _category_count(
    *,
    category: str,
    memory_ids_by_category: dict[str, set[str]],
    fallback_count_by_category: dict[str, int],
) -> int:
    memory_ids = memory_ids_by_category.get(category, set())
    if memory_ids:
        return len(memory_ids)
    return fallback_count_by_category.get(category, 0)


def _collect_pattern_sources(
    patterns: list[dict[str, Any]],
    min_count: int,
    memory_ids_by_category: dict[str, set[str]],
    fallback_count_by_category: dict[str, int],
    sources_by_category: dict[str, list[dict[str, Any]]],
) -> None:
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue

        count = _safe_count(pattern)
        if count is None or count < min_count:
            continue

        category = _safe_string(pattern, "pattern")
        source_key = _safe_string(pattern, "key")
        if category is None or source_key is None:
            continue

        memory_ids = _safe_memory_ids(pattern)
        memory_ids_by_category[category].update(memory_ids)
        if not memory_ids:
            fallback_count_by_category[category] += count

        sources_by_category[category].append(
            {
                "source_type": "pattern",
                "source_key": source_key,
                "count": count,
                "memory_ids": memory_ids,
            }
        )


def _collect_loop_sources(
    loops: list[dict[str, Any]],
    min_count: int,
    memory_ids_by_category: dict[str, set[str]],
    fallback_count_by_category: dict[str, int],
    sources_by_category: dict[str, list[dict[str, Any]]],
) -> None:
    for loop in loops:
        if not isinstance(loop, dict):
            continue

        count = _safe_count(loop)
        if count is None or count < min_count:
            continue

        source_key = _safe_string(loop, "key")
        if source_key is None:
            continue

        memory_ids = _safe_memory_ids(loop)

        for field in LOOP_CATEGORY_FIELDS:
            category = _safe_string(loop, field)
            if category is None:
                continue

            memory_ids_by_category[category].update(memory_ids)
            if not memory_ids:
                fallback_count_by_category[category] += count

            sources_by_category[category].append(
                {
                    "source_type": "loop",
                    "source_key": source_key,
                    "count": count,
                    "memory_ids": memory_ids,
                }
            )


def _side_evidence(
    *,
    side: str,
    category: str,
    sources_by_category: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    evidence = []
    for source in sources_by_category.get(category, []):
        evidence.append(
            _source_evidence(
                side=side,
                source_type=source["source_type"],
                source_key=source["source_key"],
                count=source["count"],
                memory_ids=source["memory_ids"],
            )
        )

    return sorted(
        evidence,
        key=lambda item: (
            item["source_type"],
            item["source_key"],
            item["count"],
        ),
    )


def detect_tensions(
    patterns: list[dict[str, Any]],
    loops: list[dict[str, Any]] | None = None,
    min_count: int = 2,
) -> list[dict[str, Any]]:
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    loops = loops or []

    memory_ids_by_category: dict[str, set[str]] = defaultdict(set)
    fallback_count_by_category: dict[str, int] = defaultdict(int)
    sources_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    _collect_pattern_sources(
        patterns,
        min_count,
        memory_ids_by_category,
        fallback_count_by_category,
        sources_by_category,
    )
    _collect_loop_sources(
        loops,
        min_count,
        memory_ids_by_category,
        fallback_count_by_category,
        sources_by_category,
    )

    tensions: list[dict[str, Any]] = []

    for definition in TENSION_DEFINITIONS:
        left = definition["left"]
        right = definition["right"]

        left_count = _category_count(
            category=left,
            memory_ids_by_category=memory_ids_by_category,
            fallback_count_by_category=fallback_count_by_category,
        )
        right_count = _category_count(
            category=right,
            memory_ids_by_category=memory_ids_by_category,
            fallback_count_by_category=fallback_count_by_category,
        )

        if left_count < min_count or right_count < min_count:
            continue

        left_evidence = _side_evidence(
            side="left",
            category=left,
            sources_by_category=sources_by_category,
        )
        right_evidence = _side_evidence(
            side="right",
            category=right,
            sources_by_category=sources_by_category,
        )
        evidence = left_evidence + right_evidence

        source_keys = sorted({item["source_key"] for item in evidence})

        tensions.append(
            {
                "key": definition["key"],
                "label": definition["label"],
                "left": left,
                "right": right,
                "left_count": left_count,
                "right_count": right_count,
                "support_count": min(left_count, right_count),
                "status": "active",
                "source_keys": source_keys,
                "evidence": evidence,
            }
        )

    return sorted(
        tensions,
        key=lambda tension: (-tension["support_count"], tension["key"]),
    )
