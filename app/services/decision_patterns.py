from __future__ import annotations

import re
from typing import Any

from app.models.memory import Memory


DECISION_FIELDS = ("context", "reasoning", "expected_outcome")

PATTERN_MARKERS: dict[str, tuple[str, ...]] = {
    "low_clarity": (
        "unclear",
        "ambiguous",
        "uncertain",
        "uncertainty",
        "vague",
        "clarity",
        "undefined",
        "confusing",
        "not clear",
    ),
    "avoidance_delay": (
        "avoid",
        "avoided",
        "avoiding",
        "delay",
        "delayed",
        "delaying",
        "defer",
        "deferred",
        "postpone",
        "postponed",
        "procrastinate",
        "procrastinating",
    ),
    "risk_reduction": (
        "risk",
        "risky",
        "safer",
        "safe",
        "reduce risk",
        "smaller",
        "cautious",
        "downside",
    ),
    "faster_feedback": (
        "feedback",
        "learn",
        "learning",
        "validate",
        "validation",
        "experiment",
        "sooner",
        "faster",
    ),
    "focus": (
        "focus",
        "focused",
        "deep work",
        "attention",
        "distraction",
        "distracted",
    ),
    "overcommitment": (
        "too many",
        "overcommit",
        "overcommitted",
        "overloaded",
        "capacity",
        "bandwidth",
        "stretched",
    ),
    "conflict_avoidance": (
        "conflict",
        "confrontation",
        "hard conversation",
        "uncomfortable conversation",
        "avoid conversation",
    ),
}

PATTERN_LABELS: dict[str, str] = {
    "context:low_clarity": "Decisions repeatedly happen when clarity is low.",
    "reasoning:low_clarity": "Reasoning repeatedly references low clarity.",
    "expected_outcome:low_clarity": "Expected outcomes repeatedly depend on gaining clarity.",
    "context:avoidance_delay": "Decisions repeatedly occur around avoidance or delay.",
    "reasoning:avoidance_delay": "Reasoning repeatedly includes avoidance or delay.",
    "expected_outcome:avoidance_delay": "Expected outcomes repeatedly aim to reduce avoidance or delay.",
    "context:risk_reduction": "Decisions repeatedly involve risk.",
    "reasoning:risk_reduction": "Reasoning repeatedly favors reducing risk.",
    "expected_outcome:risk_reduction": "Expected outcomes repeatedly prioritize safety or risk reduction.",
    "context:faster_feedback": "Decisions repeatedly involve feedback or learning.",
    "reasoning:faster_feedback": "Reasoning repeatedly favors faster feedback or learning.",
    "expected_outcome:faster_feedback": "Expected outcomes repeatedly prioritize faster feedback or learning.",
    "context:focus": "Decisions repeatedly involve focus or attention.",
    "reasoning:focus": "Reasoning repeatedly references focus or attention.",
    "expected_outcome:focus": "Expected outcomes repeatedly prioritize focus.",
    "context:overcommitment": "Decisions repeatedly involve capacity or overcommitment.",
    "reasoning:overcommitment": "Reasoning repeatedly references capacity or overcommitment.",
    "expected_outcome:overcommitment": "Expected outcomes repeatedly aim to reduce overcommitment.",
    "context:conflict_avoidance": "Decisions repeatedly involve conflict or hard conversations.",
    "reasoning:conflict_avoidance": "Reasoning repeatedly references conflict avoidance.",
    "expected_outcome:conflict_avoidance": "Expected outcomes repeatedly aim to reduce conflict or avoidance.",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _contains_marker(text: str, marker: str) -> bool:
    normalized_marker = _normalize_text(marker)
    if " " in normalized_marker:
        return normalized_marker in text

    return re.search(rf"\b{re.escape(normalized_marker)}\b", text) is not None


def _decision_data(memory: Memory) -> dict[str, Any] | None:
    if getattr(memory, "memory_type", None) != "decision":
        return None
    data = getattr(memory, "decision_data", None)
    if not isinstance(data, dict):
        return None
    return data


def _field_text(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _memory_id(memory: Memory) -> str:
    return str(getattr(memory, "id", ""))


def _matching_patterns(text: str) -> set[str]:
    normalized = _normalize_text(text)
    matches = set()

    for pattern, markers in PATTERN_MARKERS.items():
        if any(_contains_marker(normalized, marker) for marker in markers):
            matches.add(pattern)

    return matches


def detect_decision_patterns(
    memories: list[Memory],
    min_count: int = 2,
) -> list[dict[str, Any]]:
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    grouped: dict[str, dict[str, Any]] = {}

    for memory in memories:
        data = _decision_data(memory)
        if data is None:
            continue

        memory_id = _memory_id(memory)

        for field in DECISION_FIELDS:
            text = _field_text(data, field)
            if text is None:
                continue

            for pattern in _matching_patterns(text):
                key = f"{field}:{pattern}"

                if key not in grouped:
                    grouped[key] = {
                        "key": key,
                        "pattern": pattern,
                        "field": field,
                        "label": PATTERN_LABELS.get(
                            key,
                            "Repeated decision pattern detected.",
                        ),
                        "memory_ids": [],
                        "evidence": [],
                        "_seen_memory_ids": set(),
                    }

                entry = grouped[key]
                if memory_id in entry["_seen_memory_ids"]:
                    continue

                entry["_seen_memory_ids"].add(memory_id)
                entry["memory_ids"].append(memory_id)
                entry["evidence"].append(
                    {
                        "memory_id": memory_id,
                        "field": field,
                        "text": text,
                    }
                )

    results = []
    for entry in grouped.values():
        count = len(entry["memory_ids"])
        if count < min_count:
            continue

        cleaned = {
            "key": entry["key"],
            "pattern": entry["pattern"],
            "field": entry["field"],
            "label": entry["label"],
            "count": count,
            "memory_ids": entry["memory_ids"],
            "evidence": entry["evidence"],
        }
        results.append(cleaned)

    return sorted(results, key=lambda item: (-item["count"], item["key"]))
