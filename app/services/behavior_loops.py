from __future__ import annotations

import re
from typing import Any

from app.models.memory import Memory


TRIGGER_MARKERS: dict[str, tuple[str, ...]] = {
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
    "overcommitment": (
        "too many",
        "overcommit",
        "overcommitted",
        "overloaded",
        "capacity",
        "bandwidth",
        "stretched",
    ),
    "conflict": (
        "conflict",
        "confrontation",
        "hard conversation",
        "uncomfortable conversation",
        "disagreement",
    ),
    "distraction": (
        "distraction",
        "distracted",
        "scattered",
        "context switching",
        "interrupt",
        "interrupted",
    ),
    "pressure": (
        "pressure",
        "urgent",
        "urgency",
        "deadline",
        "rushed",
        "stress",
        "stressful",
    ),
}

ACTION_MARKERS: dict[str, tuple[str, ...]] = {
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
    "reduce_scope": (
        "smaller",
        "reduce scope",
        "narrow",
        "simplify",
        "simpler",
        "cut",
        "trim",
        "minimal",
        "smallest",
    ),
    "seek_feedback": (
        "feedback",
        "ask",
        "validate",
        "validation",
        "test",
        "experiment",
        "learn",
        "learning",
    ),
    "prioritize_focus": (
        "focus",
        "prioritize",
        "priority",
        "deep work",
        "attention",
        "block time",
        "protect time",
    ),
    "communicate": (
        "communicate",
        "message",
        "explain",
        "discuss",
        "conversation",
        "align",
        "clarify with",
    ),
}

RESULT_MARKERS: dict[str, tuple[str, ...]] = {
    "risk_reduction": (
        "risk",
        "risky",
        "safer",
        "safe",
        "reduce risk",
        "downside",
        "cautious",
    ),
    "faster_feedback": (
        "feedback",
        "learn",
        "learning",
        "validate",
        "validation",
        "sooner",
        "faster",
        "signal",
    ),
    "improved_clarity": (
        "clarity",
        "clear",
        "clearer",
        "understand",
        "understanding",
        "clarify",
        "specific",
    ),
    "reduced_overcommitment": (
        "capacity",
        "bandwidth",
        "overcommit",
        "fewer",
        "less",
        "reduce load",
        "sustainable",
    ),
    "protected_focus": (
        "focus",
        "focused",
        "deep work",
        "attention",
        "distraction",
        "protect time",
    ),
}

LOOP_LABELS: dict[str, str] = {
    "loop:low_clarity->avoidance_delay->risk_reduction": (
        "When clarity is low, you tend to delay or avoid, which aims to reduce risk."
    ),
    "loop:low_clarity->seek_feedback->improved_clarity": (
        "When clarity is low, you tend to seek feedback, which improves clarity."
    ),
    "loop:low_clarity->reduce_scope->faster_feedback": (
        "When clarity is low, you tend to reduce scope, which creates faster feedback."
    ),
    "loop:overcommitment->prioritize_focus->protected_focus": (
        "When capacity is stretched, you tend to prioritize focus, which protects attention."
    ),
    "loop:overcommitment->reduce_scope->reduced_overcommitment": (
        "When capacity is stretched, you tend to reduce scope, which lowers overcommitment."
    ),
    "loop:conflict->communicate->improved_clarity": (
        "When conflict appears, you tend to communicate, which improves clarity."
    ),
    "loop:distraction->prioritize_focus->protected_focus": (
        "When distraction appears, you tend to prioritize focus, which protects attention."
    ),
    "loop:pressure->reduce_scope->risk_reduction": (
        "When pressure is high, you tend to reduce scope, which aims to reduce risk."
    ),
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


def _primary_category(
    text: str,
    markers_by_category: dict[str, tuple[str, ...]],
) -> str | None:
    normalized = _normalize_text(text)
    for category, markers in markers_by_category.items():
        if any(_contains_marker(normalized, marker) for marker in markers):
            return category
    return None


def _loop_label(key: str) -> str:
    return LOOP_LABELS.get(key, "Repeated behavior loop detected.")


def detect_behavior_loops(
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

        context = _field_text(data, "context")
        reasoning = _field_text(data, "reasoning")
        expected_outcome = _field_text(data, "expected_outcome")
        if context is None or reasoning is None or expected_outcome is None:
            continue

        trigger = _primary_category(context, TRIGGER_MARKERS)
        action = _primary_category(reasoning, ACTION_MARKERS)
        result = _primary_category(expected_outcome, RESULT_MARKERS)
        if trigger is None or action is None or result is None:
            continue

        key = f"loop:{trigger}->{action}->{result}"
        memory_id = _memory_id(memory)

        if key not in grouped:
            grouped[key] = {
                "key": key,
                "trigger": trigger,
                "action": action,
                "result": result,
                "label": _loop_label(key),
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
                "trigger_text": context,
                "action_text": reasoning,
                "result_text": expected_outcome,
            }
        )

    results = []
    for entry in grouped.values():
        count = len(entry["memory_ids"])
        if count < min_count:
            continue

        results.append(
            {
                "key": entry["key"],
                "trigger": entry["trigger"],
                "action": entry["action"],
                "result": entry["result"],
                "label": entry["label"],
                "count": count,
                "memory_ids": entry["memory_ids"],
                "evidence": entry["evidence"],
            }
        )

    return sorted(results, key=lambda item: (-item["count"], item["key"]))
