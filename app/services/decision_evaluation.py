from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from app.models.memory import Memory

EVALUATION_CORRECT = "correct"
EVALUATION_INCORRECT = "incorrect"
EVALUATION_UNCERTAIN = "uncertain"
EVALUATED_OUTCOMES = {EVALUATION_CORRECT, EVALUATION_INCORRECT}
TREND_THRESHOLD = 0.10


def _decision_data(memory: Memory) -> dict[str, Any] | None:
    if getattr(memory, "memory_type", None) != "decision":
        return None
    data = getattr(memory, "decision_data", None)
    if not isinstance(data, dict):
        return None
    return data


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _expected_outcome(memory: Memory) -> str | None:
    data = _decision_data(memory)
    if data is None:
        return None
    return _safe_text(data.get("expected_outcome"))


def _memory_id(memory: Memory) -> str:
    return str(getattr(memory, "id", ""))


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _safe_timestamp(memory: Memory) -> datetime | None:
    timestamp = memory.outcome_timestamp or memory.created_at
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def evaluate_decision_outcome(memory: Memory) -> dict[str, Any]:
    expected_outcome = _expected_outcome(memory)
    actual_outcome = _safe_text(getattr(memory, "actual_outcome", None))
    outcome_evaluation = _safe_text(getattr(memory, "outcome_evaluation", None))

    if actual_outcome is None:
        return {
            "decision_id": _memory_id(memory),
            "expected_outcome": expected_outcome,
            "actual_outcome": None,
            "outcome_evaluation": outcome_evaluation,
            "has_outcome": False,
            "expectation_gap": False,
            "evaluation_status": "pending",
            "reason": "No actual outcome recorded",
        }

    if outcome_evaluation == EVALUATION_CORRECT:
        return {
            "decision_id": _memory_id(memory),
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "outcome_evaluation": outcome_evaluation,
            "has_outcome": True,
            "expectation_gap": False,
            "evaluation_status": "correct",
            "reason": "Outcome marked correct by explicit outcome_evaluation",
        }

    if outcome_evaluation == EVALUATION_INCORRECT:
        return {
            "decision_id": _memory_id(memory),
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "outcome_evaluation": outcome_evaluation,
            "has_outcome": True,
            "expectation_gap": True,
            "evaluation_status": "incorrect",
            "reason": "Outcome marked incorrect by explicit outcome_evaluation",
        }

    if outcome_evaluation == EVALUATION_UNCERTAIN:
        return {
            "decision_id": _memory_id(memory),
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "outcome_evaluation": outcome_evaluation,
            "has_outcome": True,
            "expectation_gap": None,
            "evaluation_status": "uncertain",
            "reason": "Outcome marked uncertain by explicit outcome_evaluation",
        }

    return {
        "decision_id": _memory_id(memory),
        "expected_outcome": expected_outcome,
        "actual_outcome": actual_outcome,
        "outcome_evaluation": outcome_evaluation,
        "has_outcome": True,
        "expectation_gap": None,
        "evaluation_status": "unevaluated",
        "reason": "Actual outcome recorded but no evaluation supplied",
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _accuracy_rate(memories: list[Memory]) -> float | None:
    correct_count = sum(1 for memory in memories if memory.outcome_evaluation == EVALUATION_CORRECT)
    incorrect_count = sum(
        1 for memory in memories if memory.outcome_evaluation == EVALUATION_INCORRECT
    )
    return _rate(correct_count, correct_count + incorrect_count)


def _evaluated_for_trend(memories: list[Memory]) -> list[Memory]:
    return [
        memory
        for memory in memories
        if getattr(memory, "outcome_evaluation", None) in EVALUATED_OUTCOMES
    ]


def _trend(memories: list[Memory]) -> str:
    evaluated = sorted(
        _evaluated_for_trend(memories),
        key=lambda memory: (
            _safe_timestamp(memory) or datetime.max.replace(tzinfo=timezone.utc),
            _memory_id(memory),
        ),
    )
    if len(evaluated) < 4:
        return "insufficient_data"

    midpoint = len(evaluated) // 2
    older = evaluated[:midpoint]
    recent = evaluated[midpoint:]
    older_rate = _accuracy_rate(older)
    recent_rate = _accuracy_rate(recent)
    if older_rate is None or recent_rate is None:
        return "insufficient_data"

    difference = recent_rate - older_rate
    if difference > TREND_THRESHOLD:
        return "improving"
    if difference < -TREND_THRESHOLD:
        return "declining"
    return "stable"


def get_decision_accuracy_patterns(memories: list[Memory]) -> dict[str, Any]:
    decisions = [
        memory
        for memory in memories
        if getattr(memory, "memory_type", None) == "decision"
    ]
    correct_count = sum(
        1 for memory in decisions if memory.outcome_evaluation == EVALUATION_CORRECT
    )
    incorrect_count = sum(
        1 for memory in decisions if memory.outcome_evaluation == EVALUATION_INCORRECT
    )
    uncertain_count = sum(
        1 for memory in decisions if memory.outcome_evaluation == EVALUATION_UNCERTAIN
    )
    total_evaluated = correct_count + incorrect_count + uncertain_count
    denominator = correct_count + incorrect_count

    return {
        "total_evaluated": total_evaluated,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "uncertain_count": uncertain_count,
        "accuracy_rate": _rate(correct_count, denominator),
        "incorrect_rate": _rate(incorrect_count, denominator),
        "uncertain_rate": _rate(uncertain_count, total_evaluated),
        "trend": _trend(decisions),
    }


def _string_from_decision_data(memory: Memory, key: str) -> str | None:
    data = _decision_data(memory)
    if data is None:
        return None
    return _safe_text(data.get(key))


def _tag_keys(memory: Memory) -> list[str]:
    tags = memory.tags or []
    normalized = sorted(
        {
            _normalized_text(tag)
            for tag in tags
            if isinstance(tag, str) and tag.strip()
        }
    )
    return [f"tag:{tag}" for tag in normalized]


def _content_key(memory: Memory) -> str:
    source = (
        _string_from_decision_data(memory, "context")
        or _safe_text(memory.summary)
        or _safe_text(memory.clean_text)
        or ""
    )
    normalized = _normalized_text(source)
    return f"content:{normalized[:80]}"


def decision_grouping_keys(memory: Memory) -> list[str]:
    for key in ("pattern_key", "pattern_id", "loop_key", "loop_id"):
        value = _string_from_decision_data(memory, key)
        if value is not None:
            return [f"{key}:{_normalized_text(value)}"]

    for key in ("decision_category", "decision_type", "category", "type"):
        value = _string_from_decision_data(memory, key)
        if value is not None:
            return [f"{key}:{_normalized_text(value)}"]

    tag_keys = _tag_keys(memory)
    if tag_keys:
        return tag_keys

    return [_content_key(memory)]


def detect_decision_mismatch_patterns(
    memories: list[Memory],
    min_count: int = 2,
) -> list[dict[str, Any]]:
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    grouped: dict[str, dict[str, Any]] = {}
    for memory in memories:
        if (
            getattr(memory, "memory_type", None) != "decision"
            or getattr(memory, "outcome_evaluation", None) != EVALUATION_INCORRECT
        ):
            continue

        for pattern_key in decision_grouping_keys(memory):
            if pattern_key not in grouped:
                grouped[pattern_key] = {
                    "pattern_key": pattern_key,
                    "decision_ids": [],
                    "latest_outcome_timestamp": None,
                }

            entry = grouped[pattern_key]
            decision_id = _memory_id(memory)
            if decision_id in entry["decision_ids"]:
                continue

            entry["decision_ids"].append(decision_id)
            timestamp = _safe_timestamp(memory)
            if timestamp is not None and (
                entry["latest_outcome_timestamp"] is None
                or timestamp > entry["latest_outcome_timestamp"]
            ):
                entry["latest_outcome_timestamp"] = timestamp

    results = []
    for entry in grouped.values():
        incorrect_count = len(entry["decision_ids"])
        if incorrect_count < min_count:
            continue

        results.append(
            {
                "pattern_key": entry["pattern_key"],
                "incorrect_count": incorrect_count,
                "decision_ids": entry["decision_ids"],
                "latest_outcome_timestamp": entry["latest_outcome_timestamp"],
                "reason": "Repeated incorrect decisions share this grouping key",
            }
        )

    return sorted(
        results,
        key=lambda item: (
            -item["incorrect_count"],
            item["pattern_key"],
        ),
    )
