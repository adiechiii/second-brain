from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models.memory import Memory
from app.services.decision_evaluation import (
    decision_grouping_keys,
    detect_decision_mismatch_patterns,
)

NO_INTERVENTION_WARNING = {
    "warning": False,
    "risk_level": "none",
    "reason": "No repeated negative decision pattern matched",
    "reference_pattern": None,
}


def _serialize_timestamp(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _reference_pattern(pattern: dict[str, Any]) -> dict[str, Any]:
    return {
        "pattern_key": pattern["pattern_key"],
        "incorrect_count": pattern["incorrect_count"],
        "decision_ids": pattern["decision_ids"],
        "latest_outcome_timestamp": _serialize_timestamp(
            pattern.get("latest_outcome_timestamp")
        ),
    }


def assess_intervention_risk(
    candidate_memory: Memory,
    past_memories: list[Memory],
) -> dict[str, Any]:
    candidate_keys = set(decision_grouping_keys(candidate_memory))
    if not candidate_keys:
        return dict(NO_INTERVENTION_WARNING)

    repeated_patterns = detect_decision_mismatch_patterns(past_memories)
    for pattern in repeated_patterns:
        pattern_key = pattern.get("pattern_key")
        if pattern_key not in candidate_keys:
            continue

        incorrect_count = pattern["incorrect_count"]
        return {
            "warning": True,
            "risk_level": "high",
            "reason": (
                f"This matches {incorrect_count} past decision outcomes marked incorrect."
            ),
            "reference_pattern": _reference_pattern(pattern),
        }

    return dict(NO_INTERVENTION_WARNING)
