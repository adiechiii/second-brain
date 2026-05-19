from __future__ import annotations

from typing import Any


SOURCE_BELIEF_STATEMENTS: dict[str, str] = {
    "context:low_clarity": "You tend to make decisions when clarity is low.",
    "reasoning:low_clarity": "You tend to reason from a need for more clarity.",
    "expected_outcome:low_clarity": "You tend to value outcomes that increase clarity.",
    "context:avoidance_delay": "You tend to make decisions around avoidance or delay.",
    "reasoning:avoidance_delay": "You tend to reason in ways that address avoidance or delay.",
    "expected_outcome:avoidance_delay": "You tend to value outcomes that reduce avoidance or delay.",
    "context:risk_reduction": "You tend to make decisions in situations involving risk.",
    "reasoning:risk_reduction": "You tend to favor lower-risk options when reasoning through decisions.",
    "expected_outcome:risk_reduction": "You tend to value outcomes that reduce risk.",
    "context:faster_feedback": "You tend to make decisions around learning or feedback.",
    "reasoning:faster_feedback": "You tend to favor faster feedback when reasoning through decisions.",
    "expected_outcome:faster_feedback": "You tend to value outcomes that create faster feedback.",
    "context:focus": "You tend to make decisions around focus and attention.",
    "reasoning:focus": "You tend to reason from a desire for more focus.",
    "expected_outcome:focus": "You tend to value outcomes that protect focus.",
    "context:overcommitment": "You tend to make decisions around capacity or overcommitment.",
    "reasoning:overcommitment": "You tend to reason from capacity constraints.",
    "expected_outcome:overcommitment": "You tend to value outcomes that reduce overcommitment.",
    "context:conflict_avoidance": "You tend to make decisions around conflict or hard conversations.",
    "reasoning:conflict_avoidance": "You tend to reason in ways that reduce conflict exposure.",
    "expected_outcome:conflict_avoidance": "You tend to value outcomes that reduce conflict or avoidance.",
}

PATTERN_BELIEF_STATEMENTS: dict[str, str] = {
    "low_clarity": "You show a repeated pattern around low clarity.",
    "avoidance_delay": "You show a repeated pattern around avoidance or delay.",
    "risk_reduction": "You show a repeated pattern around risk reduction.",
    "faster_feedback": "You show a repeated pattern around faster feedback.",
    "focus": "You show a repeated pattern around focus.",
    "overcommitment": "You show a repeated pattern around overcommitment.",
    "conflict_avoidance": "You show a repeated pattern around conflict avoidance.",
}


def _safe_count(pattern: dict[str, Any]) -> int | None:
    count = pattern.get("count")
    if isinstance(count, bool):
        return None
    if not isinstance(count, int):
        return None
    return count


def _safe_string(pattern: dict[str, Any], key: str) -> str | None:
    value = pattern.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _safe_list(pattern: dict[str, Any], key: str) -> list[Any]:
    value = pattern.get(key)
    if not isinstance(value, list):
        return []
    return value


def _confidence(support_count: int) -> str:
    if support_count >= 5:
        return "high"
    if support_count >= 3:
        return "medium"
    return "low"


def _statement(source_pattern_key: str, pattern_name: str) -> str:
    return SOURCE_BELIEF_STATEMENTS.get(
        source_pattern_key,
        PATTERN_BELIEF_STATEMENTS.get(
            pattern_name,
            "You show a repeated decision pattern.",
        ),
    )


def convert_patterns_to_beliefs(
    patterns: list[dict[str, Any]],
    min_count: int = 2,
) -> list[dict[str, Any]]:
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    beliefs: list[dict[str, Any]] = []

    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue

        support_count = _safe_count(pattern)
        if support_count is None or support_count < min_count:
            continue

        source_pattern_key = _safe_string(pattern, "key")
        pattern_name = _safe_string(pattern, "pattern")
        if source_pattern_key is None or pattern_name is None:
            continue

        belief = {
            "belief_key": f"belief:{pattern_name}",
            "source_pattern_key": source_pattern_key,
            "pattern": pattern_name,
            "statement": _statement(source_pattern_key, pattern_name),
            "support_count": support_count,
            "confidence": _confidence(support_count),
            "status": "reinforced",
            "memory_ids": [
                str(memory_id) for memory_id in _safe_list(pattern, "memory_ids")
            ],
            "evidence": _safe_list(pattern, "evidence"),
        }
        beliefs.append(belief)

    return sorted(
        beliefs,
        key=lambda belief: (
            -belief["support_count"],
            belief["belief_key"],
            belief["source_pattern_key"],
        ),
    )
