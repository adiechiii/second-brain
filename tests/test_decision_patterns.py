from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.decision_patterns import detect_decision_patterns


def build_memory(
    *,
    memory_type: str = "decision",
    context: str = "The next step is specific.",
    reasoning: str = "This option matches the current plan.",
    expected_outcome: str = "Complete the release.",
) -> Memory:
    decision_data = None
    if memory_type == "decision":
        decision_data = {
            "context": context,
            "reasoning": reasoning,
            "expected_outcome": expected_outcome,
        }

    return Memory(
        id=uuid4(),
        raw_text=context,
        memory_type=memory_type,
        decision_data=decision_data,
        clean_text=context,
        processing_state=ProcessingState.CAPTURED,
        record_state=RecordState.ACTIVE,
    )


def test_ignores_non_decision_memories():
    memories = [
        build_memory(memory_type="memory"),
        build_memory(memory_type="memory"),
    ]

    assert detect_decision_patterns(memories) == []


def test_detects_repeated_context_low_clarity_pattern():
    memories = [
        build_memory(context="The next step is unclear."),
        build_memory(context="The goal is vague and not clear."),
    ]

    patterns = detect_decision_patterns(memories)

    assert patterns[0]["key"] == "context:low_clarity"
    assert patterns[0]["count"] == 2
    assert patterns[0]["label"] == "Decisions repeatedly happen when clarity is low."
    assert len(patterns[0]["memory_ids"]) == 2
    assert len(patterns[0]["evidence"]) == 2


def test_detects_repeated_reasoning_risk_reduction_pattern():
    memories = [
        build_memory(reasoning="A smaller choice is safer."),
        build_memory(reasoning="This reduces downside risk."),
    ]

    patterns = detect_decision_patterns(memories)
    keys = {pattern["key"] for pattern in patterns}

    assert "reasoning:risk_reduction" in keys


def test_respects_min_count():
    memories = [
        build_memory(context="The next step is unclear."),
        build_memory(context="The goal is specific."),
    ]

    assert detect_decision_patterns(memories, min_count=2) == []

    patterns = detect_decision_patterns(memories, min_count=1)
    keys = {pattern["key"] for pattern in patterns}

    assert "context:low_clarity" in keys


def test_output_ordering_is_deterministic():
    memories = [
        build_memory(
            context="The next step is unclear.",
            reasoning="A smaller choice is safer.",
        ),
        build_memory(
            context="The goal is vague.",
            reasoning="This reduces risk.",
        ),
    ]

    first = detect_decision_patterns(memories)
    second = detect_decision_patterns(memories)

    assert first == second
    assert [pattern["key"] for pattern in first] == sorted(
        [pattern["key"] for pattern in first]
    )


def test_invalid_min_count_raises_value_error():
    with pytest.raises(ValueError):
        detect_decision_patterns([], min_count=0)
