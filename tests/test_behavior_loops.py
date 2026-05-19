from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.behavior_loops import detect_behavior_loops


def build_decision_memory(
    *,
    memory_type: str = "decision",
    context: str = "The task is unclear.",
    reasoning: str = "I will delay until it feels safer.",
    expected_outcome: str = "Reduce risk.",
    decision_data: dict | None | object = None,
) -> Memory:
    if decision_data is None and memory_type == "decision":
        decision_data = {
            "context": context,
            "reasoning": reasoning,
            "expected_outcome": expected_outcome,
        }
    elif memory_type != "decision":
        decision_data = None

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
        build_decision_memory(memory_type="memory"),
        build_decision_memory(memory_type="memory"),
    ]

    assert detect_behavior_loops(memories) == []


def test_ignores_malformed_decision_data():
    memories = [
        build_decision_memory(decision_data="not-a-dict"),
        build_decision_memory(decision_data={"context": "Unclear only."}),
    ]

    assert detect_behavior_loops(memories) == []


def test_detects_repeated_low_clarity_avoidance_risk_loop():
    memories = [
        build_decision_memory(
            context="The next step is unclear.",
            reasoning="I will delay until the safer option is obvious.",
            expected_outcome="Reduce downside risk.",
        ),
        build_decision_memory(
            context="The goal is vague and not clear.",
            reasoning="I tend to defer this to avoid a risky choice.",
            expected_outcome="Make the outcome safer.",
        ),
    ]

    loops = detect_behavior_loops(memories)

    assert loops == [
        {
            "key": "loop:low_clarity->avoidance_delay->risk_reduction",
            "trigger": "low_clarity",
            "action": "avoidance_delay",
            "result": "risk_reduction",
            "label": "When clarity is low, you tend to delay or avoid, which aims to reduce risk.",
            "count": 2,
            "memory_ids": [str(memory.id) for memory in memories],
            "evidence": [
                {
                    "memory_id": str(memories[0].id),
                    "trigger_text": "The next step is unclear.",
                    "action_text": "I will delay until the safer option is obvious.",
                    "result_text": "Reduce downside risk.",
                },
                {
                    "memory_id": str(memories[1].id),
                    "trigger_text": "The goal is vague and not clear.",
                    "action_text": "I tend to defer this to avoid a risky choice.",
                    "result_text": "Make the outcome safer.",
                },
            ],
        }
    ]


def test_detects_repeated_low_clarity_feedback_clarity_loop():
    memories = [
        build_decision_memory(
            context="The requirements are unclear.",
            reasoning="Ask for feedback and validate the next step.",
            expected_outcome="Get clearer direction.",
        ),
        build_decision_memory(
            context="The project scope is ambiguous.",
            reasoning="Run a small experiment to learn.",
            expected_outcome="Clarify what users need.",
        ),
    ]

    loops = detect_behavior_loops(memories)
    keys = {loop["key"] for loop in loops}

    assert "loop:low_clarity->seek_feedback->improved_clarity" in keys


def test_respects_min_count():
    memories = [
        build_decision_memory(
            context="The next step is unclear.",
            reasoning="I will delay.",
            expected_outcome="Reduce risk.",
        ),
        build_decision_memory(
            context="The deadline creates pressure.",
            reasoning="I will reduce scope.",
            expected_outcome="Reduce risk.",
        ),
    ]

    assert detect_behavior_loops(memories, min_count=2) == []

    loops = detect_behavior_loops(memories, min_count=1)
    keys = {loop["key"] for loop in loops}

    assert "loop:low_clarity->avoidance_delay->risk_reduction" in keys
    assert "loop:pressure->reduce_scope->risk_reduction" in keys


def test_output_ordering_is_deterministic():
    memories = [
        build_decision_memory(
            context="The deadline creates pressure.",
            reasoning="Reduce scope to the smallest release.",
            expected_outcome="Reduce risk.",
        ),
        build_decision_memory(
            context="The task is unclear.",
            reasoning="I will delay.",
            expected_outcome="Reduce risk.",
        ),
    ]

    first = detect_behavior_loops(memories, min_count=1)
    second = detect_behavior_loops(memories, min_count=1)

    assert first == second
    assert [loop["key"] for loop in first] == sorted(
        [loop["key"] for loop in first]
    )


def test_primary_category_priority_is_deterministic():
    memories = [
        build_decision_memory(
            context="The work is unclear and capacity is stretched.",
            reasoning="I will delay and reduce scope.",
            expected_outcome="Reduce risk and get clarity.",
        ),
        build_decision_memory(
            context="The plan is vague and bandwidth is limited.",
            reasoning="I will avoid and simplify.",
            expected_outcome="Make it safer and clearer.",
        ),
    ]

    loops = detect_behavior_loops(memories)

    assert loops[0]["key"] == "loop:low_clarity->avoidance_delay->risk_reduction"


def test_invalid_min_count_raises_value_error():
    with pytest.raises(ValueError):
        detect_behavior_loops([], min_count=0)
