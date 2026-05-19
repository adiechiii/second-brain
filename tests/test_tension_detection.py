from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.behavior_loops import detect_behavior_loops
from app.services.decision_patterns import detect_decision_patterns
from app.services.tension_detection import detect_tensions


def build_pattern(
    *,
    key: str,
    pattern: str,
    count: int = 2,
    memory_ids: list[str] | None = None,
) -> dict:
    return {
        "key": key,
        "pattern": pattern,
        "field": key.split(":", 1)[0] if ":" in key else "context",
        "label": "Repeated decision pattern detected.",
        "count": count,
        "memory_ids": memory_ids or [f"{pattern}-1", f"{pattern}-2"],
        "evidence": [],
    }


def build_loop(
    *,
    key: str = "loop:low_clarity->avoidance_delay->risk_reduction",
    trigger: str = "low_clarity",
    action: str = "avoidance_delay",
    result: str = "risk_reduction",
    count: int = 2,
    memory_ids: list[str] | None = None,
) -> dict:
    return {
        "key": key,
        "trigger": trigger,
        "action": action,
        "result": result,
        "label": "Repeated behavior loop detected.",
        "count": count,
        "memory_ids": memory_ids or ["loop-1", "loop-2"],
        "evidence": [],
    }


def build_decision_memory(
    *,
    context: str,
    reasoning: str,
    expected_outcome: str,
) -> Memory:
    return Memory(
        id=uuid4(),
        raw_text=context,
        memory_type="decision",
        decision_data={
            "context": context,
            "reasoning": reasoning,
            "expected_outcome": expected_outcome,
        },
        clean_text=context,
        processing_state=ProcessingState.CAPTURED,
        record_state=RecordState.ACTIVE,
    )


def test_detects_focus_vs_avoidance_tension_from_patterns():
    tensions = detect_tensions(
        [
            build_pattern(key="context:focus", pattern="focus"),
            build_pattern(key="reasoning:avoidance_delay", pattern="avoidance_delay"),
        ]
    )

    assert tensions == [
        {
            "key": "tension:focus_vs_avoidance",
            "label": "You show tension between wanting focus and delaying or avoiding action.",
            "left": "focus",
            "right": "avoidance_delay",
            "left_count": 2,
            "right_count": 2,
            "support_count": 2,
            "status": "active",
            "source_keys": ["context:focus", "reasoning:avoidance_delay"],
            "evidence": [
                {
                    "side": "left",
                    "source_type": "pattern",
                    "source_key": "context:focus",
                    "count": 2,
                    "memory_ids": ["focus-1", "focus-2"],
                },
                {
                    "side": "right",
                    "source_type": "pattern",
                    "source_key": "reasoning:avoidance_delay",
                    "count": 2,
                    "memory_ids": ["avoidance_delay-1", "avoidance_delay-2"],
                },
            ],
        }
    ]


def test_detects_risk_reduction_vs_faster_feedback_tension():
    tensions = detect_tensions(
        [
            build_pattern(key="reasoning:risk_reduction", pattern="risk_reduction"),
            build_pattern(
                key="expected_outcome:faster_feedback",
                pattern="faster_feedback",
            ),
        ]
    )

    keys = {tension["key"] for tension in tensions}

    assert "tension:risk_reduction_vs_faster_feedback" in keys


def test_detects_overcommitment_vs_focus_tension():
    tensions = detect_tensions(
        [
            build_pattern(key="context:overcommitment", pattern="overcommitment"),
            build_pattern(key="expected_outcome:focus", pattern="focus"),
        ]
    )

    keys = {tension["key"] for tension in tensions}

    assert "tension:overcommitment_vs_focus" in keys


def test_detects_low_clarity_vs_avoidance_from_loop():
    tensions = detect_tensions(
        patterns=[],
        loops=[
            build_loop(
                key="loop:low_clarity->avoidance_delay->risk_reduction",
                trigger="low_clarity",
                action="avoidance_delay",
                result="risk_reduction",
            )
        ],
    )

    assert tensions[0]["key"] == "tension:low_clarity_vs_avoidance"
    assert tensions[0]["source_keys"] == [
        "loop:low_clarity->avoidance_delay->risk_reduction"
    ]


def test_respects_min_count():
    tensions = detect_tensions(
        [
            build_pattern(key="context:focus", pattern="focus", count=1),
            build_pattern(
                key="reasoning:avoidance_delay",
                pattern="avoidance_delay",
                count=2,
            ),
        ],
        min_count=2,
    )

    assert tensions == []

    tensions = detect_tensions(
        [
            build_pattern(key="context:focus", pattern="focus", count=1),
            build_pattern(
                key="reasoning:avoidance_delay",
                pattern="avoidance_delay",
                count=1,
            ),
        ],
        min_count=1,
    )

    assert tensions[0]["key"] == "tension:focus_vs_avoidance"


def test_ignores_malformed_sources():
    tensions = detect_tensions(
        [
            {},
            {"key": "context:focus", "pattern": "focus", "count": "2"},
            {"key": "", "pattern": "focus", "count": 2},
            {"key": "context:focus", "pattern": "", "count": 2},
            "not-a-dict",
        ],
        loops=[
            {},
            {
                "key": "loop:x",
                "trigger": "focus",
                "action": "avoidance_delay",
                "result": "risk_reduction",
                "count": "2",
            },
            {
                "key": "",
                "trigger": "focus",
                "action": "avoidance_delay",
                "result": "risk_reduction",
                "count": 2,
            },
            "not-a-loop",
        ],
    )

    assert tensions == []


def test_uses_distinct_memory_ids_for_counts():
    tensions = detect_tensions(
        [
            build_pattern(
                key="context:focus",
                pattern="focus",
                count=2,
                memory_ids=["same-memory", "focus-2"],
            ),
            build_pattern(
                key="expected_outcome:focus",
                pattern="focus",
                count=2,
                memory_ids=["same-memory", "focus-3"],
            ),
            build_pattern(
                key="reasoning:avoidance_delay",
                pattern="avoidance_delay",
                count=2,
                memory_ids=["avoid-1", "avoid-2"],
            ),
        ]
    )

    assert tensions[0]["key"] == "tension:focus_vs_avoidance"
    assert tensions[0]["left_count"] == 3
    assert tensions[0]["right_count"] == 2
    assert tensions[0]["support_count"] == 2


def test_output_ordering_is_deterministic():
    patterns = [
        build_pattern(key="context:focus", pattern="focus"),
        build_pattern(key="reasoning:avoidance_delay", pattern="avoidance_delay"),
        build_pattern(key="context:overcommitment", pattern="overcommitment"),
        build_pattern(key="reasoning:risk_reduction", pattern="risk_reduction"),
        build_pattern(
            key="expected_outcome:faster_feedback",
            pattern="faster_feedback",
        ),
    ]

    first = detect_tensions(patterns)
    second = detect_tensions(patterns)

    assert first == second
    assert [tension["key"] for tension in first] == sorted(
        [tension["key"] for tension in first]
    )


def test_detection_to_tension_integration():
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

    patterns = detect_decision_patterns(memories)
    loops = detect_behavior_loops(memories)
    tensions = detect_tensions(patterns, loops)

    assert tensions[0]["key"] == "tension:low_clarity_vs_avoidance"


def test_invalid_min_count_raises_value_error():
    with pytest.raises(ValueError):
        detect_tensions([], min_count=0)
