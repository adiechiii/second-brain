from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.belief_consolidation import convert_patterns_to_beliefs
from app.services.decision_patterns import detect_decision_patterns


def build_pattern(
    *,
    key: str = "context:low_clarity",
    pattern: str = "low_clarity",
    count: int = 2,
    memory_ids: list[str] | None = None,
) -> dict:
    ids = memory_ids or ["memory-1", "memory-2"]
    return {
        "key": key,
        "pattern": pattern,
        "field": key.split(":", 1)[0] if ":" in key else "context",
        "label": "Repeated decision pattern detected.",
        "count": count,
        "memory_ids": ids,
        "evidence": [
            {
                "memory_id": ids[0],
                "field": "context",
                "text": "The next step is unclear.",
            }
        ],
    }


def build_decision_memory(context: str) -> Memory:
    return Memory(
        id=uuid4(),
        raw_text=context,
        memory_type="decision",
        decision_data={
            "context": context,
            "reasoning": "This option matches the current plan.",
            "expected_outcome": "Complete the release.",
        },
        clean_text=context,
        processing_state=ProcessingState.CAPTURED,
        record_state=RecordState.ACTIVE,
    )


def test_converts_repeated_pattern_to_belief():
    beliefs = convert_patterns_to_beliefs(
        [
            build_pattern(
                key="context:low_clarity",
                pattern="low_clarity",
                count=2,
            )
        ]
    )

    assert beliefs == [
        {
            "belief_key": "belief:low_clarity",
            "source_pattern_key": "context:low_clarity",
            "pattern": "low_clarity",
            "statement": "You tend to make decisions when clarity is low.",
            "support_count": 2,
            "confidence": "low",
            "status": "reinforced",
            "memory_ids": ["memory-1", "memory-2"],
            "evidence": [
                {
                    "memory_id": "memory-1",
                    "field": "context",
                    "text": "The next step is unclear.",
                }
            ],
        }
    ]


def test_ignores_patterns_below_min_count():
    beliefs = convert_patterns_to_beliefs(
        [build_pattern(count=1)],
        min_count=2,
    )

    assert beliefs == []


def test_confidence_thresholds():
    beliefs = convert_patterns_to_beliefs(
        [
            build_pattern(key="context:low_clarity", pattern="low_clarity", count=2),
            build_pattern(key="context:focus", pattern="focus", count=3),
            build_pattern(
                key="context:risk_reduction",
                pattern="risk_reduction",
                count=5,
            ),
        ],
        min_count=1,
    )

    confidence_by_pattern = {
        belief["pattern"]: belief["confidence"]
        for belief in beliefs
    }

    assert confidence_by_pattern == {
        "low_clarity": "low",
        "focus": "medium",
        "risk_reduction": "high",
    }


def test_ignores_malformed_patterns():
    beliefs = convert_patterns_to_beliefs(
        [
            {},
            {"key": "context:low_clarity", "pattern": "low_clarity", "count": "2"},
            {"key": "", "pattern": "low_clarity", "count": 2},
            {"key": "context:low_clarity", "pattern": "", "count": 2},
            "not-a-dict",
        ]
    )

    assert beliefs == []


def test_fallback_statement_by_pattern():
    beliefs = convert_patterns_to_beliefs(
        [
            build_pattern(
                key="custom:focus",
                pattern="focus",
                count=2,
            )
        ]
    )

    assert beliefs[0]["statement"] == "You show a repeated pattern around focus."


def test_final_fallback_statement_for_unknown_pattern():
    beliefs = convert_patterns_to_beliefs(
        [
            build_pattern(
                key="custom:unknown",
                pattern="unknown",
                count=2,
            )
        ]
    )

    assert beliefs[0]["statement"] == "You show a repeated decision pattern."


def test_output_ordering_is_deterministic():
    patterns = [
        build_pattern(key="context:focus", pattern="focus", count=2),
        build_pattern(key="context:low_clarity", pattern="low_clarity", count=3),
        build_pattern(key="reasoning:low_clarity", pattern="low_clarity", count=3),
    ]

    first = convert_patterns_to_beliefs(patterns)
    second = convert_patterns_to_beliefs(patterns)

    assert first == second
    assert [belief["source_pattern_key"] for belief in first] == [
        "context:low_clarity",
        "reasoning:low_clarity",
        "context:focus",
    ]


def test_detection_to_belief_conversion_integration():
    memories = [
        build_decision_memory("The next step is unclear."),
        build_decision_memory("The goal is vague and not clear."),
    ]

    patterns = detect_decision_patterns(memories)
    beliefs = convert_patterns_to_beliefs(patterns)

    assert beliefs[0]["source_pattern_key"] == "context:low_clarity"
    assert beliefs[0]["statement"] == "You tend to make decisions when clarity is low."


def test_invalid_min_count_raises_value_error():
    with pytest.raises(ValueError):
        convert_patterns_to_beliefs([], min_count=0)
