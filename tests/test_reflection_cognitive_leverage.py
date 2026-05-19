from datetime import datetime, timezone
from uuid import uuid4

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.reflection_engine import build_grounded_questions, generate_reflection


def build_memory(raw_text: str = "Plain memory") -> Memory:
    return Memory(
        id=uuid4(),
        raw_text=raw_text,
        clean_text=raw_text,
        summary=raw_text,
        tags=["test"],
        topic="test",
        importance_score=0.5,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def build_decision_memory(
    *,
    context: str,
    reasoning: str,
    expected_outcome: str,
    outcome_evaluation: str | None = None,
    actual_outcome: str | None = None,
    outcome_timestamp=None,
    tags: list[str] | None = None,
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
        actual_outcome=actual_outcome,
        outcome_timestamp=outcome_timestamp,
        outcome_evaluation=outcome_evaluation,
        clean_text=context,
        summary=context,
        tags=tags or ["decision"],
        topic="decision",
        importance_score=0.8,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def test_reflection_includes_empty_cognitive_leverage_fields_without_decisions():
    reflection = generate_reflection([build_memory("Plain memory")])

    assert reflection["dominant_patterns"] == []
    assert reflection["belief_statements"] == []
    assert reflection["detected_loops"] == []
    assert reflection["detected_tensions"] == []
    assert reflection["grounded_questions"] == []
    assert reflection["decision_feedback"]["risk_signals"] == []


def test_reflection_includes_cognitive_leverage_findings_for_repeated_decisions():
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

    reflection = generate_reflection(memories)

    assert {pattern["key"] for pattern in reflection["dominant_patterns"]} >= {
        "context:low_clarity",
        "reasoning:avoidance_delay",
        "expected_outcome:risk_reduction",
    }
    assert {belief["belief_key"] for belief in reflection["belief_statements"]} >= {
        "belief:low_clarity",
        "belief:avoidance_delay",
        "belief:risk_reduction",
    }
    assert {loop["key"] for loop in reflection["detected_loops"]} == {
        "loop:low_clarity->avoidance_delay->risk_reduction"
    }
    assert {tension["key"] for tension in reflection["detected_tensions"]} >= {
        "tension:low_clarity_vs_avoidance"
    }
    assert reflection["grounded_questions"]
    assert reflection["grounded_questions"][0].startswith(
        "Where is this tension showing up in your current decisions:"
    )


def test_grounded_questions_are_deduped_and_limited():
    questions = build_grounded_questions(
        patterns=[
            {
                "label": "Repeated decision pattern detected.",
            }
        ],
        beliefs=[
            {
                "statement": "You show a repeated decision pattern.",
            }
        ],
        loops=[
            {
                "label": "Repeated behavior loop detected.",
            },
            {
                "label": "Repeated behavior loop detected.",
            },
        ],
        tensions=[
            {
                "label": "Repeated tension detected.",
            }
        ],
        limit=3,
    )

    assert questions == [
        "Where is this tension showing up in your current decisions: Repeated tension detected.",
        "What would help you interrupt this loop: Repeated behavior loop detected.",
        "What evidence would strengthen or challenge this belief: You show a repeated decision pattern.",
    ]


def test_grounded_questions_return_empty_for_non_positive_limit():
    assert build_grounded_questions([], [], [], [], limit=0) == []


def test_reflection_includes_decision_feedback_accuracy_and_trend():
    memories = [
        build_decision_memory(
            context="Older launch decision 1.",
            reasoning="Ship the release.",
            expected_outcome="Release safely.",
            actual_outcome="Delayed.",
            outcome_evaluation="incorrect",
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            tags=["launch"],
        ),
        build_decision_memory(
            context="Older launch decision 2.",
            reasoning="Ship the release.",
            expected_outcome="Release safely.",
            actual_outcome="Delayed.",
            outcome_evaluation="incorrect",
            outcome_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            tags=["launch"],
        ),
        build_decision_memory(
            context="Recent launch decision 1.",
            reasoning="Ship the release.",
            expected_outcome="Release safely.",
            actual_outcome="Shipped.",
            outcome_evaluation="correct",
            outcome_timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
            tags=["safe-launch"],
        ),
        build_decision_memory(
            context="Recent launch decision 2.",
            reasoning="Ship the release.",
            expected_outcome="Release safely.",
            actual_outcome="Unclear.",
            outcome_evaluation="uncertain",
            outcome_timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
            tags=["safe-launch"],
        ),
    ]

    feedback = generate_reflection(memories)["decision_feedback"]

    assert feedback["accuracy"]["total_evaluated"] == 4
    assert feedback["accuracy"]["correct_count"] == 1
    assert feedback["accuracy"]["incorrect_count"] == 2
    assert feedback["accuracy"]["uncertain_count"] == 1
    assert feedback["accuracy"]["trend"] == "insufficient_data"


def test_reflection_decision_feedback_includes_failure_loops_and_risk_signals():
    memories = [
        build_decision_memory(
            context="Launch decision A.",
            reasoning="Ship the release.",
            expected_outcome="Release safely.",
            actual_outcome="Delayed.",
            outcome_evaluation="incorrect",
            tags=["launch"],
        ),
        build_decision_memory(
            context="Launch decision B.",
            reasoning="Ship the release.",
            expected_outcome="Release safely.",
            actual_outcome="Delayed again.",
            outcome_evaluation="incorrect",
            tags=["launch"],
        ),
    ]

    feedback = generate_reflection(memories)["decision_feedback"]

    assert feedback["failure_loops"][0]["pattern_key"] == "tag:launch"
    assert feedback["risk_signals"] == [
        {
            "type": "repeated_incorrect_decisions",
            "severity": "high",
            "reason": "Repeated incorrect outcomes detected for the same decision pattern",
            "pattern_key": "tag:launch",
            "incorrect_count": 2,
        }
    ]
    assert feedback["behavior_reinforcement"]["broken_patterns"] == [
        {
            "pattern_key": "tag:launch",
            "incorrect_count": 2,
            "decision_ids": [str(memory.id) for memory in memories],
            "reason": "Repeated incorrect decisions share this grouping key",
        }
    ]


def test_reflection_decision_feedback_adds_declining_accuracy_risk_signal():
    memories = [
        build_decision_memory(
            context="Decision 1.",
            reasoning="Ship.",
            expected_outcome="Release.",
            actual_outcome="Worked.",
            outcome_evaluation="correct",
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            tags=["one"],
        ),
        build_decision_memory(
            context="Decision 2.",
            reasoning="Ship.",
            expected_outcome="Release.",
            actual_outcome="Worked.",
            outcome_evaluation="correct",
            outcome_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            tags=["two"],
        ),
        build_decision_memory(
            context="Decision 3.",
            reasoning="Ship.",
            expected_outcome="Release.",
            actual_outcome="Failed.",
            outcome_evaluation="incorrect",
            outcome_timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
            tags=["three"],
        ),
        build_decision_memory(
            context="Decision 4.",
            reasoning="Ship.",
            expected_outcome="Release.",
            actual_outcome="Failed.",
            outcome_evaluation="incorrect",
            outcome_timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
            tags=["four"],
        ),
    ]

    feedback = generate_reflection(memories)["decision_feedback"]

    assert feedback["accuracy"]["trend"] == "declining"
    assert {
        "type": "declining_decision_accuracy",
        "severity": "medium",
        "reason": "Recent evaluated decisions are less accurate than earlier decisions",
    } in feedback["risk_signals"]


def test_reflection_decision_feedback_returns_empty_risk_signals_when_none_detected():
    memories = [
        build_decision_memory(
            context="Decision 1.",
            reasoning="Ship.",
            expected_outcome="Release.",
            actual_outcome="Worked.",
            outcome_evaluation="correct",
            tags=["one"],
        ),
        build_decision_memory(
            context="Decision 2.",
            reasoning="Ship.",
            expected_outcome="Release.",
            actual_outcome="Unclear.",
            outcome_evaluation="uncertain",
            tags=["one"],
        ),
    ]

    assert generate_reflection(memories)["decision_feedback"]["risk_signals"] == []


def test_reflection_decision_feedback_includes_reinforced_patterns_and_ignores_uncertain():
    correct_a = build_decision_memory(
        context="Focus decision A.",
        reasoning="Protect focus.",
        expected_outcome="More focus.",
        actual_outcome="Worked.",
        outcome_evaluation="correct",
        tags=["focus"],
    )
    correct_b = build_decision_memory(
        context="Focus decision B.",
        reasoning="Protect focus.",
        expected_outcome="More focus.",
        actual_outcome="Worked.",
        outcome_evaluation="correct",
        tags=["focus"],
    )
    uncertain = build_decision_memory(
        context="Focus decision C.",
        reasoning="Protect focus.",
        expected_outcome="More focus.",
        actual_outcome="Unclear.",
        outcome_evaluation="uncertain",
        tags=["focus"],
    )

    reinforcement = generate_reflection(
        [correct_a, correct_b, uncertain]
    )["decision_feedback"]["behavior_reinforcement"]

    assert reinforcement["reinforced_patterns"] == [
        {
            "pattern_key": "tag:focus",
            "correct_count": 2,
            "decision_ids": [str(correct_a.id), str(correct_b.id)],
            "reason": "Repeated correct decisions share this grouping key",
        }
    ]
    assert reinforcement["broken_patterns"] == []


def test_reflection_decision_feedback_does_not_mutate_memories():
    memory = build_decision_memory(
        context="Launch decision.",
        reasoning="Ship the release.",
        expected_outcome="Release safely.",
        actual_outcome="Delayed.",
        outcome_evaluation="incorrect",
        tags=["launch"],
    )
    original_decision_data = dict(memory.decision_data)
    original_tags = list(memory.tags)

    generate_reflection([memory])

    assert memory.decision_data == original_decision_data
    assert memory.decision_data["reasoning"] == "Ship the release."
    assert memory.decision_data["expected_outcome"] == "Release safely."
    assert memory.tags == original_tags
