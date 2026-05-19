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
        summary=context,
        tags=["decision"],
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
