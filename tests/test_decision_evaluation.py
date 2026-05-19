from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.decision_evaluation import (
    detect_decision_mismatch_patterns,
    evaluate_decision_outcome,
    get_decision_accuracy_patterns,
)


def build_decision_memory(
    *,
    expected_outcome: str = "Release safely and learn sooner.",
    actual_outcome: str | None = None,
    outcome_evaluation: str | None = None,
    outcome_timestamp: datetime | None = None,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    decision_data_extra: dict | None = None,
) -> Memory:
    decision_data = {
        "context": "I need to choose a release size.",
        "reasoning": "A smaller release is safer.",
        "expected_outcome": expected_outcome,
    }
    if decision_data_extra:
        decision_data.update(decision_data_extra)

    return Memory(
        id=uuid4(),
        raw_text="Decision note",
        memory_type="decision",
        decision_data=decision_data,
        actual_outcome=actual_outcome,
        outcome_timestamp=outcome_timestamp,
        outcome_evaluation=outcome_evaluation,
        clean_text="Decision note",
        summary="Decision note",
        tags=tags,
        created_at=created_at,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def test_evaluate_decision_outcome_pending_without_actual_outcome():
    memory = build_decision_memory()

    evaluation = evaluate_decision_outcome(memory)

    assert evaluation == {
        "decision_id": str(memory.id),
        "expected_outcome": "Release safely and learn sooner.",
        "actual_outcome": None,
        "outcome_evaluation": None,
        "has_outcome": False,
        "expectation_gap": False,
        "evaluation_status": "pending",
        "reason": "No actual outcome recorded",
    }


def test_evaluate_decision_outcome_correct():
    memory = build_decision_memory(
        actual_outcome="The release shipped safely.",
        outcome_evaluation="correct",
    )

    evaluation = evaluate_decision_outcome(memory)

    assert evaluation["has_outcome"] is True
    assert evaluation["evaluation_status"] == "correct"
    assert evaluation["expectation_gap"] is False
    assert evaluation["reason"] == "Outcome marked correct by explicit outcome_evaluation"


def test_evaluate_decision_outcome_incorrect_creates_expectation_gap():
    memory = build_decision_memory(
        actual_outcome="The release was delayed.",
        outcome_evaluation="incorrect",
    )

    evaluation = evaluate_decision_outcome(memory)

    assert evaluation["has_outcome"] is True
    assert evaluation["evaluation_status"] == "incorrect"
    assert evaluation["expectation_gap"] is True
    assert evaluation["reason"] == "Outcome marked incorrect by explicit outcome_evaluation"


def test_evaluate_decision_outcome_uncertain():
    memory = build_decision_memory(
        actual_outcome="The release shipped, but quality is unclear.",
        outcome_evaluation="uncertain",
    )

    evaluation = evaluate_decision_outcome(memory)

    assert evaluation["has_outcome"] is True
    assert evaluation["evaluation_status"] == "uncertain"
    assert evaluation["expectation_gap"] is None
    assert evaluation["reason"] == "Outcome marked uncertain by explicit outcome_evaluation"


def test_evaluate_decision_outcome_with_actual_outcome_without_evaluation():
    memory = build_decision_memory(
        actual_outcome="The release shipped, but no evaluation was recorded.",
    )

    evaluation = evaluate_decision_outcome(memory)

    assert evaluation["has_outcome"] is True
    assert evaluation["evaluation_status"] == "unevaluated"
    assert evaluation["expectation_gap"] is None
    assert evaluation["reason"] == "Actual outcome recorded but no evaluation supplied"


def test_accuracy_patterns_count_evaluations_and_rates():
    memories = [
        build_decision_memory(outcome_evaluation="correct", actual_outcome="A"),
        build_decision_memory(outcome_evaluation="correct", actual_outcome="B"),
        build_decision_memory(outcome_evaluation="incorrect", actual_outcome="C"),
        build_decision_memory(outcome_evaluation="uncertain", actual_outcome="D"),
        build_decision_memory(actual_outcome="E"),
    ]

    metrics = get_decision_accuracy_patterns(memories)

    assert metrics["total_evaluated"] == 4
    assert metrics["correct_count"] == 2
    assert metrics["incorrect_count"] == 1
    assert metrics["uncertain_count"] == 1
    assert metrics["accuracy_rate"] == pytest.approx(2 / 3)
    assert metrics["incorrect_rate"] == pytest.approx(1 / 3)
    assert metrics["uncertain_rate"] == pytest.approx(1 / 4)


def test_accuracy_patterns_handle_zero_denominator():
    memories = [
        build_decision_memory(outcome_evaluation="uncertain", actual_outcome="A"),
    ]

    metrics = get_decision_accuracy_patterns(memories)

    assert metrics["total_evaluated"] == 1
    assert metrics["accuracy_rate"] is None
    assert metrics["incorrect_rate"] is None
    assert metrics["uncertain_rate"] == 1.0
    assert metrics["trend"] == "insufficient_data"


def test_trend_is_insufficient_with_too_few_evaluated_decisions():
    memories = [
        build_decision_memory(outcome_evaluation="correct", actual_outcome="A"),
        build_decision_memory(outcome_evaluation="incorrect", actual_outcome="B"),
        build_decision_memory(outcome_evaluation="correct", actual_outcome="C"),
    ]

    assert get_decision_accuracy_patterns(memories)["trend"] == "insufficient_data"


def test_trend_detects_improving_accuracy():
    memories = [
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="A",
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="B",
            outcome_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="C",
            outcome_timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="D",
            outcome_timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
        ),
    ]

    assert get_decision_accuracy_patterns(memories)["trend"] == "improving"


def test_trend_detects_declining_accuracy():
    memories = [
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="A",
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="B",
            outcome_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="C",
            outcome_timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="D",
            outcome_timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
        ),
    ]

    assert get_decision_accuracy_patterns(memories)["trend"] == "declining"


def test_trend_detects_stable_accuracy():
    memories = [
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="A",
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="B",
            outcome_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="C",
            outcome_timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="D",
            outcome_timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
        ),
    ]

    assert get_decision_accuracy_patterns(memories)["trend"] == "stable"


def test_repeated_incorrect_pattern_detects_shared_tag():
    memories = [
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="A",
            tags=["launch"],
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="B",
            tags=["launch"],
            outcome_timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    ]

    patterns = detect_decision_mismatch_patterns(memories)

    assert patterns == [
        {
            "pattern_key": "tag:launch",
            "incorrect_count": 2,
            "decision_ids": [str(memory.id) for memory in memories],
            "latest_outcome_timestamp": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "reason": "Repeated incorrect decisions share this grouping key",
        }
    ]


def test_repeated_incorrect_pattern_ignores_single_incorrect_decision():
    memories = [
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="A",
            tags=["launch"],
        ),
    ]

    assert detect_decision_mismatch_patterns(memories) == []


def test_repeated_incorrect_pattern_ignores_correct_decisions():
    memories = [
        build_decision_memory(
            outcome_evaluation="correct",
            actual_outcome="A",
            tags=["launch"],
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="B",
            tags=["launch"],
        ),
    ]

    assert detect_decision_mismatch_patterns(memories) == []


def test_repeated_incorrect_pattern_prefers_explicit_pattern_key():
    memories = [
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="A",
            tags=["launch"],
            decision_data_extra={"pattern_key": "context:low_clarity"},
        ),
        build_decision_memory(
            outcome_evaluation="incorrect",
            actual_outcome="B",
            tags=["launch"],
            decision_data_extra={"pattern_key": "context:low_clarity"},
        ),
    ]

    patterns = detect_decision_mismatch_patterns(memories)

    assert patterns[0]["pattern_key"] == "pattern_key:context:low_clarity"


def test_decision_evaluation_does_not_mutate_memory():
    memory = build_decision_memory(
        expected_outcome="Release safely and learn sooner.",
        actual_outcome="The release was delayed.",
        outcome_evaluation="incorrect",
        tags=["launch"],
    )
    original_decision_data = dict(memory.decision_data)
    original_tags = list(memory.tags)

    evaluate_decision_outcome(memory)
    get_decision_accuracy_patterns([memory])
    detect_decision_mismatch_patterns([memory])

    assert memory.decision_data == original_decision_data
    assert memory.decision_data["reasoning"] == "A smaller release is safer."
    assert memory.decision_data["expected_outcome"] == "Release safely and learn sooner."
    assert memory.tags == original_tags
