from datetime import datetime, timezone
from uuid import uuid4

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.intervention import assess_intervention_risk


def build_memory(
    *,
    raw_text: str = "Decision note",
    memory_type: str = "decision",
    decision_data_extra: dict | None = None,
    tags: list[str] | None = None,
    outcome_evaluation: str | None = None,
    outcome_timestamp: datetime | None = None,
) -> Memory:
    decision_data = None
    if memory_type == "decision":
        decision_data = {
            "context": raw_text,
            "reasoning": "A smaller release is safer.",
            "expected_outcome": "Release safely and learn sooner.",
        }
        if decision_data_extra:
            decision_data.update(decision_data_extra)

    return Memory(
        id=uuid4(),
        raw_text=raw_text,
        memory_type=memory_type,
        decision_data=decision_data,
        actual_outcome="Outcome recorded." if outcome_evaluation else None,
        outcome_timestamp=outcome_timestamp,
        outcome_evaluation=outcome_evaluation,
        clean_text=raw_text,
        summary=raw_text,
        tags=tags,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def test_no_warning_when_no_past_incorrect_pattern_exists():
    candidate = build_memory(tags=["launch"])

    warning = assess_intervention_risk(candidate, [])

    assert warning == {
        "warning": False,
        "risk_level": "none",
        "reason": "No repeated negative decision pattern matched",
        "reference_pattern": None,
    }


def test_warning_when_candidate_matches_repeated_incorrect_pattern():
    latest = datetime(2026, 1, 2, tzinfo=timezone.utc)
    past = [
        build_memory(
            tags=["launch"],
            outcome_evaluation="incorrect",
            outcome_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        build_memory(
            tags=["launch"],
            outcome_evaluation="incorrect",
            outcome_timestamp=latest,
        ),
    ]
    candidate = build_memory(tags=["launch"])

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is True
    assert warning["risk_level"] == "high"
    assert "2 past decision outcomes marked incorrect" in warning["reason"]
    assert warning["reference_pattern"] == {
        "pattern_key": "tag:launch",
        "incorrect_count": 2,
        "decision_ids": [str(memory.id) for memory in past],
        "latest_outcome_timestamp": latest.isoformat(),
    }


def test_no_warning_from_one_incorrect_decision():
    candidate = build_memory(tags=["launch"])
    past = [
        build_memory(tags=["launch"], outcome_evaluation="incorrect"),
    ]

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is False


def test_correct_and_uncertain_outcomes_do_not_count():
    candidate = build_memory(tags=["launch"])
    past = [
        build_memory(tags=["launch"], outcome_evaluation="correct"),
        build_memory(tags=["launch"], outcome_evaluation="uncertain"),
    ]

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is False


def test_explicit_pattern_key_beats_category_for_grouping():
    candidate = build_memory(
        decision_data_extra={
            "pattern_key": "context:low_clarity",
            "decision_category": "launch",
        },
        tags=["fallback"],
    )
    past = [
        build_memory(
            decision_data_extra={
                "pattern_key": "context:low_clarity",
                "decision_category": "other",
            },
            tags=["fallback"],
            outcome_evaluation="incorrect",
        ),
        build_memory(
            decision_data_extra={
                "pattern_key": "context:low_clarity",
                "decision_category": "other",
            },
            tags=["fallback"],
            outcome_evaluation="incorrect",
        ),
    ]

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is True
    assert warning["reference_pattern"]["pattern_key"] == "pattern_key:context:low_clarity"


def test_category_grouping_when_no_explicit_key_exists():
    candidate = build_memory(decision_data_extra={"decision_category": "launch"})
    past = [
        build_memory(
            decision_data_extra={"decision_category": "launch"},
            tags=["other"],
            outcome_evaluation="incorrect",
        ),
        build_memory(
            decision_data_extra={"decision_category": "launch"},
            tags=["other"],
            outcome_evaluation="incorrect",
        ),
    ]

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is True
    assert warning["reference_pattern"]["pattern_key"] == "decision_category:launch"


def test_tags_grouping_when_no_category_exists():
    candidate = build_memory(tags=["launch"])
    past = [
        build_memory(tags=["launch"], outcome_evaluation="incorrect"),
        build_memory(tags=["launch"], outcome_evaluation="incorrect"),
    ]

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is True
    assert warning["reference_pattern"]["pattern_key"] == "tag:launch"


def test_content_fallback_grouping_when_no_stronger_key_exists():
    candidate = build_memory(raw_text="Choose a small launch plan.", tags=None)
    past = [
        build_memory(
            raw_text="Choose a small launch plan.",
            tags=None,
            outcome_evaluation="incorrect",
        ),
        build_memory(
            raw_text="Choose a small launch plan.",
            tags=None,
            outcome_evaluation="incorrect",
        ),
    ]

    warning = assess_intervention_risk(candidate, past)

    assert warning["warning"] is True
    assert warning["reference_pattern"]["pattern_key"] == (
        "content:choose a small launch plan."
    )


def test_intervention_assessment_does_not_mutate_memory():
    candidate = build_memory(tags=["launch"])
    past = [
        build_memory(tags=["launch"], outcome_evaluation="incorrect"),
        build_memory(tags=["launch"], outcome_evaluation="incorrect"),
    ]
    original_candidate_data = dict(candidate.decision_data)
    original_past_data = [dict(memory.decision_data) for memory in past]

    assess_intervention_risk(candidate, past)

    assert candidate.decision_data == original_candidate_data
    assert [memory.decision_data for memory in past] == original_past_data
