from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_service import (
    MemoryNotFoundError,
    MemoryOutcomeError,
    MemoryService,
)


def build_decision_memory() -> Memory:
    return Memory(
        id=uuid4(),
        raw_text="Decision note",
        memory_type="decision",
        decision_data={
            "context": "I need to choose a release size.",
            "reasoning": "A smaller release is safer.",
            "expected_outcome": "Release safely and learn sooner.",
        },
        clean_text="Decision note",
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


class FakeMemoryRepository:
    def __init__(self, memory: Memory | None, memories: list[Memory] | None = None):
        self.memory = memory
        self.memories = memories or ([] if memory is None else [memory])
        self.updated_memory = None
        self.update_count = 0

    def get_by_id(self, id):
        if self.memory is not None and self.memory.id == id:
            return self.memory
        return None

    def get_by_ids(self, ids):
        return []

    def list_decisions_with_outcomes(self):
        return self.memories

    def update_decision_outcome(
        self,
        memory: Memory,
        actual_outcome: str,
        outcome_timestamp: datetime,
        outcome_evaluation: str | None,
    ) -> Memory:
        self.update_count += 1
        memory.actual_outcome = actual_outcome
        memory.outcome_timestamp = outcome_timestamp
        memory.outcome_evaluation = outcome_evaluation
        self.updated_memory = memory
        return memory


def test_update_decision_outcome_stores_outcome_fields():
    memory = build_decision_memory()
    timestamp = datetime(2026, 5, 19, 12, 30, tzinfo=timezone.utc)
    repository = FakeMemoryRepository(memory)
    service = MemoryService(repository)

    updated = service.update_decision_outcome(
        memory.id,
        actual_outcome="The small release shipped safely.",
        outcome_timestamp=timestamp,
        outcome_evaluation="correct",
    )

    assert updated.actual_outcome == "The small release shipped safely."
    assert updated.outcome_timestamp == timestamp
    assert updated.outcome_evaluation == "correct"
    assert repository.updated_memory is memory


def test_update_decision_outcome_defaults_timestamp_when_omitted():
    memory = build_decision_memory()
    repository = FakeMemoryRepository(memory)
    service = MemoryService(repository)

    updated = service.update_decision_outcome(
        memory.id,
        actual_outcome="The release shipped.",
    )

    assert updated.outcome_timestamp is not None
    assert updated.outcome_timestamp.tzinfo is not None


def test_update_decision_outcome_normalizes_naive_timestamp_to_utc():
    memory = build_decision_memory()
    timestamp = datetime(2026, 5, 19, 12, 30)
    repository = FakeMemoryRepository(memory)
    service = MemoryService(repository)

    updated = service.update_decision_outcome(
        memory.id,
        actual_outcome="The small release shipped safely.",
        outcome_timestamp=timestamp,
    )

    assert updated.outcome_timestamp == datetime(
        2026,
        5,
        19,
        12,
        30,
        tzinfo=timezone.utc,
    )


def test_update_decision_outcome_preserves_original_decision_data():
    memory = build_decision_memory()
    original_decision_data = dict(memory.decision_data)
    repository = FakeMemoryRepository(memory)
    service = MemoryService(repository)

    service.update_decision_outcome(
        memory.id,
        actual_outcome="The small release shipped safely.",
        outcome_evaluation="correct",
    )

    assert memory.decision_data == original_decision_data
    assert memory.decision_data["reasoning"] == "A smaller release is safer."
    assert (
        memory.decision_data["expected_outcome"]
        == "Release safely and learn sooner."
    )


@pytest.mark.parametrize("evaluation", ["correct", "incorrect", "uncertain"])
def test_valid_outcome_evaluations_pass(evaluation):
    memory = build_decision_memory()
    service = MemoryService(FakeMemoryRepository(memory))

    updated = service.update_decision_outcome(
        memory.id,
        actual_outcome="Outcome recorded.",
        outcome_evaluation=evaluation,
    )

    assert updated.outcome_evaluation == evaluation


def test_invalid_outcome_evaluation_is_rejected():
    memory = build_decision_memory()
    service = MemoryService(FakeMemoryRepository(memory))

    with pytest.raises(MemoryOutcomeError):
        service.update_decision_outcome(
            memory.id,
            actual_outcome="Outcome recorded.",
            outcome_evaluation="maybe",
        )


def test_outcome_update_requires_decision_memory():
    memory = build_decision_memory()
    memory.memory_type = "memory"
    service = MemoryService(FakeMemoryRepository(memory))

    with pytest.raises(MemoryOutcomeError):
        service.update_decision_outcome(
            memory.id,
            actual_outcome="Outcome recorded.",
        )


def test_missing_decision_raises_memory_not_found():
    service = MemoryService(FakeMemoryRepository(None))

    with pytest.raises(MemoryNotFoundError):
        service.update_decision_outcome(
            uuid4(),
            actual_outcome="Outcome recorded.",
        )


def test_outcome_update_does_not_create_duplicate_decision():
    memory = build_decision_memory()
    repository = FakeMemoryRepository(memory)
    service = MemoryService(repository)

    service.update_decision_outcome(
        memory.id,
        actual_outcome="The small release shipped safely.",
    )

    assert repository.update_count == 1
    assert repository.memory is memory


def test_get_decision_evaluation_summary_uses_repository_decisions():
    correct = build_decision_memory()
    correct.actual_outcome = "The release shipped."
    correct.outcome_evaluation = "correct"
    incorrect = build_decision_memory()
    incorrect.actual_outcome = "The release was delayed."
    incorrect.outcome_evaluation = "incorrect"
    incorrect.tags = ["launch"]
    repeated = build_decision_memory()
    repeated.actual_outcome = "The release was delayed again."
    repeated.outcome_evaluation = "incorrect"
    repeated.tags = ["launch"]
    repository = FakeMemoryRepository(None, [correct, incorrect, repeated])
    service = MemoryService(repository)

    summary = service.get_decision_evaluation_summary()

    assert summary["accuracy"]["total_evaluated"] == 3
    assert summary["accuracy"]["correct_count"] == 1
    assert summary["accuracy"]["incorrect_count"] == 2
    assert summary["repeated_incorrect_patterns"][0]["pattern_key"] == "tag:launch"


def test_get_intervention_warning_for_memory_uses_repository_history():
    candidate = build_decision_memory()
    candidate.tags = ["launch"]
    incorrect = build_decision_memory()
    incorrect.actual_outcome = "The release was delayed."
    incorrect.outcome_evaluation = "incorrect"
    incorrect.tags = ["launch"]
    repeated = build_decision_memory()
    repeated.actual_outcome = "The release was delayed again."
    repeated.outcome_evaluation = "incorrect"
    repeated.tags = ["launch"]
    repository = FakeMemoryRepository(candidate, [incorrect, repeated])
    service = MemoryService(repository)

    warning = service.get_intervention_warning_for_memory(candidate)

    assert warning["warning"] is True
    assert warning["reference_pattern"]["pattern_key"] == "tag:launch"
    assert repository.memory is candidate
