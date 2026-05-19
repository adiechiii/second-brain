"""Memory ingestion service tests."""

import pytest

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_ingestion import (
    MAX_RAW_TEXT_LENGTH,
    MemoryIngestionError,
    MemoryIngestionService,
)


class FakeMemoryRepository:
    def __init__(self):
        self.created_memory = None

    def create(self, memory: Memory) -> Memory:
        self.created_memory = memory
        return memory


def test_valid_input_creates_memory():
    repository = FakeMemoryRepository()
    service = MemoryIngestionService(repository)

    memory = service.create_memory("  Keep\tthis\nmemory  ")

    assert repository.created_memory is memory
    assert memory.processing_state == ProcessingState.CAPTURED
    assert memory.record_state == RecordState.ACTIVE


def test_default_memory_type_is_memory():
    service = MemoryIngestionService(FakeMemoryRepository())

    memory = service.create_memory("Plain memory")

    assert memory.memory_type == "memory"
    assert memory.decision_data is None


def test_raw_text_is_preserved_exactly():
    repository = FakeMemoryRepository()
    service = MemoryIngestionService(repository)
    raw_text = "  Keep\tthis\nmemory  "

    memory = service.create_memory(raw_text)

    assert memory.raw_text == raw_text


def test_clean_text_is_normalized():
    repository = FakeMemoryRepository()
    service = MemoryIngestionService(repository)

    memory = service.create_memory("  Keep\tthis\nmemory  ")

    assert memory.clean_text == "Keep this memory"


@pytest.mark.parametrize("raw_text", ["", "   ", "\n\t"])
def test_empty_input_is_rejected(raw_text):
    service = MemoryIngestionService(FakeMemoryRepository())

    with pytest.raises(MemoryIngestionError):
        service.create_memory(raw_text)


def test_too_long_input_is_rejected():
    service = MemoryIngestionService(FakeMemoryRepository())

    with pytest.raises(MemoryIngestionError):
        service.create_memory("x" * (MAX_RAW_TEXT_LENGTH + 1))


def test_unsupported_memory_type_is_rejected():
    service = MemoryIngestionService(FakeMemoryRepository())

    with pytest.raises(MemoryIngestionError):
        service.create_memory("Test", memory_type="unsupported")


def test_decision_memory_requires_structured_fields():
    service = MemoryIngestionService(FakeMemoryRepository())

    with pytest.raises(MemoryIngestionError):
        service.create_memory(
            raw_text=None,
            memory_type="decision",
            context="I need to pick a launch date.",
            reasoning="",
            expected_outcome="Ship with fewer delays.",
        )


def test_decision_memory_stores_decision_data():
    service = MemoryIngestionService(FakeMemoryRepository())

    memory = service.create_memory(
        raw_text=None,
        memory_type="decision",
        context="I need to pick a launch date.",
        reasoning="A smaller release is safer.",
        expected_outcome="Ship earlier with fewer defects.",
    )

    assert memory.memory_type == "decision"
    assert memory.decision_data == {
        "context": "I need to pick a launch date.",
        "reasoning": "A smaller release is safer.",
        "expected_outcome": "Ship earlier with fewer defects.",
    }
    assert "Context: I need to pick a launch date." in memory.raw_text
    assert "Reasoning: A smaller release is safer." in memory.raw_text
    assert "Expected outcome: Ship earlier with fewer defects." in memory.raw_text
    assert memory.clean_text.startswith("Decision Context:")


def test_decision_memory_preserves_explicit_raw_text():
    service = MemoryIngestionService(FakeMemoryRepository())

    memory = service.create_memory(
        raw_text="Custom decision note",
        memory_type="decision",
        context="I need to pick a launch date.",
        reasoning="A smaller release is safer.",
        expected_outcome="Ship earlier with fewer defects.",
    )

    assert memory.raw_text == "Custom decision note"
    assert memory.clean_text == "Custom decision note"
    assert memory.decision_data["context"] == "I need to pick a launch date."
