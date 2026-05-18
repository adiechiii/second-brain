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
