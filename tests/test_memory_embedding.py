"""Memory embedding service tests."""

from app.models.memory import Memory, ProcessingState, RecordState
from app.infrastructure.embeddings import get_embedding
from app.services.memory_embedding import (
    EMBEDDING_DIMENSIONS,
    embedding_for_text,
    embed_memory,
    mock_embedding,
)


def test_embedding_is_created():
    memory = Memory(
        raw_text="Database notes",
        clean_text="Database notes",
        processing_state=ProcessingState.ENRICHED,
        record_state=RecordState.ACTIVE,
    )

    embed_memory(memory)

    assert memory.embedding is not None
    assert len(memory.embedding) == EMBEDDING_DIMENSIONS
    assert any(value > 0 for value in memory.embedding)


def test_embedding_updates_processing_state():
    memory = Memory(
        raw_text="Database notes",
        clean_text="Database notes",
        processing_state=ProcessingState.ENRICHED,
        record_state=RecordState.ACTIVE,
    )

    embed_memory(memory)

    assert memory.processing_state == ProcessingState.EMBEDDED


def test_mock_embedding_is_deterministic():
    assert mock_embedding("Database notes") == mock_embedding("Database notes")


def test_embedding_uses_openai_client_when_configured(monkeypatch):
    expected_embedding = [0.5] * EMBEDDING_DIMENSIONS

    class FakeEmbedding:
        embedding = expected_embedding

    class FakeEmbeddingsResource:
        def create(self, model, input, encoding_format):
            assert model == "text-embedding-3-small"
            assert input == "Database notes"
            assert encoding_format == "float"
            return type("Response", (), {"data": [FakeEmbedding()]})()

    class FakeOpenAI:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.embeddings = FakeEmbeddingsResource()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.infrastructure.embeddings.OpenAI", FakeOpenAI)

    assert get_embedding("Database notes") == expected_embedding


def test_embedding_for_text_falls_back_when_key_missing():
    assert embedding_for_text("Database notes") == mock_embedding("Database notes")


def test_embed_memory_stores_real_embedding_when_available(monkeypatch):
    expected_embedding = [0.25] * EMBEDDING_DIMENSIONS
    monkeypatch.setattr(
        "app.services.memory_embedding.get_embedding",
        lambda text: expected_embedding,
    )
    memory = Memory(
        raw_text="Database notes",
        clean_text="Database notes",
        processing_state=ProcessingState.ENRICHED,
        record_state=RecordState.ACTIVE,
    )

    embed_memory(memory)

    assert memory.embedding == expected_embedding
    assert memory.processing_state == ProcessingState.EMBEDDED


def test_embed_memory_falls_back_when_openai_fails(monkeypatch):
    def fail_embedding(text):
        raise RuntimeError("api unavailable")

    monkeypatch.setattr("app.services.memory_embedding.get_embedding", fail_embedding)
    memory = Memory(
        raw_text="Database notes",
        clean_text="Database notes",
        processing_state=ProcessingState.ENRICHED,
        record_state=RecordState.ACTIVE,
    )

    embed_memory(memory)

    assert memory.embedding == mock_embedding("Database notes")
