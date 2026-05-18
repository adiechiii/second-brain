"""Vector search service tests."""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects import postgresql

from app.models.memory import Memory, ProcessingState, RecordState
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_embedding import embedding_for_text, mock_embedding
from app.services.vector_search import VectorSearchService


def build_memory(clean_text: str, importance_score: float | None = None) -> Memory:
    memory = Memory(
        raw_text=clean_text,
        clean_text=clean_text,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
        importance_score=importance_score,
    )
    memory.embedding = mock_embedding(clean_text)
    return memory


def cosine_distance(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - (dot_product / (left_norm * right_norm))


class FakeVectorRepository:
    def __init__(self, memories):
        self.memories = memories
        self.received_embedding = None

    def search_by_embedding(self, query_embedding, limit):
        self.received_embedding = query_embedding
        return sorted(
            self.memories,
            key=lambda memory: cosine_distance(query_embedding, memory.embedding),
        )[:limit]


class UnavailableVectorRepository:
    def search_by_embedding(self, query_embedding, limit):
        raise SQLAlchemyError("database unavailable")


class RecordingSession:
    def __init__(self):
        self.statement = None

    def scalars(self, statement):
        self.statement = statement
        return []


def test_repository_search_by_embedding_uses_pgvector_cosine_operator():
    session = RecordingSession()
    repository = MemoryRepository(session)

    assert repository.search_by_embedding([0.1] * 1536, limit=5) == []

    compiled = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "<=>" in compiled
    assert "ORDER BY" in compiled
    assert "LIMIT" in compiled


def test_vector_search_returns_closest_matches():
    database_memory = build_memory("database schema planning")
    garden_memory = build_memory("garden watering soil")
    repository = FakeVectorRepository([garden_memory, database_memory])
    service = VectorSearchService(repository)

    results = service.vector_search("database schema", limit=1)

    assert results == [database_memory]
    assert repository.received_embedding == embedding_for_text("database schema")


def test_vector_search_falls_back_to_in_memory_search_when_db_unavailable():
    database_memory = build_memory("database schema planning", importance_score=0.2)
    garden_memory = build_memory("garden watering soil", importance_score=1.0)
    service = VectorSearchService(
        UnavailableVectorRepository(),
        fallback_memories=[garden_memory, database_memory],
    )

    results = service.vector_search("database schema", limit=10)

    assert results == [database_memory]


def test_vector_search_empty_query_returns_empty_list():
    memory = build_memory("database schema planning")
    service = VectorSearchService(FakeVectorRepository([memory]))

    assert service.vector_search("   ") == []
