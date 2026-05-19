"""Vector search service tests."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects import postgresql

from app.models.memory import Memory, ProcessingState, RecordState
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_embedding import embedding_for_text, mock_embedding
from app.services.vector_search import VectorSearchService, _rerank_memories


def build_memory(clean_text: str, importance_score: float | None = None) -> Memory:
    memory = Memory(
        raw_text=clean_text,
        clean_text=clean_text,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
        importance_score=importance_score,
    )
    memory.embedding = mock_embedding(clean_text or "")
    return memory


def build_rank_memory(
    clean_text: str = "general note",
    summary: str | None = None,
    tags=None,
    topic: str | None = None,
    importance_score: float | None = None,
    updated_at=None,
    created_at=None,
) -> Memory:
    memory = build_memory(clean_text, importance_score)
    memory.summary = summary
    memory.tags = tags
    memory.topic = topic
    memory.updated_at = updated_at
    memory.created_at = created_at
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


class RecordingVectorRepository:
    def __init__(self, memories):
        self.memories = memories
        self.received_limit = None
        self.received_embedding = None

    def search_by_embedding(self, query_embedding, limit):
        self.received_embedding = query_embedding
        self.received_limit = limit
        return self.memories[:limit]


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


def test_vector_search_requests_expanded_candidates_and_returns_limit():
    memories = [build_memory(f"database memory {index}") for index in range(5)]
    repository = RecordingVectorRepository(memories)
    service = VectorSearchService(repository)

    results = service.vector_search("database", limit=2)

    assert repository.received_limit == 6
    assert len(results) == 2


def test_vector_search_limit_zero_returns_empty_without_repository_call():
    memory = build_memory("database memory")
    repository = RecordingVectorRepository([memory])
    service = VectorSearchService(repository)

    assert service.vector_search("database", limit=0) == []
    assert repository.received_limit is None
    assert repository.received_embedding is None


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


def test_tag_match_can_promote_lower_vector_candidate():
    anchor = build_rank_memory(tags=[])
    weak = build_rank_memory(tags=[])
    strong = build_rank_memory(tags=["database"])

    results = _rerank_memories("database", [anchor, weak, strong], limit=3)

    assert results.index(strong) < results.index(weak)


def test_topic_match_can_improve_ranking():
    anchor = build_rank_memory(topic=None)
    weak = build_rank_memory(topic=None)
    strong = build_rank_memory(topic="database")

    results = _rerank_memories("database", [anchor, weak, strong], limit=3)

    assert results.index(strong) < results.index(weak)


def test_higher_importance_improves_ranking_when_other_signals_equal():
    anchor = build_rank_memory(importance_score=0.0)
    low_importance = build_rank_memory(importance_score=0.0)
    high_importance = build_rank_memory(importance_score=1.0)

    results = _rerank_memories(
        "database",
        [anchor, low_importance, high_importance],
        limit=3,
    )

    assert results.index(high_importance) < results.index(low_importance)


def test_newer_updated_at_improves_ranking_when_other_signals_equal():
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    anchors = [build_rank_memory(updated_at=now - timedelta(days=120)) for _ in range(8)]
    old = build_rank_memory(updated_at=now - timedelta(days=120))
    new = build_rank_memory(updated_at=now - timedelta(days=1))

    results = _rerank_memories("database", anchors + [old, new], limit=10, now=now)

    assert results.index(new) < results.index(old)


def test_missing_optional_ranking_fields_do_not_crash():
    memory = build_rank_memory(
        clean_text=None,
        summary=None,
        tags="database",
        topic=None,
        importance_score=None,
        updated_at=None,
        created_at=None,
    )

    assert _rerank_memories("database", [memory], limit=1) == [memory]


def test_rerank_memories_limit_zero_returns_empty():
    memory = build_rank_memory(tags=["database"])

    assert _rerank_memories("database", [memory], limit=0) == []
