"""Vector-backed memory search service with deterministic fallback."""

from collections.abc import Generator

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.infrastructure.database import DatabaseConfigurationError
from app.infrastructure.session import get_db_session
from app.models.memory import Memory
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_embedding import embedding_for_text
from app.services.memory_ranking import _rerank_memories
from app.services.memory_search import search_memories

DEFAULT_SEARCH_LIMIT = 10


class VectorSearchService:
    """Minimal pgvector search boundary."""

    def __init__(
        self,
        repository: MemoryRepository,
        fallback_memories: list[Memory] | None = None,
    ):
        self.repository = repository
        self.fallback_memories = fallback_memories or []

    def vector_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Memory]:
        if limit <= 0:
            return []
        if not query or not query.strip():
            return []

        query_embedding = embedding_for_text(query)
        try:
            candidate_limit = max(limit, limit * 3)
            candidates = self.repository.search_by_embedding(
                query_embedding,
                candidate_limit,
            )
            return _rerank_memories(query, candidates, limit)
        except (DatabaseConfigurationError, SQLAlchemyError):
            return search_memories(query, self.fallback_memories)[:limit]


def vector_search(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    fallback_memories: list[Memory] | None = None,
) -> list[Memory]:
    session_context: Generator[Session, None, None] = get_db_session()
    try:
        session = next(session_context)
    except DatabaseConfigurationError:
        return search_memories(query, fallback_memories or [])[:limit]

    try:
        service = VectorSearchService(MemoryRepository(session), fallback_memories)
        return service.vector_search(query, limit)
    finally:
        session_context.close()
