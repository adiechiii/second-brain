"""Vector-backed memory search service with deterministic fallback."""

from collections.abc import Generator
from datetime import datetime, timezone
import re

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.infrastructure.database import DatabaseConfigurationError
from app.infrastructure.session import get_db_session
from app.models.memory import Memory
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_embedding import embedding_for_text
from app.services.memory_search import search_memories

DEFAULT_SEARCH_LIMIT = 10


def _tokenize(text: object) -> set[str]:
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"[A-Za-z0-9']+", text.lower()))


def _safe_importance(memory: Memory) -> float:
    try:
        importance = float(memory.importance_score or 0.0)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(importance, 1.0))


def _safe_recency(memory: Memory, now: datetime) -> float:
    timestamp = memory.updated_at or memory.created_at
    if not isinstance(timestamp, datetime):
        return 0.0

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_seconds = max((now - timestamp).total_seconds(), 0.0)
    age_days = age_seconds / 86400.0
    return 1.0 / (1.0 + age_days / 30.0)


def _fraction_overlap(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / len(query_terms)


def _query_overlap_score(query_terms: set[str], memory: Memory) -> tuple[float, float, float]:
    text_terms = _tokenize(memory.clean_text) | _tokenize(memory.summary)

    tag_terms: set[str] = set()
    if isinstance(memory.tags, list):
        for tag in memory.tags:
            if isinstance(tag, str):
                tag_terms.update(_tokenize(tag))

    topic_terms = _tokenize(memory.topic)

    return (
        _fraction_overlap(query_terms, text_terms),
        _fraction_overlap(query_terms, tag_terms),
        _fraction_overlap(query_terms, topic_terms),
    )


def _rerank_memories(
    query: str,
    memories: list[Memory],
    limit: int,
    now: datetime | None = None,
) -> list[Memory]:
    if limit <= 0:
        return []
    if not query or not query.strip() or not memories:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    now = now or datetime.now(timezone.utc)
    scored = []
    for vector_rank, memory in enumerate(memories):
        text_overlap, tag_overlap, topic_overlap = _query_overlap_score(
            query_terms,
            memory,
        )
        vector_score = 1.0 / (1.0 + vector_rank)
        final_score = (
            vector_score * 1.00
            + text_overlap * 0.40
            + tag_overlap * 0.30
            + topic_overlap * 0.25
            + _safe_importance(memory) * 0.20
            + _safe_recency(memory, now) * 0.10
        )
        scored.append((final_score, vector_rank, memory))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [memory for _, _, memory in scored[:limit]]


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
