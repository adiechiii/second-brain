"""Simple in-memory retrieval service.

This is intentionally not semantic search and does not issue SQL queries.
"""

from app.models.memory import Memory
from app.services.memory_ranking import (
    _has_query_overlap,
    _rerank_memories,
    _tokenize,
)


def search_memories(query: str, memories: list[Memory]) -> list[Memory]:
    if not query or not query.strip():
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    matching = [
        memory
        for memory in memories
        if _has_query_overlap(query_terms, memory)
    ]

    return _rerank_memories(
        query,
        matching,
        len(matching),
        base_order_weight=0.0,
    )
