"""Simple in-memory retrieval service.

This is intentionally not semantic search and does not issue SQL queries.
"""

import re

from app.models.memory import Memory


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9']+", text.lower()))


def _similarity(query_words: set[str], memory: Memory) -> float:
    memory_words = _words(memory.clean_text or "")
    if not query_words or not memory_words:
        return 0.0
    return len(query_words & memory_words) / len(query_words)


def search_memories(query: str, memories: list[Memory]) -> list[Memory]:
    query_words = _words(query)
    if not query_words:
        return []

    scored = [
        (_similarity(query_words, memory), memory.importance_score or 0.0, memory)
        for memory in memories
    ]
    matching = [item for item in scored if item[0] > 0]
    matching.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [memory for _, _, memory in matching]
