"""Deterministic mock embedding service.

OpenAI embeddings are attempted first; deterministic mock embeddings remain as
the fallback for missing credentials, API failures, and tests.
"""

import hashlib
import re

from app.infrastructure.embeddings import (
    EmbeddingConfigurationError,
    OPENAI_EMBEDDING_DIMENSIONS,
    get_embedding,
)
from app.models.memory import Memory, ProcessingState

EMBEDDING_DIMENSIONS = OPENAI_EMBEDDING_DIMENSIONS


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def mock_embedding(clean_text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for word in _words(clean_text):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        bucket = digest[0] % EMBEDDING_DIMENSIONS
        vector[bucket] += 1.0

    total = sum(vector)
    if total == 0:
        return vector

    return [round(value / total, 6) for value in vector]


def embedding_for_text(text: str) -> list[float]:
    try:
        return get_embedding(text)
    except Exception:
        return mock_embedding(text)


def embed_memory(memory: Memory) -> Memory:
    memory.embedding = embedding_for_text(memory.clean_text or "")

    if memory.processing_state == ProcessingState.ENRICHED:
        memory.processing_state = ProcessingState.EMBEDDED

    return memory
