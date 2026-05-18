"""Memory ingestion service.

This module performs the minimal capture pipeline only: validate, normalize,
initialize, and persist. Enrichment, embeddings, and background work are out
of scope for this layer.
"""

from collections.abc import Generator
import re
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.infrastructure.session import get_db_session
from app.models.memory import Memory, ProcessingState, RecordState
from app.repositories.memory_repository import MemoryRepository

MAX_RAW_TEXT_LENGTH = 50_000


class MemoryIngestionError(ValueError):
    """Raised when memory ingestion input is invalid."""


def normalize_text(raw_text: str) -> str:
    return re.sub(r"\s+", " ", raw_text.strip())


class MemoryIngestionService:
    """Minimal ingestion pipeline for captured memories."""

    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def create_memory(self, raw_text: str) -> Memory:
        if not raw_text or not raw_text.strip():
            raise MemoryIngestionError("raw_text must not be empty")
        if len(raw_text) > MAX_RAW_TEXT_LENGTH:
            raise MemoryIngestionError("raw_text exceeds maximum length")

        memory = Memory(
            raw_text=raw_text,
            clean_text=normalize_text(raw_text),
            processing_state=ProcessingState.CAPTURED,
            record_state=RecordState.ACTIVE,
        )

        return self.repository.create(memory)


def get_memory_ingestion_service(
    session: Annotated[Session, Depends(get_db_session)],
) -> MemoryIngestionService:
    return MemoryIngestionService(MemoryRepository(session))


def create_memory(raw_text: str) -> Memory:
    session_context: Generator[Session, None, None] = get_db_session()
    session = next(session_context)
    try:
        service = MemoryIngestionService(MemoryRepository(session))
        return service.create_memory(raw_text)
    finally:
        session_context.close()
