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
MEMORY_TYPE_MEMORY = "memory"
MEMORY_TYPE_DECISION = "decision"
SUPPORTED_MEMORY_TYPES = {MEMORY_TYPE_MEMORY, MEMORY_TYPE_DECISION}


class MemoryIngestionError(ValueError):
    """Raised when memory ingestion input is invalid."""


def normalize_text(raw_text: str) -> str:
    return re.sub(r"\s+", " ", raw_text.strip())


def _require_text(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise MemoryIngestionError(f"{field_name} must not be empty")
    return normalize_text(value)


def _build_decision_raw_text(
    context: str,
    reasoning: str,
    expected_outcome: str,
) -> str:
    return (
        "Decision\n"
        f"Context: {context}\n"
        f"Reasoning: {reasoning}\n"
        f"Expected outcome: {expected_outcome}"
    )


class MemoryIngestionService:
    """Minimal ingestion pipeline for captured memories."""

    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def create_memory(
        self,
        raw_text: str | None,
        memory_type: str = MEMORY_TYPE_MEMORY,
        context: str | None = None,
        reasoning: str | None = None,
        expected_outcome: str | None = None,
    ) -> Memory:
        normalized_memory_type = normalize_text(memory_type or "").lower()
        if normalized_memory_type not in SUPPORTED_MEMORY_TYPES:
            raise MemoryIngestionError("memory_type is not supported")

        decision_data = None
        if normalized_memory_type == MEMORY_TYPE_DECISION:
            decision_context = _require_text(context, "context")
            decision_reasoning = _require_text(reasoning, "reasoning")
            decision_expected_outcome = _require_text(
                expected_outcome,
                "expected_outcome",
            )
            decision_data = {
                "context": decision_context,
                "reasoning": decision_reasoning,
                "expected_outcome": decision_expected_outcome,
            }
            if raw_text is None or not raw_text.strip():
                raw_text = _build_decision_raw_text(
                    decision_context,
                    decision_reasoning,
                    decision_expected_outcome,
                )

        if not raw_text or not raw_text.strip():
            raise MemoryIngestionError("raw_text must not be empty")
        if len(raw_text) > MAX_RAW_TEXT_LENGTH:
            raise MemoryIngestionError("raw_text exceeds maximum length")

        memory = Memory(
            raw_text=raw_text,
            memory_type=normalized_memory_type,
            decision_data=decision_data,
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
