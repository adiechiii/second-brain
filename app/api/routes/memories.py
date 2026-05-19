"""Memory API routes."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.schemas.memories import (
    CompressionMemoryResponse,
    CreateMemoryRequest,
    CreateMemoryResponse,
    DailyCompressionRequest,
    DecisionOutcomeRequest,
    MemoryResponse,
    SearchMemoriesResponse,
    WeeklyCompressionRequest,
)
from app.core.config import get_settings
from app.infrastructure.session import get_db_session
from app.models.memory import Memory
from app.services.memory_compression import MemoryCompressionError
from app.services.memory_compression_service import MemoryCompressionService
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_ingestion import MemoryIngestionError
from app.services.memory_service import MemoryNotFoundError, MemoryOutcomeError, MemoryService

router = APIRouter(prefix="/memories", tags=["memories"])
logger = logging.getLogger(__name__)


def get_memory_service(
    session: Annotated[Session, Depends(get_db_session)],
) -> MemoryService:
    return MemoryService(MemoryRepository(session))


def get_memory_compression_service(
    session: Annotated[Session, Depends(get_db_session)],
) -> MemoryCompressionService:
    return MemoryCompressionService(MemoryRepository(session))


def to_memory_response(memory: Memory) -> MemoryResponse:
    return MemoryResponse(
        id=memory.id,
        raw_text=memory.raw_text,
        memory_type=memory.memory_type,
        decision_data=memory.decision_data,
        actual_outcome=memory.actual_outcome,
        outcome_timestamp=memory.outcome_timestamp,
        outcome_evaluation=memory.outcome_evaluation,
        clean_text=memory.clean_text,
        summary=memory.summary,
        tags=memory.tags,
        topic=memory.topic,
        importance_score=memory.importance_score,
        processing_state=memory.processing_state.value,
        record_state=memory.record_state.value,
    )


def to_compression_response(memory: Memory) -> CompressionMemoryResponse:
    return CompressionMemoryResponse(
        id=memory.id,
        summary=memory.summary or "",
        topic=memory.topic,
        tags=memory.tags,
    )


def handle_unexpected_memory_error(exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled memory endpoint error: %s: %s",
        exc.__class__.__name__,
        str(exc),
    )

    if get_settings().debug_errors:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )

    raise exc


@router.post("", response_model=CreateMemoryResponse, status_code=status.HTTP_201_CREATED)
def create_memory(
    request: CreateMemoryRequest,
    service: Annotated[MemoryService, Depends(get_memory_service)],
) -> CreateMemoryResponse:
    try:
        memory = service.create_memory(
            raw_text=request.raw_text,
            memory_type=request.memory_type,
            context=request.context,
            reasoning=request.reasoning,
            expected_outcome=request.expected_outcome,
        )
    except MemoryIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        return handle_unexpected_memory_error(exc)

    return CreateMemoryResponse(id=memory.id, summary=memory.summary)


@router.post("/compressions/daily", response_model=CompressionMemoryResponse)
def create_daily_compression(
    request: DailyCompressionRequest,
    service: Annotated[
        MemoryCompressionService,
        Depends(get_memory_compression_service),
    ],
) -> CompressionMemoryResponse:
    try:
        memory = service.create_daily_summary(request.day)
    except MemoryCompressionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        return handle_unexpected_memory_error(exc)

    return to_compression_response(memory)


@router.post("/compressions/weekly", response_model=CompressionMemoryResponse)
def create_weekly_compression(
    request: WeeklyCompressionRequest,
    service: Annotated[
        MemoryCompressionService,
        Depends(get_memory_compression_service),
    ],
) -> CompressionMemoryResponse:
    try:
        memory = service.create_weekly_summary(request.week_start)
    except MemoryCompressionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        return handle_unexpected_memory_error(exc)

    return to_compression_response(memory)


@router.get("/search", response_model=SearchMemoriesResponse)
def search_memories(
    query: Annotated[str, Query(min_length=1)],
    service: Annotated[MemoryService, Depends(get_memory_service)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SearchMemoriesResponse:
    try:
        memories = service.search_memories(query, limit)
        return SearchMemoriesResponse(
            results=[to_memory_response(memory) for memory in memories],
        )
    except Exception as exc:
        return handle_unexpected_memory_error(exc)


@router.patch("/{id}/outcome", response_model=MemoryResponse)
def update_decision_outcome(
    id: UUID,
    request: DecisionOutcomeRequest,
    service: Annotated[MemoryService, Depends(get_memory_service)],
) -> MemoryResponse:
    try:
        memory = service.update_decision_outcome(
            id,
            actual_outcome=request.actual_outcome,
            outcome_timestamp=request.outcome_timestamp,
            outcome_evaluation=request.outcome_evaluation,
        )
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryOutcomeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        return handle_unexpected_memory_error(exc)

    return to_memory_response(memory)


@router.get("/{id}", response_model=MemoryResponse)
def get_memory(
    id: UUID,
    service: Annotated[MemoryService, Depends(get_memory_service)],
) -> MemoryResponse:
    try:
        return to_memory_response(service.get_memory(id))
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
