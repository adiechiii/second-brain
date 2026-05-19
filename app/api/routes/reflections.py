"""Reflection API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas.reflections import ReflectionRequest, ReflectionResponse
from app.infrastructure.session import get_db_session
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService
from app.services.reflection_service import ReflectionInputError, ReflectionService

router = APIRouter(prefix="/reflections", tags=["reflections"])


def get_reflection_service(
    session: Annotated[Session, Depends(get_db_session)],
) -> ReflectionService:
    memory_service = MemoryService(MemoryRepository(session))
    return ReflectionService(memory_service)


@router.post("", response_model=ReflectionResponse)
def create_reflection(
    request: ReflectionRequest,
    service: Annotated[ReflectionService, Depends(get_reflection_service)],
) -> ReflectionResponse:
    try:
        if request.memory_ids:
            reflection = service.generate_from_ids(request.memory_ids, depth=request.depth)
        else:
            reflection = service.generate_from_query(
                request.query or "",
                request.limit,
                depth=request.depth,
            )
    except ReflectionInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ReflectionResponse(**reflection)
