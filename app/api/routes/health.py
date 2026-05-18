from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.api.schemas.common import HealthResponse
from app.infrastructure.database import (
    DatabaseConfigurationError,
    verify_database_connection,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@router.get("/health/db", response_model=HealthResponse)
def database_health_check():
    try:
        verify_database_connection()
    except (DatabaseConfigurationError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return {"status": "ok"}
