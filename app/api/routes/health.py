from fastapi import APIRouter

from app.api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@router.get("/health/db", response_model=HealthResponse)
def database_health_check():
    return {"status": "ok"}
