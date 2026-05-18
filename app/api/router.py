"""Top-level API router registration."""

from fastapi import APIRouter

from app.api.routes import health, memories, reflections

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(memories.router)
api_router.include_router(reflections.router)
