"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from sqlalchemy.exc import SQLAlchemyError

from app.api.router import api_router
from app.core.config import get_settings
from app.infrastructure.database import (
    DatabaseConfigurationError,
    create_database_tables,
    verify_database_connection,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if settings.database_url:
            try:
                verify_database_connection()
                if settings.auto_create_tables:
                    create_database_tables()
            except (DatabaseConfigurationError, SQLAlchemyError) as exc:
                raise RuntimeError("Database startup verification failed") from exc

        yield

    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    application.include_router(api_router)

    # 🔥 FORCE OpenAPI servers field
    def custom_openapi():
        if application.openapi_schema:
            return application.openapi_schema

        openapi_schema = get_openapi(
            title=settings.app_name,
            version="0.1.0",
            routes=application.routes,
        )

        openapi_schema["servers"] = [
            {
                "url": "https://second-brain-vg8u.onrender.com",
                "description": "Production server",
            }
        ]

        application.openapi_schema = openapi_schema
        return application.openapi_schema

    application.openapi = custom_openapi

    return application


app = create_app()
