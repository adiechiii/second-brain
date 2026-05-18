"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.router import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
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