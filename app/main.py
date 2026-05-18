"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        servers=[
            {
                "url": "https://second-brain-vg8u.onrender.com",
                "description": "Production server",
            }
        ],
    )

    application.include_router(api_router)

    return application


app = create_app()