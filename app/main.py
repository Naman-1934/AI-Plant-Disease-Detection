"""
Application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import api_router
from app.core.config import settings
from app.core.logger import app_logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    """

    app_logger.info("Starting Plant Disease Detection API...")

    yield

    app_logger.info("Shutting down Plant Disease Detection API...")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan
)

app.include_router(api_router)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoints.
    """

    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }

