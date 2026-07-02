"""
Central API Router.

All application routers are registered here.
"""

from fastapi import APIRouter
from app.api.Routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)