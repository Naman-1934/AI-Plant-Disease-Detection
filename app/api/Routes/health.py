"""
Health Check endpoints.

These endpoints are used to verify that the API is running.
"""

from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)

@router.get("/")
async def health_check() -> dict[str, str]:
    """
    Return the health status of the application.
    """

    return {
        "status": "healthy",
        "message": "Plant Disease Detection API is running."
    }