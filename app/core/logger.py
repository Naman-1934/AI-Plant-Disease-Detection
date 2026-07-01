"""
Centralized application logger.
"""

from pathlib import Path
from loguru import logger

from app.core.constants import LOG_DIRECTORY
from app.core.config import settings

LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    LOG_DIRECTORY / "application.log",
    level=settings.log_level,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
)

logger.add(
    sink=lambda message: print(message, end=""),
    level=settings.log_level
)

app_logger = logger