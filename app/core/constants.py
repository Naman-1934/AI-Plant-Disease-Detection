"""
Application-wide Constants.
"""

from pathlib import Path
from app.core.config import BASE_DIR, settings

MODEL_PATH = Path(BASE_DIR) / settings.model_dir / settings.model_name

LOG_DIRECTORY = Path(BASE_DIR) / settings.log_dir