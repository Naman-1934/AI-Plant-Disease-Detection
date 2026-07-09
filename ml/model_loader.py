"""
Model Loader.

Responsible for loading and providing access to the machine learning model.
"""

from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import app_logger


class ModelLoader:
    """
    Singleton-like model manager.
    """

    _model: Any | None = None

    @classmethod
    def load_model(cls) -> None:
        """
        Load the trained model if it exists.

        Tensorflow loading will be added after the model is trained.
        """

        model_path = (Path(settings.model_dir) / settings.model_name)

        if not model_path.exists():
            app_logger.warning("Model File not Found: %s", model_path)

            app_logger.warning("Application will continue without a loaded model.")

            return 
        
        # Tensorflow loading will be added later.
        cls._model = "MODEL_PLACEHOLDER"

        app_logger.info("Model Loaded Successfully.")

    
    @classmethod
    def get_model(cls) -> Any | None:
        """
        Return the loaded model.
        """

        return cls._model
    
    @classmethod
    def is_loaded(cls) -> bool:
        """
        Check whether a model is loaded.
        """

        return cls._model is not None