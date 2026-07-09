"""
Prediction Sevices

Coordinates prediction workflow.
"""

from app.ml.model_loader import ModelLoader

class PredictionServices:
    """
    Handles prediction requests.
    """

    @classmethod
    def model_ready() -> bool:
        """
        Check whether a model is available.
        """

        return ModelLoader.is_loaded()