"""
Prediction service used by CLI and future web API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from training.config import IMAGE_SIZE
from training.data_loader import DataLoader

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img


MODEL_PATH = Path("saved_models/plant_disease_model.keras")

class Predictor:
    """
    Service responsible for loading the model and predicting disease.
    """

    def __init__(self) -> None:
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.class_names = DataLoader.get_class_names()

    def preprocess(self, image_path: str) -> tf.Tensor:
        """
        Load and preprocess an image.
        """

        image = load_img(image_path, target_size = IMAGE_SIZE,)

        image = tf.cast(image, tf.float32)

        image = image / 255.0

        image = tf.expand_dims(image, axis=0)

        return image
    
    def predict(self, image_path:str) -> tuple[str, float]:
        """
        Predict the disease from an image.
        """

        image = self.preprocess(image_path)

        probabilities = self.model.predict(image, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))

        return (self.class_names[predicted_index], float(probabilities[predicted_index]),)
    
    