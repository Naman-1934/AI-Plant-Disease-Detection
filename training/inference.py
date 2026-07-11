"""
Run inference on a single plant leaf image.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from training.config import IMAGE_SIZE
from training.data_loader import DataLoader
import argparse

import numpy as np

MODEL_PATH = Path("saved_models/plant_disease_model.keras")


def load_model() ->tf.keras.Model:
    """
    Load the trained Tensorflow model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")

    print("Loading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Predict plant disease from an image")

    parser.add_argument(
        "--image",
        type=str,
        default="sample_images/test.jpg",
        help="Path to the input image.",
    )

    return parser.parse_args()


def load_image(image_path: str) -> tf.Tensor:
    """
    Load an image from disk and convert it into a Tensor.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"loading image: {image_path.name}")

    image = load_img(image_path, target_size=IMAGE_SIZE)

    image = img_to_array(image)

    return image


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """
    Prepare the image for model prediction.
    """

    # Convert pixel values from uint8 to float32
    image = tf.cast(image, tf.float32)

    # Normalize pixel value from [0, 255] to [0, 1]
    image = image / 255.0

    # Add the batch dimension
    image = tf.expand_dims(image, axis=0)

    return image

def predict(model: tf.keras.Model,  image: tf.Tensor, class_names: list[str],) -> tuple[str, float]:
    """
    Predict the disease class for a single image.
    """

    probabilities = model.predict(image, verbose=0,)

    predicted_index = np.argmax(probabilities[0])

    confidence = float(probabilities[0][predicted_index])

    predicted_class = class_names[predicted_index]
  

    return predicted_class, confidence


def main() -> None:
    """
    Entry point
    """

    args = parse_arguments()

    model = load_model()

    image = load_image(args.image)

    # print(image.shape)

    image = preprocess_image(image)

    class_names = DataLoader.get_class_names()

    predicted_class, confidence = predict(model=model, image=image, class_names=class_names)
    print("\nPrediction Result")
    print("-" * 40)
    print(f"Disease   : {predicted_class}")
    print(f"Confidence: {confidence:.2%}")



if __name__ == "__main__":
    main()