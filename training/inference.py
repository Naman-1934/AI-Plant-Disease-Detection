"""
Run inference on a single plant leaf image.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from training.config import IMAGE_SIZE
import argparse

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
        required=True,
        type=str,
        help="Path to the input image."
    )

    return parser.parse_args()


def load_image(image_path: str) -> tf.tensor:
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

def main() -> None:
    """
    Entry point
    """

    args = parse_arguments()

    model = load_model()

    image = load_img(args.image)

    # print(image.shape)

    print(model.summary())



if __name__ == "__main__":
    main()