"""
Evaluate the trained plant disease classification model.
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from training.data_loader import DataLoader

import pandas as pd
import numpy as np

MODEL_PATH = Path("saved_models/plant_disease_model.keras")
OUTPUT_DIR = Path("evaluation")
OUTPUT_FILE = OUTPUT_DIR / "evaluation.json"


def load_model() -> tf.keras.Model:
    """
    Load the trained model
    """

    print("Loading trained model...")

    return tf.keras.models.load_model(MODEL_PATH)


def load_validation_dataset():
    """
    Load only the validation dataset.
    """

    print("Loading validation dataset...")

    _, validation_dataset, _ =  DataLoader.load_dataset()

    return validation_dataset

def evaluate_model(model: tf.keras.Model, validation_dataset) -> dict:
    """
    Evaluate the model.
    """

    loss, accuracy = model.evaluate(validation_dataset, verbose=1)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
    }


def save_results(results: dict) -> None:
    """
    Save evaluation metrics
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="UTF-8") as file:
        json.dump(results,file, indent=4)

    print(f"\nResults saved to {OUTPUT_FILE}")


def generate_predictions(model:tf.keras.Model, validation_dataset, ) -> pd.DataFrame:
    """
    Generate prediction for the validation dataset.
    """

    print("\nGenerating predictions...")

    true_labels=[]
    prediction_labels=[]
    confidence=[]

    for images, labels in validation_dataset:
        predictions = model.predict(images, verbose=0)

        # argmax() == Give  the index of the largest value.
        predicted_classes = tf.argmax(predictions, axis=1)

        actual_classes = tf.argmax(labels, axis=1)

        # Current batch [2,4,1,6],  After [2,4,1,6],  Second batch [5,7,1],  Now [2,4,1,6,5,7,1],  The list keeps growing.
        true_labels.extend(actual_classes.numpy())

        predicted_labels.extend(predicted_classes.numpy())

    return true_labels, predicted_labels


def main() -> None:
    """
    Evaluate the trained model.
    """

    model = load_model()

    validation_dataset = load_validation_dataset()

    results = evaluate_model(model, validation_dataset)
    print("\nEvaluation Results")
    
    print("-" * 30)

    print(f"Validation Accuracy : {results['accuracy']:.4f}")
    
    print(f"Validation Loss   : {results['loss']:.4f}")

    save_results(results)


if __name__ == "__main__":
    main()