"""
Training configuration
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" /"raw"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
AUTOTUNE = "AUTOTUNE"

# Learning Rate = Controls how much the model updates its weights.
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.30
EPOCHS = 10