"""
Tensorflow dataset loader.
"""

import tensorflow as tf

from training.config import DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED

class DataLoader:
    """
    Creates Tensorflow datasets.
    """

    @staticmethod
    def load_dataset():
        """
        Creates training and validation datasets.
        """

        train_dataset = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=RANDOM_SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=RANDOM_SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        class_names = train_dataset.class_names

        autotune = tf.data.AUTOTUNE


        # Randomize training order. Only training data is shuffled.
        train_dataset = (
            train_dataset.cache().shuffle(1000).prefetch(autotune)
        )

        # Cache == Read images once, Store in memory and Faster every epoch.
        # Prefetch = CPU loads the next batch while GPU trains on the current one.
        validation_dataset = (
            validation_dataset.cache().prefetch(autotune)
        )

        return train_dataset, validation_dataset, class_names