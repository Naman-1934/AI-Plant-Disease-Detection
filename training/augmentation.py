"""
Data augmentation layers.

These layer are applied only to the training dataset.

Data augmentation is a technique used in machine learning to artificially expand the diversity and size of a dataset by creating modified 
copies of existing data. It helps models generalize better and prevents overfitting without the need to collect brand new real-world data.
"""

import tensorflow as tf

def get_data_augmentation() -> tf.keras.Sequential:
    """
    Create data augmentation pipeline.
    """

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
    ],
    name = "data_augmentation",

)