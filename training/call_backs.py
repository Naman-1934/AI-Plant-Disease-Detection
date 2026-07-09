"""
Training callbacks.
"""

import tensorflow as tf

from training.config import MODEL_SAVE_PATH, PATIENCE, REDUCE_LR_PATIENCE, MIN_LEARNING_RATE


class CallbackFactory:
    """
    Factory class for creating training callbacks.
    """

    @staticmethod
    def get_callbacks() -> list[tf.keras.callbacks.Callback]:
        """
        Create training callbacks.
        """

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        )

        return [checkpoint, early_stopping, reduce_lr]