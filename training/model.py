"""
Transfer Learning model using EfficientNet80
"""

import tensorflow as tf
from training.config import IMAGE_SIZE, LEARNING_RATE, DROPOUT_RATE


class PlantDiseaseModel:
    """
    Creates the Plant Disease Classification Model.
    """

    @staticmethod
    def build(num_classes: int) -> tf.keras.Model:
        """
        Build and compile the model

        Args:
            num_classes:
                Number of disease classes

        Return:
            compiled keras model.
        """

        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3,
            ),
        )

        # Freeze pre-trained layers
        base_model.trainable = False

        inputs = tf.keras.Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3,
            ),
        )


        # EfficientNet Pre-processing
        # preprocess_input = It Converts image pixel values into the format EfficientNet expects and This ensures the model receives data in the 
        # same distribution as during ImageNet training.
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)

        x = base_model(
            x,
            training=False,
        )

        # Add Global Average Pooling
        # Instead of flattening millions of values, it computes one value per feature map.
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

        # Softmax will Converts logits into probabilities.
        outputs = tf.keras.layers.Dense(
            num_classes, 
            activation="softmax", 
            name="predictions",
            )(x)
        
        model = tf.keras.Model(
            inputs,
            outputs,
            name = "PlantDiseaseModel",
        )

        model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = LEARNING_RATE
            ),

            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                "accuracy",
            ],
        )

        return model