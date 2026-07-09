"""
Image preprocessing utilities

This module converts uploaded images into tensors (array but in tensorflow it's known as tensor) that are ready for Tensorflow inference.
"""

from io import BytesIO

import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Handles image preprocessing for inference.
    """

    IMAGE_SIZE = (224, 224)

    @classmethod
    def preprocess(cls, image_bytes: bytes) -> np.ndarray:
        """
        Convert raw image bytes into a normalized Numpy tensor.

        Parameters:
        -----------
        Image_bytes: bytes
            Raw uploaded image.

        Returns
        -------
        np.ndarray
            Tensor with shape:
            (1, 224, 224, 3)
        """

        # Rean an image.
        image = Image.open(BytesIO(image_bytes))

        # Converted into RGB
        image = image.convert("RGB")

        # Resize the image
        image = image.resize(cls.IMAGE_SIZE)

        # Convert the image into a numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize the an array
        image_array /= 255.0

        # Expand the dimesions to match the input shape with a model. Like add Batch Size and Channel = 3 (R, G, B)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array