"""
Image Service.

Coordinates image processing.
"""

from fastapi import UploadFile

from app.ml.preprocessing import ImagePreprocessor

class ImageService:
    """
    Service responsible for preparing uploaded images.
    """

    @staticmethod
    async def prepare_image(file: UploadFile):
        """
        Read and preprocess an uploaded image.
        """

        image_bytes = await file.read()

        image_tensor = ImagePreprocessor.preprocess(image_bytes)

        await file.seek(0)

        return image_tensor
    
    