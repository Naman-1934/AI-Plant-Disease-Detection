"""
Prediction  response schemas.
"""

from pydantic import BaseModel

class PreprocessingResponse(BaseModel):
    """
    Response returned after successful preprocessing.
    """

    filename: str
    tensor_shape: list[int]
    tensor_dtype: str
    minimum_pixel_value: float
    maximum_pixel_value: float