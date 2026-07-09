"""
Response shemas.

This module contains Pydantic models used for API responses.
"""

from pydantic import BaseModel

class FileUploadResponse(BaseModel):
    """
    Response returned after successful file validation.
    """

    filename: str
    content_type: str
    size_in_bytes: int
    message: str