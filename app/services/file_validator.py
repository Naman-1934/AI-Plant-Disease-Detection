"""
File validation service.

This module validates uploaded image files before they enter the machine learning pipeline.
"""

from fastapi import HTTPException, UploadFile, status

class FileValidator:
    """
    Validate uploaded image files.
    """

    ALLOWED_CONTENT_TYPES = {
        "image/jpeg",
        "image/jpg",
        "image/png",
    }

    max_file_size = 5 * 1024 * 1024  # 5 MB
    
    @classmethod
    async def validate(cls, file:UploadFile) -> int:
        """
        Validate an uploaded image.
        
        Returns
        int
            size of the uploaded file in bytes.
        """

        if file.content_type not in cls.ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                details=(
                    "Unsupported file type. "
                    "Only JPEG, JPG and PNG images are allowed."
                )
            )
        
        # FastAPI reads the file.
        # Imagine the file is: ABCDEFGHIJ and The internal pointer starts here: ^ABCDEFGHIJ  (^ = Pointer)
        # After reading: ABCDEFGHIJ^. The pointer reaches the end. If Phase 5 tries to read the file again: await file.read() 
        # it gets: (empty).because the pointer is already at the end. So we reset it:

        contents = await file.read()
        file_size = len(contents)

        if file_size > cls.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File exceeds the maximum allowed size of 5 MB."
            )
        
        # Reset the pointer so later phases can read the file again.
        # If you don't reset the file pointer, later phases (image preprocessing and prediction) will receive an empty file.
        await file.seek(0)

        return file_size