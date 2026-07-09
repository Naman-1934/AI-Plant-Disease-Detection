"""
Prediction endpoints.

Currently this endpoint only validates uploaded files.
Prediction logic will be added in later phases.
"""

from fastapi import APIRouter, File, UploadFile

from app.schemas.prediction import PreprocessingResponse
from app.services.file_validator import FileValidator
from app.services.image_services import ImageService

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

@router.post(
    "/",
    response_model=PreprocessingResponse,
)

# File(...) tells FastAPI: Expect this value from a multipart/form-data request. Without it, FastAPI won't treat 
# the parameter as a file upload.
async def predict(file: UploadFile = File(...)):
    """
    Upload and Preprocess an image
    """

    await FileValidator.validate(file)

    image_tensor = await ImageService.prepare_image(file)
    return PreprocessingResponse(
        filename=file.filename,
        tensor_shape=list(image_tensor.shape),
        tensor_dtype=str(image_tensor.dtype),
        minimum_pixel_value=float(image_tensor.min()),
        maximum_pixel_value=float(image_tensor.max()),
    )