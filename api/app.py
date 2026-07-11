"""
FastAPI application for Plant Disease Detection.
"""

from fastapi import FastAPI

from pathlib import Path
import shutil

from fastapi import File

# UploadFile = This represents the image uploaded from the browser.
from fastapi import UploadFile

from training.predictor import Predictor


app = FastAPI(
    title="Plant Disease Detection API",
    description="API for predicting plant disease using a trained CNN model.",
    version="1.0.0",
)
predictor = Predictor()

# Decorator = In FastAPI, decorators are primarily used to map HTTP request methods and URL paths directly to Python functions,
# transforming regular code into web API endpoints.
@app.get("/")
def root() -> dict[str, str]:
    """
    Health check endpoint.
    """

    return{
        "message": "Plant Disease Detection API is running."
    }

@app.get("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    """
    Predict plant disease from an uploaded image.
    """

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    image_path = temp_dir / file.filename

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    disease, confidence = predictor.predict(str(image_path))

    image_path.unlink()

    return {
        "disease": disease,
        "confidence": round(confidence * 100, 2),
    }