"""
Application Configuration.

This module loads environment variables and exposes a single configuration object for the entire application.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """ Application settings loaded from environment variables."""

    app_name: str = Field(alias="APP_NAME")
    app_version: str = Field(alias="APP_VERSION")

    debug: bool = Field(alias="DEBUG")

    host: str = Field(alias="HOST")
    port: str = Field(alias="PORT")

    model_dir: str = Field(alias="MODEL_DIR")
    model_name: str = Field(alias="MODEL_NAME")

    log_level: str = Field(alias="LOG_LEVEL")
    log_dir: str = Field(alias="LOG_DIR")

    mlflow_tracking_url: str = Field(alias="MLFLOW_TRACKING_URL")
    mlflow_experiment_name: str = Field(alias="MLFLOW_EXPERIMENT_NAME")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()