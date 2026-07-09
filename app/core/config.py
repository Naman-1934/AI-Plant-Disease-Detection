"""
Application Configuration.

This module loads environment variables and exposes a single configuration object for the entire application.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """ Application settings loaded from environment variables."""

    app_name: str 
    app_version: str 

    debug: bool 

    host: str 
    port: int 

    model_dir: str 
    model_name: str 

    log_level: str 
    log_dir: str 

    mlflow_tracking_uri: str 
    mlflow_experiment_name: str 

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", 
        env_file_encoding="utf-8", 
        env_prefix="",
        case_sensitive=False,
        extra="ignore")
    
@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()

settings = get_settings()