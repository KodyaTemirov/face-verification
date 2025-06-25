from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Основные настройки
    app_name: str = "Face Verification Service"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # DeepFace настройки
    model_name: str = "ArcFace"
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    enforce_detection: bool = False
    
    # Пороговые значения
    default_threshold: float = 0.6
    anti_spoofing_threshold: float = 0.5
    image_quality_threshold: float = 0.05
    
    # Ограничения
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    max_image_dimension: int = 2048
    request_timeout: int = 30
    
    # Кэширование моделей
    model_cache_dir: str = "/app/models"
    
    # Логирование
    log_level: str = "INFO"
    log_format: str = "json"
    
    # API настройки
    api_prefix: str = "/api/v1"
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings() 