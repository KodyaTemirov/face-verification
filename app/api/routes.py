from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import time
import structlog
from pydantic import BaseModel
from typing import Dict, Any

from app.models.schemas import (
    FaceComparisonRequest,
    FaceComparisonResponse,
    ImageValidationRequest,
    ImageValidationResponse,
    HealthResponse,
    ErrorResponse,
    ProcessingStatus
)
from app.services import face_comparison_service, image_processor, anti_spoofing_service
from app.core.config import settings
from app.utils.helpers import validate_image_url

logger = structlog.get_logger(__name__)

router = APIRouter(prefix=settings.api_prefix, tags=["Face Verification"])


@router.post(
    "/compare-faces",
    response_model=FaceComparisonResponse,
    summary="Сравнение лиц",
    description="Сравнение двух лиц по URL изображений с анти-спуфингом"
)
async def compare_faces(request: FaceComparisonRequest) -> Dict[str, Any]:
    """Сравнение лиц между эталонным и тестовым изображениями"""
    start_time = time.time()
    
    try:
        # Валидация URL
        if not validate_image_url(request.reference_image_url):
            raise HTTPException(status_code=400, detail="Неверный URL эталонного изображения")
        
        if not validate_image_url(request.candidate_image_url):
            raise HTTPException(status_code=400, detail="Неверный URL изображения кандидата")
        
        logger.info(f"Запрос сравнения лиц: {request.reference_image_url} vs {request.candidate_image_url}")
        
        # Проверка инициализации моделей
        if not face_comparison_service.models_loaded:
            await face_comparison_service.initialize()
        
        # Выполнение сравнения
        result = await face_comparison_service.compare_faces(
            reference_url=request.reference_image_url,
            candidate_url=request.candidate_image_url,
            threshold=request.threshold
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка сравнения: {result['error']}"
            )
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Внутренняя ошибка сравнения: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@router.post(
    "/validate-image",
    response_model=ImageValidationResponse,
    summary="Валидация изображения",
    description="Проверка качества изображения и детекция лиц"
)
async def validate_image(request: ImageValidationRequest) -> Dict[str, Any]:
    """Валидация изображения и проверка на наличие лиц"""
    try:
        # Валидация URL
        if not validate_image_url(request.image_url):
            raise HTTPException(status_code=400, detail="Неверный URL изображения")
        
        start_time = time.time()
        
        logger.info(f"Запрос валидации изображения: {request.image_url}")
        
        # Загрузка изображения
        image = await image_processor.download_image(request.image_url)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Валидация качества
        is_valid, validation_details = image_processor.validate_image(image)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Изображение не прошло валидацию: {validation_details.get('error')}")
        
        # Детекция лиц
        if not face_comparison_service.models_loaded:
            await face_comparison_service.initialize()
            
        face_detection = await face_comparison_service.detect_faces(image)
        
        # Анти-спуфинг анализ если есть лица
        antispoofing_result = None
        if face_detection.get("has_faces", False):
            is_real, confidence, details = await anti_spoofing_service.detect_liveness(image)
            antispoofing_result = {
                "is_real": bool(is_real),
                "confidence": float(confidence),
                "details": details
            }
        
        processing_time = time.time() - start_time
        
        result = ImageValidationResponse(
            is_valid=bool(is_valid and face_detection.get("has_faces", False)),
            has_face=bool(face_detection.get("has_faces", False)),
            face_count=int(face_detection.get("face_count", 0)),
            image_quality=float(validation_details.get("quality_score", 0.0)),
            anti_spoofing=antispoofing_result,
            processing_time=float(processing_time),
            status=ProcessingStatus.SUCCESS
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Внутренняя ошибка валидации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    description="Статус работоспособности сервиса и загруженных моделей"
)
async def health_check() -> Dict[str, Any]:
    """Проверка состояния сервиса"""
    try:
        from datetime import datetime
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": face_comparison_service.models_loaded
        }
    except Exception as e:
        logger.error(f"Ошибка health check: {e}")
        raise HTTPException(status_code=500, detail="Сервис недоступен")


# Дополнительные утилитные эндпоинты

@router.get(
    "/models/status",
    summary="Статус моделей",
    description="Информация о загруженных моделях"
)
async def models_status():
    """Статус загруженных моделей"""
    return {
        "models_loaded": face_comparison_service.models_loaded,
        "model_name": settings.model_name,
        "detector_backend": settings.detector_backend,
        "distance_metric": settings.distance_metric
    }


@router.post(
    "/models/initialize",
    summary="Инициализация моделей",
    description="Принудительная загрузка моделей"
)
async def initialize_models():
    """Принудительная инициализация моделей"""
    try:
        if not face_comparison_service.models_loaded:
            await face_comparison_service.initialize()
        
        return {
            "status": "success",
            "message": "Модели успешно инициализированы",
            "models_loaded": face_comparison_service.models_loaded
        }
    except Exception as e:
        logger.error(f"Ошибка инициализации моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки моделей: {str(e)}")


@router.post("/test-image-download")
async def test_image_download(request: dict) -> Dict[str, Any]:
    """Тестовый эндпоинт для диагностики загрузки изображений"""
    try:
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL не указан")
        
        logger.info(f"Тестовая загрузка изображения: {url}")
        
        # Загрузка изображения
        image = await image_processor.download_image(url)
        
        if image is None:
            return {"success": False, "error": "Не удалось загрузить изображение"}
        
        # Базовая проверка
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # Валидация
        is_valid, details = image_processor.validate_image(image)
        
        return {
            "success": True,
            "image_loaded": True,
            "dimensions": {"width": width, "height": height, "channels": channels},
            "validation": {"is_valid": is_valid, "details": details}
        }
        
    except Exception as e:
        logger.error(f"Ошибка тестовой загрузки: {e}")
        return {"success": False, "error": str(e)} 