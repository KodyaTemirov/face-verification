import asyncio
import time
from typing import Tuple, Dict, Optional
import structlog
import numpy as np
from deepface import DeepFace
import cv2

from app.core.config import settings
from app.services.image_processor import image_processor
from app.services.anti_spoofing import anti_spoofing_service
from app.models.schemas import AntiSpoofingResult

logger = structlog.get_logger(__name__)


class FaceComparisonService:
    """Сервис сравнения лиц"""
    
    def __init__(self):
        self.model_name = settings.model_name
        self.detector_backend = settings.detector_backend
        self.distance_metric = settings.distance_metric
        self.enforce_detection = settings.enforce_detection
        self._models_loaded = False
        
    async def initialize(self):
        """Инициализация и предзагрузка моделей"""
        if not self._models_loaded:
            try:
                logger.info("Загрузка моделей DeepFace...")
                # Предзагрузка модели для ускорения первого запроса
                await asyncio.to_thread(
                    DeepFace.build_model,
                    model_name=self.model_name
                )
                self._models_loaded = True
                logger.info("Модели DeepFace успешно загружены")
            except Exception as e:
                logger.error(f"Ошибка загрузки моделей: {e}")
                raise
    
    async def compare_faces(
        self,
        reference_url: str,
        candidate_url: str,
        threshold: float = None
    ) -> Dict:
        """
        Сравнение двух лиц по URL изображений
        """
        start_time = time.time()
        
        if threshold is None:
            threshold = settings.default_threshold
        
        try:
            # Загрузка изображений параллельно
            reference_task = image_processor.download_image(reference_url)
            candidate_task = image_processor.download_image(candidate_url)
            
            reference_image, candidate_image = await asyncio.gather(
                reference_task, candidate_task
            )
            
            logger.info(f"Результат загрузки: reference_image is None = {reference_image is None}, candidate_image is None = {candidate_image is None}")
            
            if reference_image is None:
                raise ValueError("Не удалось загрузить эталонное изображение")
            
            if candidate_image is None:
                raise ValueError("Не удалось загрузить изображение кандидата")
            
            # Валидация изображений
            logger.info("Начинаю валидацию изображений")
            ref_valid, ref_details = image_processor.validate_image(reference_image)
            cand_valid, cand_details = image_processor.validate_image(candidate_image)
            
            logger.info(f"Валидация эталонного: valid={ref_valid}, details={ref_details}")
            logger.info(f"Валидация кандидата: valid={cand_valid}, details={cand_details}")
            
            if not ref_valid:
                raise ValueError(f"Эталонное изображение невалидно: {ref_details.get('error')}")
            
            if not cand_valid:
                raise ValueError(f"Изображение кандидата невалидно: {cand_details.get('error')}")
            
            # Предобработка изображений
            reference_processed = image_processor.preprocess_for_face_detection(reference_image)
            candidate_processed = image_processor.preprocess_for_face_detection(candidate_image)
            
            # Сравнение лиц с использованием DeepFace
            comparison_result = await self._perform_face_comparison(
                reference_processed, candidate_processed, threshold
            )
            
            # Анти-спуфинг анализ параллельно
            ref_antispoofing_task = anti_spoofing_service.detect_liveness(reference_processed)
            cand_antispoofing_task = anti_spoofing_service.detect_liveness(candidate_processed)
            
            ref_antispoofing, cand_antispoofing = await asyncio.gather(
                ref_antispoofing_task, cand_antispoofing_task
            )
            
            # Формирование результата
            processing_time = time.time() - start_time
            
            result = {
                "is_same_person": bool(comparison_result["is_same_person"]),
                "confidence": float(comparison_result["confidence"]),
                "distance": float(comparison_result["distance"]),
                "anti_spoofing": AntiSpoofingResult(
                    reference_is_real=bool(ref_antispoofing[0]),
                    candidate_is_real=bool(cand_antispoofing[0]),
                    reference_confidence=float(ref_antispoofing[1]),
                    candidate_confidence=float(cand_antispoofing[1])
                ),
                "processing_time": float(processing_time),
                "status": "success"
            }
            
            logger.info(
                f"Сравнение завершено: same_person={result['is_same_person']}, "
                f"confidence={result['confidence']:.3f}, time={processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Ошибка сравнения лиц: {e}")
            
            return {
                "is_same_person": False,
                "confidence": 0.0,
                "distance": 1.0,
                "anti_spoofing": AntiSpoofingResult(
                    reference_is_real=False,
                    candidate_is_real=False,
                    reference_confidence=0.0,
                    candidate_confidence=0.0
                ),
                "processing_time": float(processing_time),
                "status": "error",
                "error": str(e)
            }
    
    async def _perform_face_comparison(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        threshold: float
    ) -> Dict:
        """Выполнение сравнения лиц через DeepFace"""
        try:
            # Сохранение временных файлов для DeepFace
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1, \
                 tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
                
                # Сохранение изображений
                cv2.imwrite(tmp1.name, img1)
                cv2.imwrite(tmp2.name, img2)
                
                try:
                    # Выполнение сравнения в отдельном потоке
                    result = await asyncio.to_thread(
                        DeepFace.verify,
                        img1_path=tmp1.name,
                        img2_path=tmp2.name,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        distance_metric=self.distance_metric,
                        enforce_detection=self.enforce_detection
                    )
                    
                    # Обработка результата
                    is_same_person = result['verified']
                    distance = result['distance']
                    
                    # Преобразование дистанции в confidence
                    if self.distance_metric == 'cosine':
                        confidence = 1.0 - distance
                    elif self.distance_metric == 'euclidean':
                        confidence = max(0.0, 1.0 - (distance / 2.0))
                    else:
                        confidence = 1.0 - min(distance, 1.0)
                    
                    confidence = max(0.0, min(confidence, 1.0))
                    
                    return {
                        "is_same_person": is_same_person,
                        "confidence": confidence,
                        "distance": distance
                    }
                    
                finally:
                    # Очистка временных файлов
                    try:
                        os.unlink(tmp1.name)
                        os.unlink(tmp2.name)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Ошибка DeepFace сравнения: {e}")
            raise ValueError(f"Ошибка сравнения лиц: {str(e)}")
    
    async def detect_faces(self, image: np.ndarray) -> Dict:
        """Детекция лиц на изображении"""
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                
                try:
                    # Детекция лиц
                    faces = await asyncio.to_thread(
                        DeepFace.extract_faces,
                        img_path=tmp.name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )
                    
                    face_count = len(faces) if faces else 0
                    
                    return {
                        "face_count": face_count,
                        "has_faces": face_count > 0,
                        "multiple_faces": face_count > 1
                    }
                    
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Ошибка детекции лиц: {e}")
            return {
                "face_count": 0,
                "has_faces": False,
                "multiple_faces": False,
                "error": str(e)
            }
    
    @property
    def models_loaded(self) -> bool:
        """Проверка загрузки моделей"""
        return self._models_loaded


# Глобальный экземпляр сервиса сравнения лиц
face_comparison_service = FaceComparisonService() 