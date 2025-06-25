import aiohttp
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class ImageProcessor:
    """Сервис обработки изображений"""
    
    def __init__(self):
        self.max_size = settings.max_image_size
        self.max_dimension = settings.max_image_dimension
        self.timeout = settings.request_timeout
    
    async def download_image(self, url: str) -> Optional[np.ndarray]:
        """Загрузка изображения по URL"""
        try:
            logger.info(f"Начинаю загрузку изображения: {url}")
            
            # Создаем connector с отключенной проверкой SSL для внешних URL
            connector = aiohttp.TCPConnector(ssl=False)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {
                    'User-Agent': 'Face-Verification-Service/1.0',
                    'Accept': 'image/jpeg,image/png,image/webp,image/*'
                }
                
                async with session.get(
                    url, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=headers
                ) as response:
                    logger.info(f"HTTP статус: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"Ошибка загрузки изображения: HTTP {response.status}, URL: {url}")
                        return None
                    
                    # Проверка content-type
                    content_type = response.headers.get('content-type', '')
                    logger.info(f"Content-Type: {content_type}")
                    
                    if not content_type.startswith('image/'):
                        logger.warning(f"Неожиданный content-type: {content_type}")
                    
                    # Проверка размера
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_size:
                        logger.error(f"Изображение слишком большое: {content_length} bytes")
                        return None
                    
                    image_data = await response.read()
                    logger.info(f"Загружено {len(image_data)} байт")
                    
                    # Проверка фактического размера
                    if len(image_data) > self.max_size:
                        logger.error(f"Изображение слишком большое: {len(image_data)} bytes")
                        return None
                    
                    if len(image_data) == 0:
                        logger.error("Получен пустой файл")
                        return None
                    
                    return self._bytes_to_image(image_data)
                    
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка сети при загрузке изображения {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке изображения {url}: {e}")
            return None
    
    def _bytes_to_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Конвертация байтов в изображение OpenCV"""
        try:
            logger.info(f"Начинаю конвертацию {len(image_bytes)} байт в изображение")
            
            if len(image_bytes) == 0:
                logger.error("Получены пустые байты")
                return None
                
            # Проверка магических байтов для JPEG/PNG
            if not (image_bytes.startswith(b'\xff\xd8') or image_bytes.startswith(b'\x89PNG')):
                logger.warning("Неожиданный формат изображения, пытаемся продолжить")
            
            # Попытка открыть через PIL для лучшей совместимости
            pil_image = Image.open(BytesIO(image_bytes))
            logger.info(f"PIL изображение открыто: {pil_image.size}, mode={pil_image.mode}")
            
            # Конвертация в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                logger.info("Изображение конвертировано в RGB")
            
            # Проверка размеров
            width, height = pil_image.size
            logger.info(f"Размеры изображения: {width}x{height}")
            
            if width > self.max_dimension or height > self.max_dimension:
                logger.warning(f"Изображение большое: {width}x{height}, уменьшаем до {self.max_dimension}")
                # Уменьшаем вместо отклонения
                ratio = self.max_dimension / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Изображение уменьшено до: {new_width}x{new_height}")
            
            # Конвертация в OpenCV формат
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            logger.info(f"OpenCV изображение создано: shape={opencv_image.shape}")
            
            return opencv_image
            
        except Exception as e:
            logger.error(f"Ошибка конвертации изображения: {e}", exc_info=True)
            return None
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, dict]:
        """Валидация качества изображения"""
        try:
            logger.info("Начинаю валидацию изображения")
            
            if image is None:
                logger.error("Изображение None при валидации")
                return False, {"error": "Изображение не загружено"}
            
            height, width = image.shape[:2]
            logger.info(f"Размеры для валидации: {width}x{height}")
            
            # Проверка минимальных размеров
            if width < 64 or height < 64:
                logger.error(f"Изображение слишком маленькое: {width}x{height}")
                return False, {"error": "Изображение слишком маленькое"}
            
            # Оценка качества (контраст, резкость)
            quality_score = self._calculate_image_quality(image)
            logger.info(f"Качество изображения: {quality_score}")
            
            validation_result = {
                "width": width,
                "height": height,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "quality_score": quality_score
            }
            
            # Используем настраиваемый порог качества
            from app.core.config import settings
            threshold = settings.image_quality_threshold
            is_valid = quality_score > threshold
            logger.info(f"Валидация результат: is_valid={is_valid}, threshold={threshold}")
            
            return is_valid, validation_result
            
        except Exception as e:
            logger.error(f"Ошибка валидации изображения: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Расчет качества изображения"""
        try:
            # Конвертация в grayscale для анализа
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Расчет резкости (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # Расчет контраста
            contrast_score = gray.std() / 255.0
            
            # Проверка на размытие
            blur_score = 1.0 - min(cv2.blur(gray, (5, 5)).var() / gray.var(), 1.0) if gray.var() > 0 else 0.0
            
            # Общая оценка качества
            quality_score = (sharpness_score * 0.4 + contrast_score * 0.4 + blur_score * 0.2)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Ошибка расчета качества изображения: {e}")
            return 0.0
    
    def preprocess_for_face_detection(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для детекции лиц"""
        try:
            # Нормализация размера если изображение слишком большое
            height, width = image.shape[:2]
            max_size = 800
            
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Улучшение контраста
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"Ошибка предобработки изображения: {e}")
            return image


# Глобальный экземпляр процессора изображений
image_processor = ImageProcessor() 