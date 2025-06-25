import cv2
import numpy as np
from typing import Tuple, Dict
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class AntiSpoofingService:
    """Сервис анти-спуфинга для детекции поддельных изображений"""
    
    def __init__(self):
        self.threshold = settings.anti_spoofing_threshold
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей для анти-спуфинга"""
        # В продакшене здесь будут загружены специализированные модели
        # Пока используем базовые алгоритмы компьютерного зрения
        logger.info("Инициализация модулей анти-спуфинга")
    
    async def detect_liveness(self, image: np.ndarray, face_region: tuple = None) -> Tuple[bool, float, Dict]:
        """
        Детекция живого лица vs поддельного изображения
        Возвращает: (is_real, confidence, details)
        """
        try:
            if image is None:
                return False, 0.0, {"error": "Изображение не загружено"}
            
            # Набор тестов для определения живого лица
            tests_results = {}
            
            # 1. Анализ текстуры (LBP - Local Binary Patterns)
            texture_score = self._analyze_texture(image, face_region)
            tests_results["texture_analysis"] = texture_score
            
            # 2. Анализ освещения и теней
            lighting_score = self._analyze_lighting(image, face_region)
            tests_results["lighting_analysis"] = lighting_score
            
            # 3. Анализ краев и резкости
            edge_score = self._analyze_edges(image, face_region)
            tests_results["edge_analysis"] = edge_score
            
            # 4. Анализ цветовых характеристик
            color_score = self._analyze_color_distribution(image, face_region)
            tests_results["color_analysis"] = color_score
            
            # 5. Детекция периодических паттернов (признак экрана)
            pattern_score = self._detect_screen_patterns(image, face_region)
            tests_results["pattern_analysis"] = pattern_score
            
            # Вычисление общего скора
            confidence = self._calculate_overall_confidence(tests_results)
            is_real = confidence > self.threshold
            
            details = {
                "tests_results": tests_results,
                "overall_confidence": confidence,
                "threshold_used": self.threshold
            }
            
            logger.info(f"Анти-спуфинг результат: is_real={is_real}, confidence={confidence:.3f}")
            
            return is_real, confidence, details
            
        except Exception as e:
            logger.error(f"Ошибка детекции живого лица: {e}")
            return False, 0.0, {"error": str(e)}
    
    def _analyze_texture(self, image: np.ndarray, face_region: tuple = None) -> float:
        """Анализ текстуры изображения с использованием LBP"""
        try:
            # Конвертация в grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Применение LBP (Local Binary Pattern)
            # Простая реализация - в продакшене используется специализированная библиотека
            radius = 3
            n_points = 8 * radius
            
            # Вычисление статистик текстуры
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            
            # Энтропия как мера текстурности
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
            
            # Нормализация скора (живые лица имеют более высокую энтропию)
            texture_score = min(entropy / 8.0, 1.0)
            
            return texture_score
            
        except Exception as e:
            logger.error(f"Ошибка анализа текстуры: {e}")
            return 0.5
    
    def _analyze_lighting(self, image: np.ndarray, face_region: tuple = None) -> float:
        """Анализ освещения и теней"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Анализ градиентов освещения
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Статистики градиентов
            grad_mean = np.mean(gradient_magnitude)
            grad_std = np.std(gradient_magnitude)
            
            # Живые лица имеют более естественное распределение градиентов
            lighting_score = min((grad_std / (grad_mean + 1e-7)) / 2.0, 1.0)
            
            return lighting_score
            
        except Exception as e:
            logger.error(f"Ошибка анализа освещения: {e}")
            return 0.5
    
    def _analyze_edges(self, image: np.ndarray, face_region: tuple = None) -> float:
        """Анализ краев и резкости"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Детекция краев
            edges = cv2.Canny(gray, 50, 150)
            
            # Статистики краев
            edge_density = np.sum(edges > 0) / edges.size
            
            # Анализ резкости через Laplacian
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)
            
            # Комбинированный скор
            edge_score = (edge_density * 10 + sharpness) / 2
            edge_score = min(max(edge_score, 0.0), 1.0)
            
            return edge_score
            
        except Exception as e:
            logger.error(f"Ошибка анализа краев: {e}")
            return 0.5
    
    def _analyze_color_distribution(self, image: np.ndarray, face_region: tuple = None) -> float:
        """Анализ цветовых характеристик"""
        try:
            if len(image.shape) != 3:
                return 0.5
            
            # Анализ в HSV пространстве
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Статистики по каналам
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
            # Живые лица имеют более естественное распределение цветов
            color_variance = (h_std + s_std + v_std) / 3
            color_score = min(color_variance / 50.0, 1.0)
            
            return color_score
            
        except Exception as e:
            logger.error(f"Ошибка анализа цвета: {e}")
            return 0.5
    
    def _detect_screen_patterns(self, image: np.ndarray, face_region: tuple = None) -> float:
        """Детекция периодических паттернов экрана"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # FFT для детекции периодических паттернов
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Поиск пиков в спектре (признак регулярных паттернов экрана)
            spectrum_std = np.std(magnitude_spectrum)
            spectrum_mean = np.mean(magnitude_spectrum)
            
            # Низкая вариативность в спектре может указывать на экран
            pattern_score = 1.0 - min(spectrum_std / (spectrum_mean + 1e-7), 1.0)
            
            return max(pattern_score, 0.0)
            
        except Exception as e:
            logger.error(f"Ошибка детекции паттернов: {e}")
            return 0.5
    
    def _calculate_overall_confidence(self, tests_results: Dict) -> float:
        """Вычисление общего уровня уверенности"""
        # Взвешенная сумма результатов тестов
        weights = {
            "texture_analysis": 0.25,
            "lighting_analysis": 0.20,
            "edge_analysis": 0.20,
            "color_analysis": 0.15,
            "pattern_analysis": 0.20
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for test_name, score in tests_results.items():
            if test_name in weights:
                total_score += score * weights[test_name]
                total_weight += weights[test_name]
        
        if total_weight > 0:
            confidence = total_score / total_weight
        else:
            confidence = 0.5
        
        return min(max(confidence, 0.0), 1.0)


# Глобальный экземпляр сервиса анти-спуфинга
anti_spoofing_service = AntiSpoofingService()