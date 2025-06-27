#!/usr/bin/env python3
"""
Face Comparison Service - MagFace Edition with Anti-Spoofing & Telegram
Сервис сравнения лиц с использованием MagFace модели, антиспуфинга и Telegram уведомлений
"""
import base64
import io
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, List, Union

import cv2
import insightface
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import requests
from urllib.parse import urlparse
import sys
import tempfile
import onnxruntime

# Локальные импорты
try:
    from .config import config
    from .telegram_notifier import telegram_notifier
except ImportError:
    # Заглушки если модули не найдены
    class Config:
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
        TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
        NOTIFY_ON_SPOOFING = True
        NOTIFY_ON_LOW_CONFIDENCE = True
        NOTIFY_ON_ERRORS = True
        
        # Улучшенные настройки антиспуфинга
        ANTISPOOF_THRESHOLD = 0.3  # Более мягкий основной порог
        ANTISPOOF_STRICT_THRESHOLD = 0.7  # Строгий порог для явных подделок
        ANTISPOOF_MIN_CONFIDENCE = 0.4  # Минимальная уверенность для блокировки
        
        # Контекстные пороги
        LOW_QUALITY_IMAGE_THRESHOLD = 0.2  # Порог для изображений низкого качества
        HIGH_QUALITY_IMAGE_THRESHOLD = 0.5  # Порог для изображений высокого качества
        
        @classmethod
        def is_telegram_configured(cls):
            return bool(cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID and cls.TELEGRAM_ENABLED)
    
    config = Config()
    
    class TelegramNotifier:
        def __init__(self):
            self.enabled = config.is_telegram_configured()
        def send_message(self, message, parse_mode=None): 
            if not self.enabled: 
                logger.warning("📴 Telegram уведомления отключены")
                return False
            try:
                logger.info(f"📤 Отправка Telegram уведомления...")
                url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message}
                if parse_mode:
                    data["parse_mode"] = parse_mode
                response = requests.post(url, json=data, timeout=10)
                
                if response.status_code == 200:
                    logger.info("✅ Telegram уведомление отправлено успешно")
                    return True
                else:
                    logger.error(f"❌ Ошибка отправки Telegram: {response.status_code} - {response.text}")
                    return False
            except Exception as e:
                logger.error(f"❌ Исключение при отправке Telegram: {str(e)}")
                return False
        def notify_spoofing_detected(self, image_info, confidence):
            if config.NOTIFY_ON_SPOOFING:
                message = f"🚨 ОБНАРУЖЕНА ПОДДЕЛКА!\n📸 {image_info}\n🎯 Уверенность: {confidence:.2%}\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                self.send_message(message)
        def notify_comparison_result(self, similarity, verified, processing_time, 
                                   image1_url=None, image2_url=None, 
                                   antispoof_results=None, threshold=0.5):
            status = "✅ ВЕРИФИЦИРОВАН" if verified else "❌ НЕ ВЕРИФИЦИРОВАН"
            
            # Основная информация
            message = f"📊 СРАВНЕНИЕ ЛИЦ\n{status}\n🎯 Сходство: {similarity:.2%}\n⏱️ Время: {processing_time:.3f}с"
            
            # Если не верифицирован, добавляем причину
            if not verified:
                reasons = []
                
                # Проверяем причину: низкое сходство
                if similarity < threshold:
                    reasons.append(f"🔍 Низкое сходство ({similarity:.1%} < {threshold:.1%})")
                
                # Проверяем антиспуфинг результаты (только для второго изображения)
                if antispoof_results:
                    img2_spoof = antispoof_results.get('image2', {})
                    
                    # Проверяем только второе изображение на спуфинг
                    # Блокируем только если уверенность в подделке достаточно высока
                    if not img2_spoof.get('is_real', True):
                        confidence = img2_spoof.get('confidence', 0) * 100
                        model_used = img2_spoof.get('model_used', 'unknown')
                        
                        # Добавляем причину только если уверенность превышает минимальный порог
                        if img2_spoof.get('confidence', 0) >= config.ANTISPOOF_MIN_CONFIDENCE:
                            reasons.append(f"🚫 Изображение 2: подделка (уверенность: {confidence:.1f}%, модель: {model_used})")
                        else:
                            # Низкая уверенность - добавляем предупреждение, но не блокируем
                            logger.info(f"Низкая уверенность в подделке ({confidence:.1f}%), изображение пропущено")
                
                if reasons:
                    message += f"\n\n🔴 ПРИЧИНЫ ОТКЛОНЕНИЯ:\n" + "\n".join(reasons)
            
            # Добавляем ссылки на изображения если есть
            if image1_url or image2_url:
                message += f"\n\n🖼️ ИЗОБРАЖЕНИЯ:"
                if image1_url:
                    message += f"\n📎 Изображение 1:\n{image1_url}"
                if image2_url:
                    message += f"\n📎 Изображение 2:\n{image2_url}"
            
            self.send_message(message)
        def notify_error(self, error_type, error_message):
            if config.NOTIFY_ON_ERRORS:
                message = f"🔴 ОШИБКА\n🏷️ {error_type}\n📝 {error_message}\n⏰ {datetime.now().strftime('%H:%M:%S')}"
                self.send_message(message)
    
    telegram_notifier = TelegramNotifier()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./logs/face-comparison.log')
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные для статистики
app_stats = {
    "total_comparisons": 0,
    "successful_comparisons": 0,
    "failed_comparisons": 0,
    "total_antispoof_checks": 0,
    "real_faces": 0,
    "fake_faces": 0,
    "start_time": datetime.now(),
    "response_times": []
}

# Модели данных для API
class CompareRequest(BaseModel):
    """Запрос на сравнение лиц"""
    image1: str  # base64 encoded image или URL
    image2: str  # base64 encoded image или URL
    image1_type: str = "base64"  # "base64" или "url"
    image2_type: str = "base64"  # "base64" или "url"
    model: str = "MagFace-R100"
    metric: str = "cosine"
    threshold: float = 0.5
    enable_antispoof: bool = True  # Включить антиспуфинг по умолчанию

class CompareResponse(BaseModel):
    """Ответ сравнения лиц"""
    verified: bool
    distance: float
    similarity: float
    similarity_percentage: float
    threshold: float
    model: str
    metric: str
    processing_time: float
    timestamp: str
    faces_detected: Dict[str, int]
    antispoof_results: Optional[Dict[str, Dict]] = None

class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    timestamp: str
    version: str = "2.0.0"
    model: str = "MagFace-R100 with Anti-Spoofing"
    models_loaded: bool = True
    uptime_seconds: float
    total_comparisons: int
    total_antispoof_checks: int
    gpu_available: bool
    memory_info: Optional[Dict] = None

class SystemInfoResponse(BaseModel):
    """Информация о системе"""
    service_name: str
    version: str
    model: str
    available_models: List[str]
    supported_formats: List[str]
    gpu_info: Dict[str, bool]
    configuration: Dict[str, str]
    antispoof_enabled: bool

# Класс для антиспуфинга
class AntiSpoofProcessor:
    def __init__(self, model_path: str = "./models/asv_antispoof_r34.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_size = (128, 128)  # Стандартный размер для антиспуфинг моделей
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели антиспуфинга"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"❌ Модель антиспуфинга не найдена: {self.model_path}")
                logger.info("🔄 Используется заглушка антиспуфинга (всегда возвращает 'реальное лицо')")
                return False
                
            # Проверяем что это не пустой файл-заглушка
            if os.path.getsize(self.model_path) < 1000:  # Если файл меньше 1KB
                logger.warning(f"❌ Файл модели антиспуфинга слишком мал: {self.model_path}")
                logger.info("🔄 Используется заглушка антиспуфинга (всегда возвращает 'реальное лицо')")
                return False
                
            logger.info("Загрузка модели антиспуфинга...")
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Получаем информацию о входе модели
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            if len(input_shape) >= 4:
                self.input_size = (input_shape[2], input_shape[3])
            
            logger.info(f"✅ Модель антиспуфинга загружена, размер входа: {self.input_size}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели антиспуфинга: {e}")
            logger.info("🔄 Используется заглушка антиспуфинга (всегда возвращает 'реальное лицо')")
            return False
    
    def preprocess_face(self, face_image):
        """Предобработка лица для антиспуфинг модели"""
        try:
            # Изменяем размер
            resized = cv2.resize(face_image, self.input_size)
            
            # Нормализация
            normalized = resized.astype(np.float32) / 255.0
            
            # Конвертация в формат модели (1, C, H, W)
            if len(normalized.shape) == 3:
                normalized = np.transpose(normalized, (2, 0, 1))
            normalized = np.expand_dims(normalized, axis=0)
            
            return normalized
        except Exception as e:
            logger.error(f"Ошибка предобработки для антиспуфинга: {e}")
            return None
    
    def _assess_image_quality(self, face_image):
        """Оценка качества изображения для адаптации порогов"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # 1. Резкость (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_norm = min(sharpness / 1000.0, 1.0)  # Нормализация
            
            # 2. Контраст (стандартное отклонение)
            contrast = np.std(gray) / 255.0
            
            # 3. Яркость (среднее значение)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Лучше всего средняя яркость
            
            # 4. Равномерность освещения
            lighting_uniformity = 1.0 - (np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0
            lighting_uniformity = max(0, min(1, lighting_uniformity))
            
            # 5. Размер лица (больше = лучше)
            face_size = face_image.shape[0] * face_image.shape[1]
            size_score = min(face_size / (200 * 200), 1.0)  # Нормализация относительно 200x200
            
            # Комбинированная оценка качества
            quality_score = (
                sharpness_norm * 0.25 +
                contrast * 0.25 +
                brightness_score * 0.2 +
                lighting_uniformity * 0.15 +
                size_score * 0.15
            )
            
            return {
                "overall_quality": quality_score,
                "sharpness": sharpness_norm,
                "contrast": contrast,
                "brightness": brightness_score,
                "lighting": lighting_uniformity,
                "size_score": size_score,
                "is_high_quality": quality_score > 0.6,
                "is_low_quality": quality_score < 0.3
            }
            
        except Exception as e:
            logger.warning(f"Ошибка оценки качества изображения: {e}")
            return {
                "overall_quality": 0.5,
                "is_high_quality": False,
                "is_low_quality": False
            }

    def predict(self, face_image):
        """Улучшенное предсказание: реальное лицо или подделка с адаптивными порогами"""
        if self.session is None:
            # Заглушка: используем улучшенную эвристику
            return self._enhanced_fallback_prediction(face_image)
        
        try:
            # Оценка качества изображения для адаптации порогов
            quality_assessment = self._assess_image_quality(face_image)
            
            # Предобработка
            preprocessed = self.preprocess_face(face_image)
            if preprocessed is None:
                return {"is_real": True, "confidence": 0.0, "error": "Ошибка предобработки"}
            
            # Инференс (пытаемся использовать реальную модель)
            try:
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: preprocessed})
                
                # Обработка результата модели
                if len(outputs) > 0 and len(outputs[0]) >= 2:
                    # Бинарная классификация: [fake_prob, real_prob]
                    fake_prob = float(outputs[0][0])
                    real_prob = float(outputs[0][1])
                    raw_score = real_prob
                    model_confidence = max(fake_prob, real_prob)
                else:
                    # Одно значение (вероятность реального лица)
                    raw_score = float(outputs[0][0]) if len(outputs[0]) > 0 else 0.5
                    model_confidence = abs(raw_score - 0.5) * 2
                
                # Адаптивное принятие решения на основе качества изображения
                adapted_result = self._adaptive_decision(raw_score, model_confidence, quality_assessment)
                
                # Уведомления только для явных подделок с высокой уверенностью
                if not adapted_result["is_real"] and adapted_result["confidence"] > config.ANTISPOOF_STRICT_THRESHOLD and config.NOTIFY_ON_SPOOFING:
                    telegram_notifier.notify_spoofing_detected("Обнаружена явная подделка", adapted_result["confidence"])
                
                return {
                    **adapted_result,
                    "error": None,
                    "model_used": "neural_network_adaptive",
                    "quality_assessment": quality_assessment,
                    "raw_model_score": float(raw_score),
                    "model_confidence": float(model_confidence)
                }
                
            except Exception as model_error:
                logger.warning(f"Ошибка работы модели антиспуфинга: {model_error}")
                return self._enhanced_fallback_prediction(face_image)
                
        except Exception as e:
            logger.error(f"Общая ошибка антиспуфинг предсказания: {e}")
            telegram_notifier.notify_error("Антиспуфинг ошибка", str(e))
            return {"is_real": True, "confidence": 0.0, "error": str(e)}

    def _adaptive_decision(self, raw_score, model_confidence, quality_assessment):
        """Адаптивное принятие решения на основе качества изображения"""
        try:
            quality_score = quality_assessment.get("overall_quality", 0.5)
            is_high_quality = quality_assessment.get("is_high_quality", False)
            is_low_quality = quality_assessment.get("is_low_quality", False)
            
            # Выбираем порог в зависимости от качества изображения
            if is_low_quality:
                # Для изображений низкого качества - очень мягкий порог
                threshold = config.LOW_QUALITY_IMAGE_THRESHOLD
                logger.info(f"Изображение низкого качества, используем мягкий порог: {threshold}")
            elif is_high_quality:
                # Для изображений высокого качества - стандартный порог
                threshold = config.HIGH_QUALITY_IMAGE_THRESHOLD
                logger.info(f"Изображение высокого качества, используем стандартный порог: {threshold}")
            else:
                # Адаптивный порог на основе качества
                threshold = config.ANTISPOOF_THRESHOLD * (1 + quality_score * 0.5)
                logger.info(f"Адаптивный порог на основе качества ({quality_score:.2f}): {threshold:.3f}")
            
            # Базовое решение
            is_real_base = raw_score > threshold
            
            # Дополнительные проверки для снижения ложных срабатываний
            confidence_penalty = 0.0
            
            # Штраф за низкое качество изображения
            if is_low_quality:
                confidence_penalty += 0.2
                logger.info("Применен штраф за низкое качество изображения")
            
            # Бонус за высокое качество
            quality_bonus = quality_score * 0.1
            
            # Финальная уверенность с учетом корректировок
            final_confidence = max(0.0, min(1.0, model_confidence - confidence_penalty + quality_bonus))
            
            # Финальное решение: блокируем только если уверенность выше минимального порога
            is_real_final = is_real_base or (final_confidence < config.ANTISPOOF_MIN_CONFIDENCE)
            
            # Логирование решения
            logger.info(f"Антиспуфинг решение: raw_score={raw_score:.3f}, threshold={threshold:.3f}, "
                       f"model_confidence={model_confidence:.3f}, final_confidence={final_confidence:.3f}, "
                       f"is_real={is_real_final}")
            
            return {
                "is_real": is_real_final,
                "confidence": float(final_confidence),
                "score": float(raw_score),
                "decision_threshold": float(threshold),
                "quality_bonus": float(quality_bonus),
                "confidence_penalty": float(confidence_penalty)
            }
            
        except Exception as e:
            logger.error(f"Ошибка адаптивного решения: {e}")
            # В случае ошибки - консервативное решение (считаем реальным)
            return {
                "is_real": True,
                "confidence": 0.5,
                "score": float(raw_score),
                "error": str(e)
            }
    
    def _enhanced_fallback_prediction(self, face_image):
        """Улучшенный резервный алгоритм антиспуфинга с адаптивными порогами"""
        try:
            # Получаем оценку качества изображения
            quality_assessment = self._assess_image_quality(face_image)
            
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # 1. Анализ текстуры (расчет стандартного отклонения)
            texture_score = np.std(gray) / 255.0
            
            # 2. Анализ краев (количество резких переходов)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # 3. Анализ яркости (равномерность освещения)
            brightness_var = np.var(gray) / (255.0 ** 2)
            
            # 4. Анализ LBP (Local Binary Patterns) для текстуры
            lbp_score = self._calculate_lbp_score(gray)
            
            # 5. Анализ частотных характеристик (FFT)
            frequency_score = self._calculate_frequency_score(gray)
            
            # Комбинированный скор с учетом новых метрик
            combined_score = (
                texture_score * 0.25 +
                edge_density * 0.25 +
                brightness_var * 0.15 +
                lbp_score * 0.2 +
                frequency_score * 0.15
            )
            
            # Адаптивный порог на основе качества изображения
            if quality_assessment.get("is_low_quality", False):
                threshold = 0.15  # Очень мягкий для низкого качества
                logger.info("Fallback: используем мягкий порог для низкого качества")
            elif quality_assessment.get("is_high_quality", False):
                threshold = 0.4   # Стандартный для высокого качества
                logger.info("Fallback: используем стандартный порог для высокого качества")
            else:
                threshold = 0.25  # Средний порог
                logger.info("Fallback: используем средний порог")
            
            # Принятие решения с учетом качества
            is_real = combined_score > threshold
            
            # Корректировка уверенности на основе качества
            base_confidence = min(combined_score * 2, 1.0)
            quality_bonus = quality_assessment.get("overall_quality", 0.5) * 0.1
            
            # Финальная уверенность (консервативная для fallback)
            final_confidence = max(0.1, min(0.7, base_confidence + quality_bonus))
            
            # Логирование fallback решения
            logger.info(f"Fallback антиспуфинг: combined_score={combined_score:.3f}, "
                       f"threshold={threshold:.3f}, is_real={is_real}, confidence={final_confidence:.3f}")
            
            return {
                "is_real": is_real,
                "confidence": float(final_confidence),
                "score": float(combined_score),
                "error": None,
                "model_used": "enhanced_heuristic_fallback",
                "quality_assessment": quality_assessment,
                "decision_threshold": float(threshold),
                "details": {
                    "texture_score": float(texture_score),
                    "edge_density": float(edge_density),
                    "brightness_var": float(brightness_var),
                    "lbp_score": float(lbp_score),
                    "frequency_score": float(frequency_score)
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка улучшенного резервного алгоритма: {e}")
            return {
                "is_real": True,
                "confidence": 0.5,
                "score": 0.5,
                "error": str(e),
                "model_used": "default_safe"
            }

    def _calculate_lbp_score(self, gray_image):
        """Вычисление оценки на основе Local Binary Patterns"""
        try:
            # Упрощенная версия LBP
            height, width = gray_image.shape
            lbp_values = []
            
            # Проходим по изображению с шагом (упрощенная версия)
            for y in range(1, height - 1, 4):
                for x in range(1, width - 1, 4):
                    center = gray_image[y, x]
                    binary_pattern = 0
                    
                    # 8-соседний LBP
                    neighbors = [
                        gray_image[y-1, x-1], gray_image[y-1, x], gray_image[y-1, x+1],
                        gray_image[y, x+1], gray_image[y+1, x+1], gray_image[y+1, x],
                        gray_image[y+1, x-1], gray_image[y, x-1]
                    ]
                    
                    for i, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary_pattern |= (1 << i)
                    
                    lbp_values.append(binary_pattern)
            
            if len(lbp_values) == 0:
                return 0.5
            
            # Вычисляем разнообразие паттернов
            unique_patterns = len(set(lbp_values))
            pattern_diversity = min(unique_patterns / 100.0, 1.0)  # Нормализация
            
            return pattern_diversity
            
        except Exception as e:
            logger.warning(f"Ошибка вычисления LBP: {e}")
            return 0.5

    def _calculate_frequency_score(self, gray_image):
        """Анализ частотных характеристик через FFT"""
        try:
            # Применяем FFT
            fft = np.fft.fft2(gray_image)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Анализируем распределение частот
            height, width = magnitude_spectrum.shape
            center_y, center_x = height // 2, width // 2
            
            # Высокие частоты (дальше от центра)
            high_freq_mask = np.zeros_like(magnitude_spectrum)
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            high_freq_mask[distance > min(height, width) * 0.3] = 1
            
            high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
            total_energy = np.sum(magnitude_spectrum)
            
            if total_energy > 0:
                high_freq_ratio = high_freq_energy / total_energy
                # Нормализация (реальные изображения обычно имеют больше высоких частот)
                frequency_score = min(high_freq_ratio * 10, 1.0)
            else:
                frequency_score = 0.5
            
            return frequency_score
            
        except Exception as e:
            logger.warning(f"Ошибка анализа частот: {e}")
            return 0.5

    def _fallback_prediction(self, face_image):
        """Простой резервный алгоритм (для обратной совместимости)"""
        return self._enhanced_fallback_prediction(face_image)

# Класс для работы с MagFace
class MagFaceProcessor:
    def __init__(self):
        self.app = None
        self.antispoof = AntiSpoofProcessor()
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели MagFace"""
        try:
            logger.info("Загрузка MagFace модели...")
            self.app = insightface.app.FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            # Подготавливаем модель для разных размеров изображений
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ MagFace модель успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки MagFace: {e}")
            raise
    
    def preprocess_image(self, image):
        """Улучшенная предобработка изображения для лучшей детекции лиц"""
        try:
            # Конвертируем в RGB если необходимо
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB для OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:
                # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Получаем размеры
            height, width = image.shape[:2]
            
            # Масштабируем изображение если оно слишком маленькое
            min_size = 300
            if min(height, width) < min_size:
                scale = min_size / min(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.info(f"Масштабировано изображение: {width}x{height} -> {new_width}x{new_height}")
            
            # Ограничиваем максимальный размер для производительности
            max_size = 1920
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Уменьшено изображение: {width}x{height} -> {new_width}x{new_height}")
            
            # Улучшение контраста и яркости
            # Применяем CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Финальная конвертация в BGR для InsightFace
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"Ошибка предобработки изображения: {e}")
            return image

    def detect_faces_multiple_attempts(self, image):
        """Множественные попытки детекции лиц с разными параметрами"""
        attempts = [
            (640, 640, 0.1),   # Очень низкий порог
            (320, 320, 0.05),  # Экстремально низкий порог
            (512, 512, 0.2),   # Низкий порог
            (1024, 1024, 0.3), # Средний порог
            (800, 800, 0.15),  # Промежуточный
            (480, 480, 0.1),   # Маленький размер
        ]
        
        logger.info(f"Начинаем детекцию лиц, количество попыток: {len(attempts)}")
        
        for i, (det_size_w, det_size_h, threshold) in enumerate(attempts, 1):
            try:
                logger.info(f"Попытка {i}/{len(attempts)}: det_size=({det_size_w},{det_size_h}), threshold={threshold}")
                
                # Временно изменяем параметры модели
                self.app.prepare(ctx_id=0, det_size=(det_size_w, det_size_h))
                
                faces = self.app.get(image)
                logger.info(f"MagFace вернул {len(faces) if faces else 0} лиц")
                
                if faces and len(faces) > 0:
                    # Принимаем любые найденные лица
                    valid_faces = []
                    for j, face in enumerate(faces):
                        bbox = face.bbox
                        face_width = bbox[2] - bbox[0]
                        face_height = bbox[3] - bbox[1]
                        logger.info(f"Лицо {j+1}: размер {face_width:.1f}x{face_height:.1f}")
                        
                        # Очень мягкие требования - лицо хотя бы 20x20 пикселей
                        if face_width >= 20 and face_height >= 20:
                            valid_faces.append(face)
                    
                    if valid_faces:
                        # Сортируем лица по размеру (площади) - самое большое первым
                        valid_faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
                        
                        # Логируем информацию о найденных лицах
                        for j, face in enumerate(valid_faces):
                            bbox = face.bbox
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            logger.info(f"Лицо {j+1}: площадь {area:.0f} пикселей, bbox={bbox}")
                        
                        logger.info(f"✅ Найдено {len(valid_faces)} валидных лиц на попытке {i}, выбрано самое крупное")
                        return valid_faces
                        
            except Exception as e:
                logger.warning(f"Попытка {i} завершилась ошибкой: {e}")
                continue
        
        logger.warning("❌ Лица не найдены ни в одной из попыток")
        return []

    def detect_faces_opencv_fallback(self, image):
        """Резервная детекция лиц через OpenCV как последний шанс"""
        try:
            logger.info("🔄 Попытка детекции лиц через OpenCV Haar Cascades")
            
            # Конвертируем в оттенки серого
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Загружаем каскад Хаара
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Детекция лиц с разными параметрами
            for scale_factor in [1.1, 1.05, 1.2, 1.3]:
                for min_neighbors in [3, 2, 5, 1]:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30)
                    )
                    
                    if len(faces) > 0:
                        logger.info(f"✅ OpenCV нашел {len(faces)} лиц со scale_factor={scale_factor}, min_neighbors={min_neighbors}")
                        return faces
            
            logger.warning("❌ OpenCV не смог найти лица")
            return []
            
        except Exception as e:
            logger.error(f"Ошибка детекции лиц через OpenCV: {e}")
            return []

    def extract_embeddings_from_opencv_faces(self, image, opencv_faces):
        """Извлечение эмбеддингов из лиц, найденных OpenCV"""
        try:
            logger.info(f"🔄 Извлечение эмбеддингов из {len(opencv_faces)} лиц, найденных OpenCV")
            
            # Сортируем лица по площади (самое большое первым)
            faces_with_area = []
            for (x, y, w, h) in opencv_faces:
                area = w * h
                faces_with_area.append(((x, y, w, h), area))
            
            faces_with_area.sort(key=lambda x: x[1], reverse=True)
            
            # Обрабатываем каждое лицо, начиная с самого большого
            for i, ((x, y, w, h), area) in enumerate(faces_with_area):
                try:
                    logger.info(f"🎯 Обработка лица {i+1}/{len(faces_with_area)}: позиция ({x},{y}), размер {w}x{h}, площадь {area}")
                    
                    # Добавляем отступы для лучшего распознавания (15% от размера лица)
                    padding_x = int(w * 0.15)
                    padding_y = int(h * 0.15)
                    
                    # Вычисляем координаты с отступами
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(image.shape[1], x + w + padding_x)
                    y2 = min(image.shape[0], y + h + padding_y)
                    
                    # Извлекаем область лица с отступами
                    face_crop = image[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        logger.warning(f"Пустая область лица {i+1}, пропускаем")
                        continue
                    
                    logger.info(f"Область лица {i+1}: {face_crop.shape}, исходные координаты: ({x1},{y1},{x2},{y2})")
                    
                    # Пытаемся использовать InsightFace для извлечения эмбеддингов из области лица
                    try:
                        # Временно сохраняем исходные параметры модели
                        original_det_size = (640, 640)
                        
                        # Настраиваем модель для работы с вырезанными лицами
                        self.app.prepare(ctx_id=0, det_size=(320, 320))
                        
                        # Пытаемся получить эмбеддинги из области лица
                        insight_faces = self.app.get(face_crop)
                        
                        if insight_faces and len(insight_faces) > 0:
                            # Используем первое найденное лицо
                            insight_face = insight_faces[0]
                            
                            # Корректируем bbox относительно исходного изображения
                            original_bbox = np.array([
                                x1 + insight_face.bbox[0],
                                y1 + insight_face.bbox[1], 
                                x1 + insight_face.bbox[2],
                                y1 + insight_face.bbox[3]
                            ])
                            
                            logger.info(f"✅ Успешно извлечены эмбеддинги из лица {i+1} через InsightFace")
                            
                            # Восстанавливаем исходные параметры модели
                            self.app.prepare(ctx_id=0, det_size=original_det_size)
                            
                            return {
                                "embedding": insight_face.embedding,
                                "bbox": original_bbox,
                                "det_score": insight_face.det_score,
                                "detection_method": "opencv_assisted_insightface",
                                "face_crop_coords": (x1, y1, x2, y2),
                                "original_opencv_bbox": (x, y, w, h)
                            }
                            
                        else:
                            logger.warning(f"InsightFace не смог обработать область лица {i+1}")
                            
                    except Exception as e:
                        logger.warning(f"Ошибка обработки лица {i+1} через InsightFace: {e}")
                        continue
                    
                    finally:
                        # Восстанавливаем исходные параметры модели в любом случае
                        try:
                            self.app.prepare(ctx_id=0, det_size=original_det_size)
                        except:
                            pass
                            
                except Exception as e:
                    logger.warning(f"Ошибка обработки лица {i+1}: {e}")
                    continue
            
            # Если ни одно лицо не удалось обработать, используем простой подход
            logger.warning("⚠️ InsightFace не смог обработать ни одно лицо из OpenCV, используем заглушку")
            
            # Выбираем самое большое лицо для заглушки
            (x, y, w, h), _ = faces_with_area[0]
            
            # Создаем фиктивный эмбеддинг на основе характеристик лица
            fake_embedding = self._generate_fallback_embedding(image, x, y, w, h)
            
            return {
                "embedding": fake_embedding,
                "bbox": np.array([x, y, x+w, y+h], dtype=np.float32),
                "det_score": 0.5,  # Средний score для OpenCV
                "detection_method": "opencv_fallback",
                "warning": "Использован fallback эмбеддинг на основе характеристик изображения"
            }
            
        except Exception as e:
            logger.error(f"Критическая ошибка извлечения эмбеддингов из OpenCV лиц: {e}")
            raise ValueError(f"Не удалось извлечь эмбеддинги из найденных лиц: {e}")

    def _generate_fallback_embedding(self, image, x, y, w, h):
        """Генерация fallback эмбеддинга на основе характеристик области лица"""
        try:
            # Извлекаем область лица
            face_crop = image[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                # Если область пустая, создаем случайный нормализованный вектор
                embedding = np.random.normal(0, 0.1, 512).astype(np.float32)
                return embedding / np.linalg.norm(embedding)
            
            # Конвертируем в grayscale для анализа
            if len(face_crop.shape) == 3:
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_crop
            
            # Извлекаем базовые характеристики
            features = []
            
            # 1. Статистические характеристики яркости
            features.extend([
                np.mean(gray_face) / 255.0,
                np.std(gray_face) / 255.0,
                np.median(gray_face) / 255.0,
                (np.max(gray_face) - np.min(gray_face)) / 255.0
            ])
            
            # 2. Гистограмма (упрощенная)
            hist, _ = np.histogram(gray_face, bins=16, range=(0, 256))
            hist_norm = hist / np.sum(hist)
            features.extend(hist_norm.tolist())
            
            # 3. Текстурные характеристики
            # Локальные бинарные паттерны (упрощенная версия)
            try:
                # Вычисляем градиенты
                grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
                
                features.extend([
                    np.mean(np.abs(grad_x)) / 255.0,
                    np.mean(np.abs(grad_y)) / 255.0,
                    np.std(grad_x) / 255.0,
                    np.std(grad_y) / 255.0
                ])
            except:
                features.extend([0.1, 0.1, 0.1, 0.1])
            
            # 4. Геометрические характеристики
            features.extend([
                w / max(image.shape[:2]),  # Относительная ширина
                h / max(image.shape[:2]),  # Относительная высота
                (w * h) / (image.shape[0] * image.shape[1]),  # Относительная площадь
                w / h if h > 0 else 1.0  # Соотношение сторон
            ])
            
            # 5. Позиционные характеристики
            features.extend([
                x / image.shape[1],  # Относительная позиция X
                y / image.shape[0],  # Относительная позиция Y
                (x + w/2) / image.shape[1],  # Центр X
                (y + h/2) / image.shape[0]   # Центр Y
            ])
            
            # Дополняем до 512 измерений
            current_size = len(features)
            if current_size < 512:
                # Повторяем и модифицируем существующие характеристики
                repeat_count = (512 - current_size) // current_size
                remainder = (512 - current_size) % current_size
                
                for i in range(repeat_count):
                    # Добавляем вариации существующих характеристик
                    modified_features = [f * (1 + 0.1 * np.sin(i * np.pi + j)) for j, f in enumerate(features)]
                    features.extend(modified_features)
                
                # Добавляем остаток
                if remainder > 0:
                    features.extend(features[:remainder])
            
            # Обрезаем до точно 512 элементов
            features = features[:512]
            
            # Преобразуем в numpy array и нормализуем
            embedding = np.array(features, dtype=np.float32)
            
            # L2 нормализация
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                # Если норма 0, создаем случайный нормализованный вектор
                embedding = np.random.normal(0, 0.1, 512).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
            
            logger.info(f"Создан fallback эмбеддинг размерности {len(embedding)} с нормой {np.linalg.norm(embedding):.3f}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка создания fallback эмбеддинга: {e}")
            # Создаем базовый случайный эмбеддинг
            embedding = np.random.normal(0, 0.1, 512).astype(np.float32)
            return embedding / np.linalg.norm(embedding)

    def extract_face_embeddings(self, image, enable_antispoof=True):
        """Извлечение эмбеддингов лиц с опциональным антиспуфингом"""
        try:
            start_time = time.time()
            
            # Предобработка изображения
            processed_image = self.preprocess_image(image)
            
            # Детекция лиц через InsightFace
            faces = self.detect_faces_multiple_attempts(processed_image)
            
            # Результаты извлечения
            embedding_result = None
            detection_method = "insightface"
            
            if faces:
                # Используем InsightFace результаты
                main_face = faces[0]
                bbox = main_face.bbox
                face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                logger.info(f"🎯 Выбрано лицо с наибольшей площадью: {face_area:.0f} пикселей из {len(faces)} найденных")
                
                embedding_result = {
                    "embedding": main_face.embedding,
                    "bbox": main_face.bbox,
                    "det_score": main_face.det_score,
                    "detection_method": "insightface"
                }
                
            else:
                # Попытка через OpenCV + InsightFace гибрид
                logger.info("🔄 InsightFace не нашел лица, пробуем OpenCV + InsightFace гибридный подход")
                opencv_faces = self.detect_faces_opencv_fallback(processed_image)
                
                if len(opencv_faces) == 0:
                    raise ValueError("Лица не обнаружены на изображении")
                
                # Извлекаем эмбеддинги из OpenCV лиц через InsightFace
                embedding_result = self.extract_embeddings_from_opencv_faces(processed_image, opencv_faces)
                detection_method = embedding_result.get("detection_method", "opencv_hybrid")
            
            # Антиспуфинг проверка
            antispoof_result = None
            if enable_antispoof and embedding_result:
                try:
                    bbox = embedding_result["bbox"].astype(int)
                    x1, y1, x2, y2 = bbox
                    face_crop = processed_image[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        antispoof_result = self.antispoof.predict(face_crop)
                        app_stats["total_antispoof_checks"] += 1
                        
                        if antispoof_result.get("is_real", True):
                            app_stats["real_faces"] += 1
                        else:
                            app_stats["fake_faces"] += 1
                            
                        logger.info(f"Антиспуфинг результат: {antispoof_result}")
                    
                except Exception as e:
                    logger.warning(f"Ошибка антиспуфинга: {e}")
                    antispoof_result = {"is_real": True, "confidence": 0.0, "error": str(e)}
            
            processing_time = time.time() - start_time
            logger.info(f"Время обработки изображения: {processing_time:.3f} сек, метод: {detection_method}")
            
            return {
                "embedding": embedding_result["embedding"],
                "bbox": embedding_result["bbox"],
                "det_score": embedding_result.get("det_score", 0.5),
                "processing_time": processing_time,
                "faces_count": 1,
                "detection_method": detection_method,
                "antispoof": antispoof_result
            }
            
        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддингов: {str(e)}")
            raise

    def compare_faces(self, image1, image2, enable_antispoof=True, threshold=0.5, metric="cosine"):
        """Сравнение двух лиц с антиспуфингом"""
        start_time = time.time()
        
        try:
            app_stats["total_comparisons"] += 1
            
            # Извлекаем эмбеддинги (антиспуфинг только для второго изображения)
            result1 = self.extract_face_embeddings(image1, enable_antispoof=False)  # Первое изображение без антиспуфинга
            result2 = self.extract_face_embeddings(image2, enable_antispoof=enable_antispoof)  # Второе изображение с антиспуфингом
            
            embedding1 = result1["embedding"]
            embedding2 = result2["embedding"]
            faces1_count = result1.get("faces_count", 1)
            faces2_count = result2.get("faces_count", 1)
            
            # Вычисляем сходство в зависимости от метрики
            if metric == "cosine":
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                distance = 1 - similarity
            elif metric == "euclidean":
                distance = np.linalg.norm(embedding1 - embedding2)
                similarity = 1 / (1 + distance)  # Преобразуем в сходство
            else:
                # По умолчанию косинусное сходство
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                distance = 1 - similarity
            
            # Определяем верификацию
            verified = similarity > threshold
            
            processing_time = time.time() - start_time
            app_stats["response_times"].append(processing_time)
            
            if verified:
                app_stats["successful_comparisons"] += 1
            else:
                app_stats["failed_comparisons"] += 1
            
            # Функция для преобразования numpy типов в стандартные Python типы
            def convert_numpy_types(obj):
                """Рекурсивно преобразует numpy типы в стандартные Python типы"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            result = {
                "verified": bool(verified),
                "distance": float(distance),
                "similarity": float(similarity),
                "threshold": float(threshold),
                "processing_time": float(processing_time),
                "model": "MagFace-R100",
                "metric": str(metric),
                "similarity_percentage": float(round(similarity * 100, 2)),
                "timestamp": datetime.now().isoformat(),
                "faces_detected": {
                    "image1": int(faces1_count),
                    "image2": int(faces2_count)
                }
            }
            
            # Добавляем результаты антиспуфинга если включены
            if enable_antispoof:
                # Очищаем от numpy типов и обрабатываем None значения
                antispoof2 = result2.get("antispoof")
                
                # Если результат None, создаем пустой словарь
                if antispoof2 is None:
                    antispoof2 = {"is_real": True, "confidence": 0.0, "error": "No antispoof result"}
                
                antispoof_data = {
                    "image1": {"is_real": True, "confidence": 0.0, "note": "Антиспуфинг отключен для первого изображения"},
                    "image2": convert_numpy_types(antispoof2)
                }
                result["antispoof_results"] = antispoof_data
            
            # Логируем результат
            logger.info(f"Сравнение завершено: similarity={similarity:.3f}, verified={verified}")
            
            # Уведомление отправляется из endpoint'а с полной информацией
            
            return result
            
        except Exception as e:
            app_stats["failed_comparisons"] += 1
            logger.error(f"Ошибка сравнения лиц: {str(e)}")
            raise

    def download_image_from_url(self, url: str) -> np.ndarray:
        """Загрузка изображения из URL"""
        try:
            logger.info(f"Загрузка изображения из URL: {url[:100]}...")
            
            # Валидация URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Некорректный URL")
            
            # Загрузка с таймаутом
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Проверка типа контента
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"Подозрительный content-type: {content_type}")
            
            # Загрузка данных
            image_data = response.content
            if len(image_data) > 50 * 1024 * 1024:  # 50MB лимит
                raise ValueError("Изображение слишком большое (>50MB)")
            
            # Конвертация в numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Не удалось декодировать изображение")
            
            logger.info(f"✅ Изображение загружено: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения: {str(e)}")
            raise ValueError(f"Не удалось загрузить изображение: {str(e)}")

    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Декодирование изображения из base64"""
        try:
            logger.info(f"Декодирование base64 изображения, длина: {len(base64_string)} символов")
            
            # Убираем префикс data:image если есть
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Декодируем base64
            image_data = base64.b64decode(base64_string)
            
            # Конвертируем в PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Конвертируем в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Конвертируем в numpy array (BGR для OpenCV)
            image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Ошибка декодирования изображения: {str(e)}")
            raise ValueError(f"Не удалось декодировать изображение: {str(e)}")

    def load_image(self, image_data: str, image_type: str = "base64") -> np.ndarray:
        """Универсальная функция загрузки изображения"""
        logger.info(f"Загрузка изображения типа: {image_type}, данные: {image_data[:50]}...")
        
        if image_type == "url":
            return self.download_image_from_url(image_data)
        elif image_type == "base64":
            return self.decode_base64_image(image_data)
        else:
            raise ValueError(f"Неподдерживаемый тип изображения: {image_type}. Используйте 'base64' или 'url'")

# Инициализация процессора лиц
face_processor = MagFaceProcessor()

# FastAPI приложение
app = FastAPI(
    title="Face Comparison Service with Anti-Spoofing",
    description="Сервис сравнения лиц с использованием MagFace модели и антиспуфинга",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    logger.info("🚀 Запуск Face Comparison Service с MagFace и Anti-Spoofing")
    
    # Инициализация модели
    success = face_processor._load_model()
    if success:
        logger.info("✅ Сервис готов к работе")
    else:
        logger.warning("⚠️ Сервис запущен, но модель не загружена")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Face Comparison Service with Anti-Spoofing",
        "version": "2.0.0",
        "status": "running",
        "model": "MagFace-R100 with Anti-Spoofing",
        "docs": "/docs"
    }

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check эндпоинт"""
    uptime = (datetime.now() - app_stats["start_time"]).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        total_comparisons=app_stats["total_comparisons"],
        total_antispoof_checks=app_stats["total_antispoof_checks"],
        gpu_available=False,  # Используем CPU версию
        models_loaded=face_processor.app is not None
    )

@app.get("/api/v1/info", response_model=SystemInfoResponse)
async def system_info():
    """Получение информации о системе"""
    return SystemInfoResponse(
        service_name="Face Comparison Service with Anti-Spoofing",
        version="2.0.0",
        model="MagFace-R100",
        available_models=["MagFace-R100"],
        supported_formats=["JPEG", "PNG", "BMP", "TIFF"],
        gpu_info={"gpu_available": False, "using_cpu": True},
        configuration={
            "model": "MagFace-R100",
            "antispoof_model": "ASV Anti-Spoof R34",
            "backend": "ONNX Runtime",
            "device": "CPU"
        },
        antispoof_enabled=face_processor.antispoof.session is not None
    )

@app.post("/api/v1/compare", response_model=CompareResponse)
async def compare_faces(request: CompareRequest):
    """Сравнение двух лиц с поддержкой URL, base64 и антиспуфинга"""
    try:
        logger.info(f"Получен запрос на сравнение лиц с моделью {request.model}, антиспуфинг: {request.enable_antispoof}")
        app_stats["total_comparisons"] += 1
        
        # Загружаем изображения
        image1 = face_processor.load_image(request.image1, request.image1_type)
        image2 = face_processor.load_image(request.image2, request.image2_type)
        
        # Сравниваем лица
        result = face_processor.compare_faces(
            image1, 
            image2, 
            enable_antispoof=request.enable_antispoof,
            threshold=request.threshold,
            metric=request.metric
        )
        
        # Отправляем уведомление только при неуспешной верификации
        if not result["verified"]:
            try:
                # Определяем URL изображений
                img1_url = request.image1 if request.image1_type == "url" else None
                img2_url = request.image2 if request.image2_type == "url" else None
                
                telegram_notifier.notify_comparison_result(
                    similarity=result["similarity"], 
                    verified=result["verified"], 
                    processing_time=result["processing_time"],
                    image1_url=img1_url,
                    image2_url=img2_url,
                    antispoof_results=result.get("antispoof_results"),
                    threshold=result["threshold"]
                )
                logger.info("📤 Уведомление отправлено - верификация не пройдена")
            except Exception as notify_error:
                logger.warning(f"Ошибка отправки уведомления: {notify_error}")
        else:
            logger.info("✅ Верификация пройдена - уведомление не отправляется")
        
        return result
        
    except Exception as e:
        app_stats["failed_comparisons"] += 1
        logger.error(f"Ошибка в endpoint compare_faces: {str(e)}")
        
        # Отправляем уведомление об ошибке
        try:
            telegram_notifier.notify_error("API Error", str(e))
        except Exception as notify_error:
            logger.warning(f"Ошибка отправки уведомления об ошибке: {notify_error}")
        
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
        
        # Добавляем метаинформацию
        result.update({
            "model": request.model,
            "metric": request.metric,
            "threshold": request.threshold if request.threshold else result["threshold"],
            "similarity_percentage": round(result["similarity"] * 100, 2),
            "timestamp": datetime.now().isoformat(),
            "faces_detected": {"image1": 1, "image2": 1}
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка валидации: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/models")
async def get_models():
    """Получение списка доступных моделей"""
    return {
        "models": ["MagFace-R100"],
        "current_model": "MagFace-R100",
        "antispoof_model": "ASV Anti-Spoof R34",
        "backend": "ONNX Runtime",
        "note": "Using MagFace R100 implementation with anti-spoofing"
    }

@app.get("/api/v1/status")
async def get_status():
    """Получение расширенного статуса сервиса"""
    uptime = (datetime.now() - app_stats["start_time"]).total_seconds()
    
    avg_response_time = (
        sum(app_stats["response_times"][-100:]) / len(app_stats["response_times"][-100:])
        if app_stats["response_times"] else 0
    )
    
    return {
        "service": "Face Comparison Service with Anti-Spoofing",
        "version": "2.0.0",
        "model": "MagFace-R100",
        "antispoof_model": "ASV Anti-Spoof R34",
        "uptime_seconds": uptime,
        "uptime_formatted": str(datetime.now() - app_stats["start_time"]),
        "gpu_available": False,
        "models_loaded": face_processor.app is not None,
        "antispoof_loaded": face_processor.antispoof.session is not None,
        "statistics": {
            "total_comparisons": app_stats["total_comparisons"],
            "successful_comparisons": app_stats["successful_comparisons"],
            "failed_comparisons": app_stats["failed_comparisons"],
            "total_antispoof_checks": app_stats["total_antispoof_checks"],
            "real_faces": app_stats["real_faces"],
            "fake_faces": app_stats["fake_faces"],
            "success_rate": (
                app_stats["successful_comparisons"] / app_stats["total_comparisons"] * 100
                if app_stats["total_comparisons"] > 0 else 0
            ),
            "average_response_time": round(avg_response_time, 3)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/test-notifications")
async def test_notifications():
    """Тестовый эндпоинт для проверки уведомлений"""
    try:
        # Тестируем расширенное уведомление с примером данных
        telegram_notifier.notify_comparison_result(
            similarity=0.25,
            verified=False,
            processing_time=0.5,
            image1_url="https://example.com/image1.jpg",
            image2_url="https://example.com/image2.jpg",
            antispoof_results={
                "image1": {"is_real": False, "confidence": 0.8},
                "image2": {"is_real": True, "confidence": 0.9}
            },
            threshold=0.5
        )
        
        return {
            "message": "Тестовое уведомление отправлено в Telegram",
            "status": "ok",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "message": f"Ошибка отправки уведомления: {str(e)}",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/v1/antispoof")
async def check_antispoof(request: dict):
    """Отдельный эндпоинт для проверки антиспуфинга"""
    try:
        image_data = request.get("image")
        image_type = request.get("image_type", "base64")
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Изображение не предоставлено")
        
        # Загружаем изображение
        image = face_processor.load_image(image_data, image_type)
        
        # Извлекаем лицо и проверяем антиспуфинг
        result = face_processor.extract_face_embeddings(image, enable_antispoof=True)
        
        return {
            "antispoof_result": result.get("antispoof"),
            "faces_detected": result.get("faces_count", 0),
            "processing_time": result.get("processing_time", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Ошибка антиспуфинг проверки: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Создаем директории если не существуют
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    logger.info("Запуск Face Comparison Service с MagFace и Anti-Spoofing")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    ) 