#!/usr/bin/env python3
"""
Telegram уведомления для Face Comparison Service
"""
import requests
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from .config import config

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Класс для отправки уведомлений в Telegram"""
    
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = config.TELEGRAM_ENABLED and config.is_telegram_configured()
        
        if self.enabled:
            logger.info("✅ Telegram уведомления активированы")
        else:
            logger.warning("⚠️ Telegram уведомления не настроены")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Отправка сообщения в Telegram"""
        if not self.enabled:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                logger.debug("📤 Telegram сообщение отправлено")
                return True
            else:
                logger.error(f"❌ Ошибка отправки в Telegram: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Исключение при отправке в Telegram: {e}")
            return False
    
    def notify_spoofing_detected(self, image_info: str, confidence: float):
        """Уведомление о обнаружении подделки"""
        if not config.NOTIFY_ON_SPOOFING:
            return
            
        message = f"""
🚨 <b>ОБНАРУЖЕНА ПОДДЕЛКА ЛИЦА!</b>

📸 Изображение: {image_info}
�� Уверенность: {confidence:.2%}
⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ Возможная попытка обмана системы!
"""
        self.send_message(message)
    
    def notify_low_confidence(self, image_info: str, confidence: float):
        """Уведомление о низкой уверенности"""
        if not config.NOTIFY_ON_LOW_CONFIDENCE:
            return
            
        message = f"""
⚠️ <b>НИЗКАЯ УВЕРЕННОСТЬ АНТИСПУФИНГА</b>

📸 Изображение: {image_info}
🎯 Уверенность: {confidence:.2%}
⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 Требуется дополнительная проверка
"""
        self.send_message(message)
    
    def notify_comparison_result(self, similarity: float, verified: bool, processing_time: float):
        """Уведомление о результате сравнения"""
        status = "✅ ВЕРИФИЦИРОВАН" if verified else "❌ НЕ ВЕРИФИЦИРОВАН"
        
        message = f"""
📊 <b>РЕЗУЛЬТАТ СРАВНЕНИЯ ЛИЦ</b>

{status}
🎯 Сходство: {similarity:.2%}
⏱️ Время обработки: {processing_time:.3f}с
⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_error(self, error_type: str, error_message: str):
        """Уведомление об ошибке"""
        if not config.NOTIFY_ON_ERRORS:
            return
            
        message = f"""
🔴 <b>ОШИБКА СИСТЕМЫ</b>

🏷️ Тип: {error_type}
📝 Сообщение: {error_message}
⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 Требуется внимание администратора
"""
        self.send_message(message)
    
    def notify_high_volume(self, requests_count: int, time_window: int):
        """Уведомление о высокой нагрузке"""
        if not config.NOTIFY_ON_HIGH_VOLUME:
            return
            
        message = f"""
📈 <b>ВЫСОКАЯ НАГРУЗКА НА СИСТЕМУ</b>

📊 Запросов: {requests_count}
⏱️ За период: {time_window} сек
⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚡ Система работает в режиме повышенной нагрузки
"""
        self.send_message(message)

# Глобальный экземпляр
telegram_notifier = TelegramNotifier()
