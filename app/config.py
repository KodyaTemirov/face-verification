#!/usr/bin/env python3
"""
Конфигуратор для Face Comparison Service
"""
import os

class Config:
    """Конфигурация сервиса"""
    
    # Telegram настройки
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
    
    # Уведомления
    NOTIFY_ON_SPOOFING = os.getenv("NOTIFY_ON_SPOOFING", "true").lower() == "true"
    NOTIFY_ON_LOW_CONFIDENCE = os.getenv("NOTIFY_ON_LOW_CONFIDENCE", "true").lower() == "true"
    NOTIFY_ON_ERRORS = os.getenv("NOTIFY_ON_ERRORS", "true").lower() == "true"
    
    # Антиспуфинг
    ANTISPOOF_THRESHOLD = float(os.getenv("ANTISPOOF_THRESHOLD", "0.5"))
    ANTISPOOF_CONFIDENCE_MIN = float(os.getenv("ANTISPOOF_CONFIDENCE_MIN", "0.7"))
    
    @classmethod
    def is_telegram_configured(cls) -> bool:
        """Проверка настройки Telegram"""
        return bool(cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID and cls.TELEGRAM_ENABLED)

config = Config()
