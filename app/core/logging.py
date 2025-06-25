import logging
import sys
import structlog
from typing import Any, Dict
import json
from datetime import datetime

from app.core.config import settings


def setup_logging():
    """Настройка структурированного логирования"""
    
    # Конфигурация structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Настройка стандартного логирования
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper())
    )
    
    # Установка уровня для различных компонентов
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("deepface").setLevel(logging.WARNING)


class JSONRenderer:
    """Кастомный JSON рендерер для логов"""
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict) -> str:
        """Рендеринг лога в JSON формат"""
        
        # Базовая структура лога
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": event_dict.get("level", "info").upper(),
            "logger": event_dict.get("logger", "app"),
            "message": event_dict.get("event", ""),
        }
        
        # Добавление дополнительных полей
        for key, value in event_dict.items():
            if key not in ["level", "logger", "event", "timestamp"]:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """Получение структурированного логгера"""
    return structlog.get_logger(name or __name__)


# Инициализация логирования при импорте модуля
setup_logging() 