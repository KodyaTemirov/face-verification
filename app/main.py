from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import structlog

# Импорт модулей приложения
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router
from app.services import face_comparison_service

# Настройка логирования
setup_logging()
logger = structlog.get_logger(__name__)

# Создание экземпляра FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Сервис сравнения лиц для прокторинга с функциями анти-спуфинга",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware для CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware для доверенных хостов (в продакшене настроить нужные хосты)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # В продакшене указать конкретные домены
)


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование всех HTTP запросов"""
    start_time = time.time()
    
    # Логирование входящего запроса
    logger.info(
        "Входящий запрос",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Логирование ответа
        process_time = time.time() - start_time
        logger.info(
            "Ответ отправлен",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=f"{process_time:.3f}s"
        )
        
        # Добавление заголовка с временем обработки
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Ошибка обработки запроса",
            method=request.method,
            url=str(request.url),
            error=str(e),
            process_time=f"{process_time:.3f}s"
        )
        raise


# Обработчик глобальных ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик HTTP ошибок"""
    logger.error(
        "HTTP ошибка",
        method=request.method,
        url=str(request.url),
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих ошибок"""
    logger.error(
        "Необработанная ошибка",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервера",
            "status_code": 500,
            "path": str(request.url)
        }
    )


# События приложения
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    logger.info(f"Запуск {settings.app_name} v{settings.app_version}")
    
    try:
        # Предзагрузка моделей при старте (опционально)
        if not settings.debug:
            logger.info("Предзагрузка моделей...")
            await face_comparison_service.initialize()
            logger.info("Модели успешно загружены")
        
        logger.info("Сервис готов к работе")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        # В продакшене можно завершить приложение при критических ошибках
        # raise


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении приложения"""
    logger.info("Завершение работы сервиса")


# Подключение роутеров
app.include_router(router)

# Корневой эндпоинт
@app.get("/", include_in_schema=False)
async def root():
    """Корневой эндпоинт"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Дополнительный health check эндпоинт на корневом уровне
@app.get("/health", include_in_schema=False)
async def health():
    """Простой health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 