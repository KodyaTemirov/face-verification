# Face Verification Service

Микросервис для сравнения лиц с функциями анти-спуфинга, предназначенный для систем прокторинга онлайн-экзаменов.

## Функциональность

- ✅ Сравнение лиц по URL изображений
- ✅ Анти-спуфинг детекция (живые лица vs поддельные)
- ✅ Валидация качества изображений
- ✅ RESTful API с автоматической документацией
- ✅ Структурированное логирование
- ✅ Docker контейнеризация

## Технический стек

- **Backend**: FastAPI
- **ML библиотека**: DeepFace с моделью ArcFace
- **Computer Vision**: OpenCV
- **Контейнеризация**: Docker & Docker Compose
- **Python**: 3.9+

## Быстрый старт

### Запуск через Docker Compose

```bash
# Клонирование и переход в директорию
git clone <repository-url>
cd face-verification

# Запуск сервиса
docker-compose up --build

# Сервис будет доступен по адресу: http://localhost:8000
```

### Локальный запуск

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Документация

После запуска сервиса документация доступна по адресам:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Основные эндпоинты

### 1. Сравнение лиц
```http
POST /api/v1/compare-faces
Content-Type: application/json

{
  "reference_image_url": "https://example.com/person1.jpg",
  "candidate_image_url": "https://example.com/person2.jpg",
  "threshold": 0.6
}
```

**Ответ:**
```json
{
  "is_same_person": true,
  "confidence": 0.85,
  "distance": 0.23,
  "anti_spoofing": {
    "reference_is_real": true,
    "candidate_is_real": true,
    "reference_confidence": 0.92,
    "candidate_confidence": 0.88
  },
  "processing_time": 1.2,
  "status": "success"
}
```

### 2. Валидация изображения
```http
POST /api/v1/validate-image
Content-Type: application/json

{
  "image_url": "https://example.com/photo.jpg"
}
```

### 3. Health Check
```http
GET /api/v1/health
```

## Конфигурация

Основные параметры настраиваются через переменные окружения:

```bash
# DeepFace настройки
MODEL_NAME=ArcFace
DETECTOR_BACKEND=opencv
DISTANCE_METRIC=cosine

# Пороговые значения
DEFAULT_THRESHOLD=0.6
ANTI_SPOOFING_THRESHOLD=0.5

# Ограничения
MAX_IMAGE_SIZE=10485760  # 10MB
REQUEST_TIMEOUT=30

# Логирование
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Архитектура

```
app/
├── main.py              # Точка входа FastAPI
├── api/
│   └── routes.py        # API маршруты
├── core/
│   ├── config.py        # Конфигурация
│   └── logging.py       # Настройка логирования
├── services/
│   ├── face_comparison.py    # Сервис сравнения лиц
│   ├── anti_spoofing.py      # Анти-спуфинг модуль
│   └── image_processor.py    # Обработка изображений
├── models/
│   └── schemas.py       # Pydantic схемы
└── utils/
    └── helpers.py       # Утилитарные функции
```

## Требования к производительности

- **Время обработки**: < 3 секунд на запрос
- **Точность сравнения**: > 95% на качественных изображениях
- **Точность анти-спуфинга**: > 90% детекции атак

## Безопасность

- Валидация всех входных URL
- Ограничение размера изображений
- Санитизация URL и блокировка внутренних сетей
- Структурированное логирование без чувствительных данных

## Мониторинг

Сервис предоставляет метрики через:
- Health check эндпоинты
- Структурированные логи в JSON формате
- Время обработки в заголовках ответов

## Тестирование

```bash
# Запуск тестов (если реализованы)
pytest tests/

# Проверка health check
curl http://localhost:8000/api/v1/health

# Тестовый запрос сравнения
curl -X POST "http://localhost:8000/api/v1/compare-faces" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_image_url": "https://example.com/person1.jpg",
    "candidate_image_url": "https://example.com/person2.jpg"
  }'
```

## Развертывание

### Docker
```bash
docker build -t face-verification .
docker run -p 8000:8000 face-verification
```

### Docker Compose
```bash
docker-compose up -d
```

## Автозапуск при перезагрузке сервера

Для автоматического запуска сервиса при включении сервера настроены специальные скрипты:

### Управление сервисом
```bash
# Запуск сервиса
./manage-service.sh start

# Остановка сервиса
./manage-service.sh stop

# Перезапуск сервиса
./manage-service.sh restart

# Проверка состояния
./manage-service.sh status

# Просмотр логов
./manage-service.sh logs

# Проверка здоровья API
./manage-service.sh health
```

### Автозапуск
✅ **Настроен автозапуск через crontab**

Сервис автоматически запускается при перезагрузке сервера. Подробные инструкции в файле `AUTOSTART_SETUP.md`.

## Лицензия

MIT License

## Поддержка

Для вопросов и поддержки создавайте issues в репозитории проекта. 