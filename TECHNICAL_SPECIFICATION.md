# Техническое задание: Сервис сравнения лиц для прокторинга

## Описание проекта
Создать микросервис для сравнения лиц с функциями анти-спуфинга, предназначенный для систем прокторинга онлайн-экзаменов.

## Основные требования

### Функциональность
- **Сравнение лиц**: Определение идентичности двух лиц по переданным URL изображений
- **Анти-спуфинг**: Детекция попыток обмана (фотографии, видео, маски, deepfake)
- **API интерфейс**: RESTful API для интеграции с системами прокторинга
- **Обработка URL**: Загрузка и обработка изображений по HTTP ссылкам

### Технический стек
- **Фреймворк**: FastAPI
- **Библиотека для работы с лицами**: DeepFace
- **Модель распознавания**: ArcFace
- **Контейнеризация**: Docker
- **Язык программирования**: Python 3.9+

## Архитектура API

### Эндпоинты

#### 1. Сравнение лиц
```
POST /api/v1/compare-faces
```

**Тело запроса:**
```json
{
  "reference_image_url": "https://example.com/image1.jpg",
  "candidate_image_url": "https://example.com/image2.jpg",
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

#### 2. Проверка качества изображения
```
POST /api/v1/validate-image
```

#### 3. Статус сервиса
```
GET /api/v1/health
```

## Требования к реализации

### Основные компоненты

1. **Face Comparison Service**
   - Использование DeepFace с моделью ArcFace
   - Предобработка изображений (выравнивание, нормализация)
   - Вычисление векторных представлений лиц
   - Расчет метрик схожести

2. **Anti-Spoofing Module**
   - Детекция живых лиц vs фотографий
   - Анализ текстур и глубины
   - Проверка на deepfake атаки
   - Интеграция с дополнительными моделями детекции

3. **Image Processing Pipeline**
   - Загрузка изображений по URL
   - Валидация формата и размера
   - Детекция и выделение лиц
   - Обработка ошибок загрузки

4. **API Layer**
   - FastAPI с автоматической документацией
   - Валидация входных данных с Pydantic
   - Обработка ошибок и логирование
   - Метрики производительности

### Конфигурация DeepFace
```python
# Основные настройки
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"
ENFORCE_DETECTION = True
```

### Docker конфигурация

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml
```yaml
version: '3.8'
services:
  face-comparison-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## Структура проекта
```
face-comparison-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_comparison.py
│   │   ├── anti_spoofing.py
│   │   └── image_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Требования к производительности
- **Время обработки**: < 3 секунд на запрос
- **Пропускная способность**: 100+ запросов в минуту
- **Точность сравнения**: > 95% на качественных изображениях
- **Точность анти-спуфинга**: > 90% детекции атак

## Обработка ошибок
- Неверные URL изображений
- Отсутствие лиц на изображениях
- Множественные лица на изображении
- Низкое качество изображений
- Сетевые ошибки при загрузке

## Логирование и мониторинг
- Структурированное логирование всех операций
- Метрики времени обработки
- Статистика успешных/неуспешных запросов
- Мониторинг использования ресурсов

## Безопасность
- Валидация всех входных данных
- Ограничение размера загружаемых изображений
- Санитизация URL
- Rate limiting для предотвращения злоупотреблений

## Дополнительные возможности (опционально)
- Кэширование результатов сравнения
- Поддержка batch обработки
- Веб-интерфейс для тестирования
- Метрики качества изображений
- Интеграция с облачными хранилищами

## Тестирование
- Unit тесты для всех компонентов
- Интеграционные тесты API
- Нагрузочное тестирование
- Тесты на различных типах изображений

## Развертывание
- Поддержка Docker Compose для локальной разработки
- Готовность к развертыванию в Kubernetes
- CI/CD pipeline для автоматического тестирования и развертывания 