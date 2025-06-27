# Face Comparison Service - InsightFace Edition

Сервис сравнения лиц с использованием InsightFace (ArcFace модель) и автоматическим запуском при включении сервера.

## 🚀 Особенности

- **InsightFace ArcFace** - высокоточная модель распознавания лиц
- **ONNX Runtime** - быстрый inference без TensorFlow
- **REST API** - простое интегрирование через HTTP
- **Docker контейнеризация** - легкое развертывание
- **Автозапуск** - systemd сервис для автоматического запуска
- **Health monitoring** - автоматическое восстановление при сбоях
- **Telegram уведомления** - мониторинг статуса сервиса

## 📋 Требования

### Системные требования
- **OS**: Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- **RAM**: 4GB+ (рекомендуется 8GB)
- **CPU**: 4+ cores (поддержка AVX2 рекомендуется)
- **Диск**: 5GB свободного места
- **Сеть**: порт 8000 должен быть свободен

### Программное обеспечение
- **Docker** 20.10+
- **Docker Compose** 1.29+
- **Python** 3.9+ (для разработки)
- **curl** (для health checks)

## 🛠 Установка

### Быстрая установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd face-verification

# Запуск установки
sudo chmod +x scripts/install-service.sh
sudo ./scripts/install-service.sh

# Проверка статуса
sudo systemctl status face-comparison.service
```

### Ручная установка

1. **Установка Docker и Docker Compose**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker

# Добавление пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker
```

2. **Настройка проекта**
```bash
# Создание рабочей директории
sudo mkdir -p /opt/face-comparison-service
sudo chown $USER:$USER /opt/face-comparison-service
cd /opt/face-comparison-service

# Копирование файлов проекта
cp -r /path/to/face-verification/* .

# Создание необходимых директорий
mkdir -p models logs temp
```

3. **Настройка переменных окружения**
```bash
# Создание .env файла
cat > .env << EOF
# Telegram уведомления (опционально)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Настройки сервиса
LOG_LEVEL=INFO
MAX_RETRIES=3
RETRY_DELAY=30
EOF
```

4. **Первый запуск**
```bash
# Сборка и запуск контейнера
docker-compose up -d --build

# Проверка статуса
docker-compose ps
curl http://localhost:8000/api/v1/health
```

## 🔧 Конфигурация

### Основные файлы конфигурации

- `docker-compose.yml` - конфигурация Docker контейнера
- `requirements.txt` - Python зависимости
- `face-comparison.service` - systemd сервис
- `startup.sh` - скрипт запуска с проверками
- `healthcheck.sh` - скрипт мониторинга здоровья

### Переменные окружения

| Переменная | Описание | По умолчанию | Обязательная |
|-----------|----------|--------------|--------------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота | - | Нет |
| `TELEGRAM_CHAT_ID` | ID чата для уведомлений | - | Нет |
| `LOG_LEVEL` | Уровень логирования | INFO | Нет |
| `MODEL_CACHE_DIR` | Директория для кеша моделей | /app/models | Нет |
| `MAX_RETRIES` | Максимум попыток запуска | 3 | Нет |
| `RETRY_DELAY` | Задержка между попытками (сек) | 30 | Нет |

## 📡 API Документация

### Базовый URL
```
http://localhost:8000
```

### Эндпоинты

#### 1. Информация о сервисе
```http
GET /
```

**Ответ:**
```json
{
  "message": "Face Comparison Service",
  "version": "1.0.0",
  "status": "running",
  "model": "InsightFace ArcFace",
  "docs": "/docs"
}
```

#### 2. Health Check
```http
GET /api/v1/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0",
  "model": "InsightFace ArcFace",
  "models_loaded": true,
  "uptime_seconds": 3600,
  "total_comparisons": 150,
  "gpu_available": false
}
```

#### 3. Сравнение лиц
```http
POST /api/v1/compare
```

**Тело запроса:**
```json
{
  "image1": "base64_encoded_image",
  "image2": "base64_encoded_image",
  "model": "ArcFace",
  "metric": "cosine",
  "threshold": 0.5
}
```

**Ответ:**
```json
{
  "verified": true,
  "distance": 0.23,
  "similarity": 0.77,
  "similarity_percentage": 77.0,
  "threshold": 0.5,
  "model": "InsightFace ArcFace",
  "metric": "cosine",
  "processing_time": 0.156,
  "timestamp": "2024-01-01T12:00:00",
  "faces_detected": {
    "image1": 1,
    "image2": 1
  }
}
```

#### 4. Список моделей
```http
GET /api/v1/models
```

**Ответ:**
```json
{
  "models": ["ArcFace"],
  "current_model": "InsightFace ArcFace",
  "backend": "ONNX Runtime",
  "note": "Using InsightFace ArcFace implementation"
}
```

#### 5. Расширенный статус
```http
GET /api/v1/status
```

**Ответ:**
```json
{
  "service": "Face Comparison Service",
  "version": "1.0.0",
  "model": "InsightFace ArcFace",
  "uptime_seconds": 3600,
  "uptime_formatted": "1:00:00",
  "gpu_available": false,
  "models_loaded": true,
  "statistics": {
    "total_comparisons": 150,
    "successful_comparisons": 147,
    "failed_comparisons": 3,
    "success_rate": 98.0,
    "average_response_time": 0.156
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Swagger документация
Доступна по адресу: `http://localhost:8000/docs`

## 🔄 Управление сервисом

### Systemd команды
```bash
# Запуск сервиса
sudo systemctl start face-comparison.service

# Остановка сервиса
sudo systemctl stop face-comparison.service

# Перезапуск сервиса
sudo systemctl restart face-comparison.service

# Статус сервиса
sudo systemctl status face-comparison.service

# Автозапуск при загрузке
sudo systemctl enable face-comparison.service

# Отключение автозапуска
sudo systemctl disable face-comparison.service
```

### Docker Compose команды
```bash
# Запуск в фоне
docker-compose up -d

# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Просмотр логов
docker-compose logs -f

# Пересборка и запуск
docker-compose up -d --build
```

### Скрипты управления
```bash
# Использование startup.sh
./startup.sh start    # Запуск с проверками
./startup.sh stop     # Остановка
./startup.sh restart  # Перезапуск
./startup.sh status   # Проверка статуса

# Ручная проверка здоровья
./healthcheck.sh
```

## 📊 Мониторинг и логирование

### Логи
- **Основные логи**: `logs/face-comparison.log`
- **Health check логи**: `logs/health.log`
- **Startup логи**: `logs/startup.log`
- **Docker логи**: `docker-compose logs`

### Мониторинг
- **Health check**: каждые 30 секунд
- **Автоперезапуск**: при 3 неудачных проверках подряд
- **Cooldown**: 5 минут между перезапусками
- **Уведомления**: Telegram (если настроен)

### Просмотр логов
```bash
# Основные логи сервиса
tail -f logs/face-comparison.log

# Health check логи
tail -f logs/health.log

# Docker логи
docker-compose logs -f face-comparison

# Systemd логи
sudo journalctl -u face-comparison.service -f
```

## 🔧 Разработка

### Локальная разработка
```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск в режиме разработки
mkdir -p logs
python app/main.py
```

### Тестирование API
```bash
# Тест health check
curl http://localhost:8000/api/v1/health

# Тест сравнения (требуется base64 изображения)
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "image1": "base64_image_1",
    "image2": "base64_image_2",
    "threshold": 0.5
  }'
```

### Структура проекта
```
face-verification/
├── app/
│   ├── __init__.py
│   └── main.py              # Основное приложение
├── scripts/
│   ├── install-service.sh   # Скрипт установки
│   └── uninstall-service.sh # Скрипт удаления
├── logs/                    # Логи (создается автоматически)
├── models/                  # Кеш моделей (создается автоматически)
├── temp/                    # Временные файлы
├── docker-compose.yml       # Docker Compose конфигурация
├── Dockerfile              # Docker образ
├── requirements.txt        # Python зависимости
├── startup.sh              # Скрипт запуска
├── healthcheck.sh          # Скрипт мониторинга
├── face-comparison.service # Systemd сервис
├── .env                    # Переменные окружения (создается вручную)
└── README.md              # Документация
```

## 🚨 Устранение неполадок

### Частые проблемы

1. **Сервис не запускается**
```bash
# Проверка статуса Docker
sudo systemctl status docker

# Проверка портов
sudo netstat -tlnp | grep :8000

# Проверка логов
docker-compose logs face-comparison
```

2. **Модель не загружается**
```bash
# Проверка места на диске
df -h

# Очистка Docker кеша
docker system prune -f

# Пересборка контейнера
docker-compose build --no-cache
```

3. **Health check не проходит**
```bash
# Ручная проверка API
curl -v http://localhost:8000/api/v1/health

# Проверка контейнера
docker-compose exec face-comparison ps aux
```

4. **Высокое использование памяти**
```bash
# Мониторинг ресурсов
docker stats

# Перезапуск сервиса
docker-compose restart
```

### Диагностика
```bash
# Полная диагностика
./scripts/diagnose.sh

# Проверка конфигурации
docker-compose config

# Тест всех эндпоинтов
curl http://localhost:8000/
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/models
curl http://localhost:8000/api/v1/status
```

## 📞 Поддержка

### Логи для отправки при проблемах
```bash
# Сбор всех логов
mkdir -p debug_logs
cp logs/*.log debug_logs/
docker-compose logs > debug_logs/docker.log
sudo journalctl -u face-comparison.service > debug_logs/systemd.log
tar -czf debug_logs.tar.gz debug_logs/
```

### Полезные команды
```bash
# Информация о системе
uname -a
docker --version
docker-compose --version

# Состояние сервиса
sudo systemctl status face-comparison.service
docker-compose ps
curl http://localhost:8000/api/v1/status
```

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл LICENSE для деталей. 