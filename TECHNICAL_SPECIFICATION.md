# Дополнения к ТЗ: Автоматический запуск при включении сервера

## Требования к автозапуску

### Системные требования
- **Автозапуск**: Сервис должен автоматически запускаться при включении/перезагрузке сервера
- **Самовосстановление**: Автоматический перезапуск при сбоях
- **Проверка GPU**: Валидация доступности GPU перед запуском
- **Логирование запуска**: Полное логирование процесса автозапуска
- **Graceful shutdown**: Корректное завершение работы при выключении сервера

## Способы реализации автозапуска

### 1. Systemd Service (рекомендуется для Ubuntu/Linux)

#### Создание systemd сервиса
```ini
# /etc/systemd/system/face-comparison.service
[Unit]
Description=Face Comparison Service with GPU Support
After=network.target docker.service nvidia-persistenced.service
Requires=docker.service
Wants=nvidia-persistenced.service

[Service]
Type=exec
User=ubuntu
Group=docker
WorkingDirectory=/opt/face-comparison-service
ExecStartPre=/bin/bash -c 'nvidia-smi > /dev/null 2>&1 || (echo "GPU not available" && exit 1)'
ExecStart=/usr/bin/docker-compose up --build
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=face-comparison
KillMode=mixed
TimeoutStartSec=180
TimeoutStopSec=30

# Environment variables
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="COMPOSE_PROJECT_NAME=face_comparison"

[Install]
WantedBy=multi-user.target
```

#### Команды для настройки systemd
```bash
# Копирование файла сервиса
sudo cp face-comparison.service /etc/systemd/system/

# Перезагрузка systemd
sudo systemctl daemon-reload

# Включение автозапуска
sudo systemctl enable face-comparison.service

# Запуск сервиса
sudo systemctl start face-comparison.service

# Проверка статуса
sudo systemctl status face-comparison.service

# Просмотр логов
sudo journalctl -u face-comparison.service -f
```

### 2. Docker Compose с restart policy

#### Обновленный docker-compose.yml
```yaml
version: '3.8'
services:
  face-comparison-api:
    build: .
    container_name: face-comparison-service
    ports:
      - "8000:8000"
    environment:
      - MODEL_CACHE_DIR=/app/models
      - CUDA_VISIBLE_DEVICES=0
      - TF_FORCE_GPU_ALLOW_GROWTH=true
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - gpu-monitor
    
  gpu-monitor:
    image: nvidia/cuda:12.2-runtime-ubuntu22.04
    container_name: gpu-monitor
    command: ["nvidia-smi", "-l", "30"]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. Startup скрипт с проверками

#### startup.sh
```bash
#!/bin/bash

# Startup script для Face Comparison Service
# Файл: /opt/face-comparison-service/startup.sh

set -e

LOG_FILE="/var/log/face-comparison-startup.log"
SERVICE_DIR="/opt/face-comparison-service"
MAX_RETRIES=3
RETRY_DELAY=30

# Функция логирования
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Проверка GPU
check_gpu() {
    log "Проверка доступности GPU..."
    if nvidia-smi > /dev/null 2>&1; then
        log "GPU доступен: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
        return 0
    else
        log "ОШИБКА: GPU недоступен"
        return 1
    fi
}

# Проверка Docker
check_docker() {
    log "Проверка Docker..."
    if systemctl is-active --quiet docker; then
        log "Docker активен"
        return 0
    else
        log "ОШИБКА: Docker не активен"
        return 1
    fi
}

# Проверка портов
check_port() {
    local port=$1
    if ss -tuln | grep -q ":$port "; then
        log "ПРЕДУПРЕЖДЕНИЕ: Порт $port уже используется"
        return 1
    else
        log "Порт $port свободен"
        return 0
    fi
}

# Основная функция запуска
start_service() {
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        log "Попытка запуска #$attempt"
        
        # Проверки перед запуском
        if ! check_gpu; then
            log "Ожидание доступности GPU..."
            sleep $RETRY_DELAY
            continue
        fi
        
        if ! check_docker; then
            log "Запуск Docker..."
            sudo systemctl start docker
            sleep 10
        fi
        
        # Переход в рабочую директорию
        cd "$SERVICE_DIR" || {
            log "ОШИБКА: Не удается перейти в $SERVICE_DIR"
            exit 1
        }
        
        # Остановка предыдущих контейнеров
        log "Остановка предыдущих контейнеров..."
        docker-compose down || true
        
        # Проверка свободности порта
        if ! check_port 8000; then
            log "Освобождение порта 8000..."
            sudo fuser -k 8000/tcp || true
            sleep 5
        fi
        
        # Запуск сервиса
        log "Запуск Face Comparison Service..."
        if docker-compose up -d --build; then
            log "Сервис успешно запущен"
            
            # Проверка здоровья сервиса
            log "Проверка работоспособности сервиса..."
            sleep 30
            
            if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
                log "Сервис работает корректно"
                return 0
            else
                log "Сервис не отвечает на health check"
            fi
        else
            log "Ошибка запуска сервиса"
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -le $MAX_RETRIES ]; then
            log "Повторная попытка через $RETRY_DELAY секунд..."
            sleep $RETRY_DELAY
        fi
    done
    
    log "КРИТИЧЕСКАЯ ОШИБКА: Не удалось запустить сервис после $MAX_RETRIES попыток"
    exit 1
}

# Функция остановки
stop_service() {
    log "Остановка Face Comparison Service..."
    cd "$SERVICE_DIR"
    docker-compose down
    log "Сервис остановлен"
}

# Обработка сигналов
trap 'stop_service; exit 0' SIGTERM SIGINT

# Основная логика
case "${1:-start}" in
    start)
        log "=== Запуск Face Comparison Service ==="
        start_service
        ;;
    stop)
        log "=== Остановка Face Comparison Service ==="
        stop_service
        ;;
    restart)
        log "=== Перезапуск Face Comparison Service ==="
        stop_service
        sleep 5
        start_service
        ;;
    *)
        echo "Использование: $0 {start|stop|restart}"
        exit 1
        ;;
esac
```

### 4. Cron задача (альтернативный способ)

#### Crontab для автозапуска
```bash
# Редактирование crontab
sudo crontab -e

# Добавление задачи для автозапуска при перезагрузке
@reboot sleep 60 && /opt/face-comparison-service/startup.sh start >> /var/log/face-comparison-cron.log 2>&1

# Проверка работоспособности каждые 5 минут
*/5 * * * * /opt/face-comparison-service/healthcheck.sh >> /var/log/face-comparison-health.log 2>&1
```

### 5. Health Check скрипт

#### healthcheck.sh
```bash
#!/bin/bash

# Health check script
# Файл: /opt/face-comparison-service/healthcheck.sh

HEALTH_URL="http://localhost:8000/api/v1/health"
LOG_FILE="/var/log/face-comparison-health.log"
SERVICE_DIR="/opt/face-comparison-service"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Проверка API
if curl -f -s "$HEALTH_URL" > /dev/null; then
    # Проверка GPU в ответе
    response=$(curl -s "$HEALTH_URL")
    if echo "$response" | grep -q '"gpu_available":true'; then
        log "✓ Сервис работает нормально с GPU"
        exit 0
    else
        log "⚠ Сервис работает, но GPU недоступен"
        exit 1
    fi
else
    log "✗ Сервис недоступен, попытка перезапуска..."
    cd "$SERVICE_DIR"
    docker-compose restart
    sleep 30
    
    if curl -f -s "$HEALTH_URL" > /dev/null; then
        log "✓ Сервис восстановлен после перезапуска"
        exit 0
    else
        log "✗ Не удалось восстановить сервис"
        exit 1
    fi
fi
```

## Структура файлов для автозапуска

```
/opt/face-comparison-service/
├── docker-compose.yml
├── Dockerfile
├── startup.sh                  # Основной скрипт запуска
├── healthcheck.sh             # Скрипт проверки здоровья
├── app/                       # Код приложения
├── models/                    # Модели DeepFace
├── logs/                      # Логи сервиса
└── scripts/
    ├── install-service.sh     # Скрипт установки systemd сервиса
    └── uninstall-service.sh   # Скрипт удаления сервиса
```

## Логирование автозапуска

### Конфигурация логирования в приложении
```python
# app/core/logging.py
import logging
import sys
from pathlib import Path

def setup_logging():
    # Создание директории для логов
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Настройка логгера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "face-comparison.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Специальный логгер для GPU
    gpu_logger = logging.getLogger("gpu")
    gpu_handler = logging.FileHandler(log_dir / "gpu.log")
    gpu_handler.setFormatter(
        logging.Formatter('%(asctime)s - GPU - %(levelname)s - %(message)s')
    )
    gpu_logger.addHandler(gpu_handler)
    
    return logging.getLogger(__name__)
```

## Мониторинг автозапуска

### Уведомления о статусе
```python
# app/utils/notifications.py
import requests
import logging

logger = logging.getLogger(__name__)

def send_startup_notification(status: str, details: str = ""):
    """Отправка уведомления о статусе запуска"""
    try:
        # Можно настроить отправку в телеграм, slack, email и т.д.
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url:
            payload = {
                "text": f"Face Comparison Service: {status}\n{details}",
                "timestamp": datetime.now().isoformat()
            }
            requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Ошибка отправки уведомления: {e}")

# Использование в main.py
@app.on_event("startup")
async def startup_event():
    logger.info("Face Comparison Service запускается...")
    
    # Проверка GPU
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        send_startup_notification("✅ Успешно запущен", "GPU доступен")
        logger.info("Сервис успешно запущен с поддержкой GPU")
    else:
        send_startup_notification("⚠️ Запущен без GPU", "GPU недоступен")
        logger.warning("Сервис запущен без поддержки GPU")
```

## Скрипт установки

### install-service.sh
```bash
#!/bin/bash

# Скрипт установки Face Comparison Service
# Файл: scripts/install-service.sh

set -e

echo "=== Установка Face Comparison Service ==="

# Проверка прав root
if [[ $EUID -ne 0 ]]; then
   echo "Этот скрипт должен быть запущен от имени root"
   exit 1
fi

# Создание директории для сервиса
SERVICE_DIR="/opt/face-comparison-service"
mkdir -p "$SERVICE_DIR"

# Копирование файлов
cp -r . "$SERVICE_DIR/"
chmod +x "$SERVICE_DIR/startup.sh"
chmod +x "$SERVICE_DIR/healthcheck.sh"

# Создание пользователя для сервиса (если не существует)
if ! id "faceservice" &>/dev/null; then
    useradd -r -s /bin/false faceservice
    usermod -aG docker faceservice
fi

# Установка прав
chown -R faceservice:faceservice "$SERVICE_DIR"

# Создание директории для логов
mkdir -p /var/log/face-comparison
chown faceservice:faceservice /var/log/face-comparison

# Установка systemd сервиса
cp face-comparison.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable face-comparison.service

echo "✅ Face Comparison Service установлен и настроен для автозапуска"
echo "Для запуска используйте: sudo systemctl start face-comparison.service"
echo "Для просмотра статуса: sudo systemctl status face-comparison.service"
echo "Для просмотра логов: sudo journalctl -u face-comparison.service -f"
```

## Обновленные требования к производительности

### Время запуска
- **Автозапуск**: < 2 минут после включения сервера
- **Готовность API**: < 60 секунд после запуска контейнера
- **Проверка GPU**: < 10 секунд
- **Загрузка моделей**: < 90 секунд

### Надежность
- **Доступность**: 99.9% (с учетом автоперезапуска)
- **Восстановление после сбоя**: < 30 секунд
- **Мониторинг здоровья**: каждые 30 секунд
- **Автоматические перезапуски**: при критических ошибках GPU

Эти дополнения обеспечат полную автоматизацию запуска и мониторинга сервиса при включении сервера.