#!/bin/bash

# Startup script для Face Comparison Service
# Файл: startup.sh

set -e

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SERVICE_DIR/logs/startup.log"
MAX_RETRIES=${MAX_RETRIES:-3}
RETRY_DELAY=${RETRY_DELAY:-30}

# Создаем лог файл если не существует
mkdir -p $(dirname "$LOG_FILE")
touch "$LOG_FILE"

# Функция логирования
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Проверка GPU
check_gpu() {
    log "Проверка доступности GPU..."
    if nvidia-smi > /dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        log "GPU доступен: $GPU_INFO"
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
        log "Запустите Docker: sudo systemctl start docker"
        return 1
    fi
}

# Проверка портов
check_port() {
    local port=$1
    if ss -tuln | grep -q ":$port "; then
        log "ПРЕДУПРЕЖДЕНИЕ: Порт $port уже используется"
        log "Освободите порт: sudo fuser -k $port/tcp"
        return 1
    else
        log "Порт $port свободен"
        return 0
    fi
}

# Очистка старых контейнеров
cleanup_containers() {
    log "Очистка старых контейнеров..."
    cd "$SERVICE_DIR"
    docker-compose down --remove-orphans || true
    
    # Удаляем неиспользуемые образы и контейнеры
    docker system prune -f || true
}

# Основная функция запуска
start_service() {
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        log "=== Попытка запуска #$attempt ==="
        
        # Проверки перед запуском
        if ! check_gpu; then
            log "Ожидание доступности GPU..."
            sleep $RETRY_DELAY
            continue
        fi
        
        if ! check_docker; then
            log "Docker недоступен, переходим к следующей попытке..."
            sleep $RETRY_DELAY
            continue
        fi
        
        # Переход в рабочую директорию
        cd "$SERVICE_DIR" || {
            log "ОШИБКА: Не удается перейти в $SERVICE_DIR"
            exit 1
        }
        
        # Очистка предыдущих контейнеров
        cleanup_containers
        
        # Проверка свободности порта
        check_port 8000
        
        # Создание необходимых директорий
        mkdir -p models logs temp
        
        # Запуск сервиса
        log "Запуск Face Comparison Service (InsightFace)..."
        if timeout 300 docker-compose up -d --build; then
            log "Сервис запущен, проверка работоспособности..."
            
            # Ожидание инициализации InsightFace модели
            sleep 45
            
            # Проверка здоровья сервиса
            local health_attempts=0
            local max_health_attempts=10
            
            while [ $health_attempts -lt $max_health_attempts ]; do
                if curl -f -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
                    log "✅ Сервис работает корректно"
                    
                    # Проверка GPU в ответе
                    HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
                    if echo "$HEALTH_RESPONSE" | grep -q '"gpu_available":true'; then
                        log "✅ GPU доступен в сервисе"
                    else
                        log "⚠️ GPU недоступен в сервисе, но сервис работает"
                    fi
                    
                    return 0
                else
                    log "Ожидание готовности сервиса... ($((health_attempts + 1))/$max_health_attempts)"
                    sleep 15
                    health_attempts=$((health_attempts + 1))
                fi
            done
            
            log "❌ Сервис не отвечает на health check"
            docker-compose logs --tail=50
            
        else
            log "❌ Ошибка запуска сервиса"
            docker-compose logs --tail=50
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -le $MAX_RETRIES ]; then
            log "Повторная попытка через $RETRY_DELAY секунд..."
            sleep $RETRY_DELAY
        fi
    done
    
    log "💥 КРИТИЧЕСКАЯ ОШИБКА: Не удалось запустить сервис после $MAX_RETRIES попыток"
    
    # Отправка критического уведомления
    if command -v curl > /dev/null && [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" \
            -d text="🚨 КРИТИЧЕСКАЯ ОШИБКА: Face Comparison Service не удалось запустить после $MAX_RETRIES попыток на $(hostname)" \
            > /dev/null 2>&1 || true
    fi
    
    exit 1
}

# Функция остановки
stop_service() {
    log "=== Остановка Face Comparison Service ==="
    cd "$SERVICE_DIR"
    
    # Graceful shutdown
    if docker-compose ps -q > /dev/null 2>&1; then
        log "Остановка контейнеров..."
        timeout 60 docker-compose down || {
            log "Принудительная остановка контейнеров..."
            docker-compose kill
            docker-compose down --remove-orphans
        }
    else
        log "Контейнеры уже остановлены"
    fi
    
    log "✅ Сервис остановлен"
}

# Функция перезапуска
restart_service() {
    log "=== Перезапуск Face Comparison Service ==="
    stop_service
    sleep 5
    start_service
}

# Функция проверки статуса
status_service() {
    log "=== Статус Face Comparison Service ==="
    cd "$SERVICE_DIR"
    
    if docker-compose ps | grep -q "Up"; then
        log "✅ Сервис запущен"
        
        # Проверка здоровья
        if curl -f -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            log "✅ Health check прошел"
        else
            log "⚠️ Health check не прошел"
        fi
        
        # Показываем статус контейнеров
        docker-compose ps
        
    else
        log "❌ Сервис не запущен"
        exit 1
    fi
}

# Обработка сигналов
trap 'log "Получен сигнал остановки"; stop_service; exit 0' SIGTERM SIGINT

# Основная логика
case "${1:-start}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Использование: $0 {start|stop|restart|status}"
        echo "Переменные окружения:"
        echo "  MAX_RETRIES - максимальное количество попыток запуска (по умолчанию: 3)"
        echo "  RETRY_DELAY - задержка между попытками в секундах (по умолчанию: 30)"
        exit 1
        ;;
esac 