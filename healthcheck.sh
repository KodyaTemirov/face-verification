#!/bin/bash

# Health check script для Face Comparison Service
# Файл: healthcheck.sh

HEALTH_URL="http://localhost:8000/api/v1/health"
SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SERVICE_DIR/logs/health.log"
RESTART_THRESHOLD=${RESTART_THRESHOLD:-3}
RESTART_COOLDOWN=${RESTART_COOLDOWN:-300}  # 5 минут
STATE_FILE="/tmp/face-comparison-health-state"

# Создаем лог файл если не существует
mkdir -p $(dirname "$LOG_FILE")
touch "$LOG_FILE"

# Функция логирования
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Получение состояния из файла
get_failure_count() {
    if [ -f "$STATE_FILE" ]; then
        local line=$(head -1 "$STATE_FILE")
        echo "${line%%:*}"
    else
        echo "0"
    fi
}

get_last_restart() {
    if [ -f "$STATE_FILE" ]; then
        local line=$(head -1 "$STATE_FILE")
        echo "${line##*:}"
    else
        echo "0"
    fi
}

# Сохранение состояния
save_state() {
    local failures=$1
    local last_restart=$2
    echo "$failures:$last_restart" > "$STATE_FILE"
}

# Отправка уведомления в Telegram
send_notification() {
    local message=$1
    local severity=${2:-"warning"}
    
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        local emoji
        case $severity in
            "info") emoji="ℹ️" ;;
            "warning") emoji="⚠️" ;;
            "error") emoji="❌" ;;
            "critical") emoji="🚨" ;;
            "success") emoji="✅" ;;
            *) emoji="📋" ;;
        esac
        
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" \
            -d text="$emoji Face Comparison Service ($(hostname)): $message" \
            > /dev/null 2>&1 || true
    fi
}

# Основная проверка здоровья
health_check() {
    local current_time=$(date +%s)
    local failure_count=$(get_failure_count)
    local last_restart=$(get_last_restart)
    
    log "Проверка здоровья сервиса..."
    
    # Базовая проверка API
    if ! curl -f -s -m 10 "$HEALTH_URL" > /dev/null 2>&1; then
        failure_count=$((failure_count + 1))
        log "❌ API недоступен (попытка $failure_count/$RESTART_THRESHOLD)"
        
        # Проверяем, не слишком ли недавно был перезапуск
        local time_since_restart=$((current_time - last_restart))
        
        if [ $failure_count -ge $RESTART_THRESHOLD ] && [ $time_since_restart -gt $RESTART_COOLDOWN ]; then
            log "🔄 Инициируем перезапуск сервиса..."
            send_notification "Сервис недоступен, выполняется перезапуск" "error"
            
            if restart_service; then
                save_state "0" "$current_time"
                log "✅ Сервис успешно перезапущен"
                send_notification "Сервис успешно перезапущен" "success"
                return 0
            else
                log "💥 Не удалось перезапустить сервис"
                send_notification "КРИТИЧЕСКАЯ ОШИБКА: Не удалось перезапустить сервис!" "critical"
                save_state "$failure_count" "$last_restart"
                return 1
            fi
            
        elif [ $time_since_restart -le $RESTART_COOLDOWN ]; then
            log "⏳ Перезапуск недавно выполнен, ожидаем..."
            save_state "$failure_count" "$last_restart"
            
        else
            save_state "$failure_count" "$last_restart"
        fi
        
        return 1
    fi
    
    # Детальная проверка health endpoint
    local health_response
    health_response=$(curl -s -m 10 "$HEALTH_URL" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$health_response" ]; then
        # Парсим JSON ответ (простая проверка)
        if echo "$health_response" | grep -q '"status":"healthy"'; then
            
            # Проверка InsightFace модели
            local model_status="неизвестен"
            if echo "$health_response" | grep -q '"models_loaded":true'; then
                model_status="загружена"
            elif echo "$health_response" | grep -q '"models_loaded":false'; then
                model_status="не загружена"
                log "⚠️ InsightFace модель не загружена"
                send_notification "InsightFace модель не загружена" "warning"
            fi
            
            # Проверка статистики
            local total_comparisons=$(echo "$health_response" | grep -o '"total_comparisons":[0-9]*' | cut -d':' -f2)
            local success_rate=$(echo "$health_response" | grep -o '"success_rate":[0-9.]*' | cut -d':' -f2)
            local avg_response_time=$(echo "$health_response" | grep -o '"average_response_time":[0-9.]*' | cut -d':' -f2)
            
            log "✅ Сервис работает нормально (InsightFace: $model_status)"
            log "📊 Статистика: сравнений=$total_comparisons, успешность=${success_rate}%, время=${avg_response_time}с"
            
            # Сброс счетчика ошибок при успешной проверке
            if [ "$failure_count" -gt 0 ]; then
                log "🔄 Сброс счетчика ошибок"
                save_state "0" "$last_restart"
            fi
            
            # Проверка критических параметров
            if [ -n "$success_rate" ] && [ "${success_rate%.*}" -lt 90 ] && [ -n "$total_comparisons" ] && [ "$total_comparisons" -gt 10 ]; then
                log "⚠️ Низкая успешность сравнений: ${success_rate}%"
                send_notification "Низкая успешность сравнений: ${success_rate}%" "warning"
            fi
            
            if [ -n "$avg_response_time" ] && [ "${avg_response_time%.*}" -gt 10 ]; then
                log "⚠️ Высокое время отклика: ${avg_response_time}с"
                send_notification "Высокое время отклика: ${avg_response_time}с" "warning"
            fi
            
            return 0
            
        else
            log "⚠️ Health endpoint отвечает, но статус не healthy"
            failure_count=$((failure_count + 1))
            save_state "$failure_count" "$last_restart"
            return 1
        fi
        
    else
        log "❌ Некорректный ответ от health endpoint"
        failure_count=$((failure_count + 1))
        save_state "$failure_count" "$last_restart"
        return 1
    fi
}

# Функция перезапуска сервиса
restart_service() {
    log "Попытка перезапуска сервиса..."
    
    cd "$SERVICE_DIR" || {
        log "❌ Не удается перейти в директорию сервиса: $SERVICE_DIR"
        return 1
    }
    
    # Используем startup.sh для перезапуска
    if [ -x "./startup.sh" ]; then
        timeout 600 ./startup.sh restart
        return $?
    else
        # Fallback к docker-compose
        log "startup.sh не найден, используем docker-compose..."
        docker-compose restart
        sleep 30
        
        # Проверяем что сервис запустился
        if curl -f -s -m 10 "$HEALTH_URL" > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
}

# Функция мониторинга ресурсов
monitor_resources() {
    local memory_usage
    local disk_usage
    local cpu_usage
    
    # Использование памяти контейнером
    if command -v docker >/dev/null 2>&1; then
        memory_usage=$(docker stats --no-stream --format "table {{.MemUsage}}" face-comparison-service 2>/dev/null | tail -1 | grep -o '[0-9.]*GiB' | head -1)
        if [ -n "$memory_usage" ]; then
            log "💾 Использование памяти: $memory_usage"
            
            # Проверка высокого использования памяти (более 4GB)
            local mem_value=$(echo "$memory_usage" | grep -o '[0-9.]*')
            if [ -n "$mem_value" ] && [ "${mem_value%.*}" -gt 4 ]; then
                log "⚠️ Высокое использование памяти: $memory_usage"
                send_notification "Высокое использование памяти: $memory_usage" "warning"
            fi
        fi
    fi
    
    # Использование диска
    disk_usage=$(df -h "$(pwd)" | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log "⚠️ Высокое использование диска: ${disk_usage}%"
        send_notification "Высокое использование диска: ${disk_usage}%" "warning"
    fi
}

# Главная функция
main() {
    # Создаем lock file для предотвращения одновременного выполнения
    local lock_file="/tmp/face-comparison-healthcheck.lock"
    
    if [ -f "$lock_file" ]; then
        local lock_pid=$(cat "$lock_file")
        if kill -0 "$lock_pid" 2>/dev/null; then
            log "Другой процесс healthcheck уже выполняется (PID: $lock_pid)"
            exit 0
        else
            log "Удаляем устаревший lock file"
            rm -f "$lock_file"
        fi
    fi
    
    echo $$ > "$lock_file"
    
    # Выполняем проверку
    if health_check; then
        monitor_resources
        local exit_code=0
    else
        local exit_code=1
    fi
    
    # Удаляем lock file
    rm -f "$lock_file"
    
    exit $exit_code
}

# Обработка аргументов командной строки
case "${1:-check}" in
    check)
        main
        ;;
    restart)
        restart_service
        ;;
    reset)
        log "Сброс состояния healthcheck"
        rm -f "$STATE_FILE"
        ;;
    status)
        echo "Failure count: $(get_failure_count)"
        echo "Last restart: $(date -d @$(get_last_restart) 2>/dev/null || echo 'never')"
        ;;
    *)
        echo "Использование: $0 {check|restart|reset|status}"
        echo "Переменные окружения:"
        echo "  RESTART_THRESHOLD - количество неудач для перезапуска (по умолчанию: 3)"
        echo "  RESTART_COOLDOWN - время ожидания между перезапусками в секундах (по умолчанию: 300)"
        exit 1
        ;;
esac 