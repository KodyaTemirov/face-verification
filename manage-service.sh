#!/bin/bash

# Скрипт управления Face Verification Service
# Использование: ./manage-service.sh {start|stop|restart|status|logs}

SERVICE_DIR="/home/kodyatemirov/face-verification"
LOG_FILE="$SERVICE_DIR/startup.log"

# Функция для вывода цветного текста
print_status() {
    case $1 in
        "success") echo -e "\033[32m✅ $2\033[0m" ;;
        "error") echo -e "\033[31m❌ $2\033[0m" ;;
        "info") echo -e "\033[34mℹ️  $2\033[0m" ;;
        "warning") echo -e "\033[33m⚠️  $2\033[0m" ;;
    esac
}

# Переход в рабочую директорию
cd "$SERVICE_DIR" || {
    print_status "error" "Не найдена директория $SERVICE_DIR"
    exit 1
}

case "$1" in
    start)
        print_status "info" "Запуск Face Verification Service..."
        if docker-compose up -d --remove-orphans; then
            sleep 5
            print_status "success" "Сервис запущен!"
            print_status "info" "Проверка состояния:"
            docker-compose ps
        else
            print_status "error" "Ошибка запуска сервиса"
            exit 1
        fi
        ;;
    
    stop)
        print_status "info" "Остановка Face Verification Service..."
        if docker-compose down; then
            print_status "success" "Сервис остановлен!"
        else
            print_status "error" "Ошибка остановки сервиса"
            exit 1
        fi
        ;;
    
    restart)
        print_status "info" "Перезапуск Face Verification Service..."
        docker-compose down
        if docker-compose up -d --remove-orphans; then
            sleep 5
            print_status "success" "Сервис перезапущен!"
            print_status "info" "Проверка состояния:"
            docker-compose ps
        else
            print_status "error" "Ошибка перезапуска сервиса"
            exit 1
        fi
        ;;
    
    status)
        print_status "info" "Состояние Face Verification Service:"
        docker-compose ps
        echo ""
        
        # Проверка доступности API
        if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            print_status "success" "API доступен: http://localhost:8000"
            print_status "info" "Документация: http://localhost:8000/docs"
        else
            print_status "warning" "API недоступен на http://localhost:8000"
        fi
        ;;
    
    logs)
        print_status "info" "Логи Face Verification Service:"
        echo "--- Docker Compose логи ---"
        docker-compose logs --tail=20
        echo ""
        echo "--- Логи автозапуска ---"
        if [ -f "$LOG_FILE" ]; then
            tail -20 "$LOG_FILE"
        else
            print_status "warning" "Файл логов автозапуска не найден: $LOG_FILE"
        fi
        ;;
    
    health)
        print_status "info" "Проверка здоровья сервиса..."
        response=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "$response" | jq '.' 2>/dev/null || echo "$response"
            print_status "success" "Сервис работает корректно"
        else
            print_status "error" "Сервис недоступен"
            exit 1
        fi
        ;;
    
    *)
        echo "Использование: $0 {start|stop|restart|status|logs|health}"
        echo ""
        echo "Команды:"
        echo "  start   - Запустить сервис"
        echo "  stop    - Остановить сервис"
        echo "  restart - Перезапустить сервис"
        echo "  status  - Показать состояние сервиса"
        echo "  logs    - Показать логи"
        echo "  health  - Проверить здоровье API"
        exit 1
        ;;
esac 