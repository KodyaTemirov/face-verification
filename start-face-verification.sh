#!/bin/bash

# Скрипт автозапуска Face Verification Service
# Используется для автоматического запуска при перезагрузке сервера

# Логирование
LOG_FILE="/home/kodyatemirov/face-verification/startup.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "$(date): Запуск Face Verification Service..."

# Переход в рабочую директорию
cd /home/kodyatemirov/face-verification || {
    echo "$(date): Ошибка: не найдена директория /home/kodyatemirov/face-verification"
    exit 1
}

# Проверка наличия docker-compose.yml
if [ ! -f "docker-compose.yml" ]; then
    echo "$(date): Ошибка: не найден файл docker-compose.yml"
    exit 1
fi

# Ожидание готовности Docker
echo "$(date): Ожидание готовности Docker..."
max_attempts=30
attempt=0

while ! docker info >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
        echo "$(date): Ошибка: Docker не запустился за $max_attempts попыток"
        exit 1
    fi
    
    echo "$(date): Docker еще не готов, ожидание... (попытка $((attempt + 1))/$max_attempts)"
    sleep 5
    ((attempt++))
done

echo "$(date): Docker готов к работе"

# Остановка старых контейнеров (если есть)
echo "$(date): Остановка старых контейнеров..."
docker-compose down --remove-orphans >/dev/null 2>&1

# Запуск сервиса
echo "$(date): Запуск Face Verification Service..."
if docker-compose up -d --remove-orphans; then
    echo "$(date): Face Verification Service успешно запущен"
    
    # Проверка состояния контейнеров
    sleep 10
    echo "$(date): Проверка состояния контейнеров:"
    docker-compose ps
    
    # Проверка логов на наличие ошибок
    echo "$(date): Последние логи:"
    docker-compose logs --tail=5
    
else
    echo "$(date): Ошибка запуска Face Verification Service"
    exit 1
fi

echo "$(date): Скрипт автозапуска завершен успешно" 