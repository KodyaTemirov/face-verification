#!/bin/bash

# Скрипт удаления Face Comparison Service
# Файл: scripts/uninstall-service.sh

set -e

echo "=== Удаление Face Comparison Service ==="

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Проверка прав root
if [[ $EUID -ne 0 ]]; then
   log_error "Этот скрипт должен быть запущен от имени root"
   echo "Используйте: sudo $0"
   exit 1
fi

SERVICE_DIR="/opt/face-comparison-service"
SERVICE_NAME="face-comparison.service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"

# Остановка и отключение сервиса
stop_service() {
    log_info "Остановка сервиса..."
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
        log_success "Сервис остановлен"
    else
        log_info "Сервис уже остановлен"
    fi
    
    if systemctl is-enabled --quiet "$SERVICE_NAME"; then
        systemctl disable "$SERVICE_NAME"
        log_success "Автозапуск отключен"
    fi
}

# Удаление контейнеров
remove_containers() {
    log_info "Удаление Docker контейнеров..."
    
    if [ -d "$SERVICE_DIR" ]; then
        cd "$SERVICE_DIR"
        docker-compose down --remove-orphans --volumes || true
        docker system prune -f || true
        log_success "Контейнеры удалены"
    fi
}

# Удаление systemd сервиса
remove_systemd_service() {
    log_info "Удаление systemd сервиса..."
    
    if [ -f "$SERVICE_FILE" ]; then
        rm -f "$SERVICE_FILE"
        systemctl daemon-reload
        log_success "Systemd сервис удален"
    else
        log_info "Systemd сервис не найден"
    fi
}

# Удаление cron задач
remove_cron() {
    log_info "Удаление cron задач..."
    
    if [ -f "/etc/cron.d/face-comparison" ]; then
        rm -f "/etc/cron.d/face-comparison"
        log_success "Cron задачи удалены"
    else
        log_info "Cron задачи не найдены"
    fi
}

# Удаление файлов
remove_files() {
    log_info "Удаление файлов сервиса..."
    
    if [ -d "$SERVICE_DIR" ]; then
        read -p "Удалить директорию $SERVICE_DIR и все данные? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$SERVICE_DIR"
            log_success "Файлы сервиса удалены"
        else
            log_info "Файлы сервиса сохранены"
        fi
    else
        log_info "Директория сервиса не найдена"
    fi
    
    # Удаление логов
    if [ -d "/var/log/face-comparison" ]; then
        read -p "Удалить логи в /var/log/face-comparison? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "/var/log/face-comparison"
            log_success "Логи удалены"
        else
            log_info "Логи сохранены"
        fi
    fi
}

# Главная функция
main() {
    log_warning "Это действие необратимо удалит Face Comparison Service"
    read -p "Продолжить? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Удаление отменено"
        exit 0
    fi
    
    stop_service
    remove_containers
    remove_systemd_service
    remove_cron
    remove_files
    
    log_success "✅ Face Comparison Service полностью удален"
}

main "$@" 