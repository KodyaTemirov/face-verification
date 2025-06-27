#!/bin/bash

# Скрипт установки Face Comparison Service
# Файл: scripts/install-service.sh

set -e

echo "=== Установка Face Comparison Service ==="

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для цветного вывода
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

# Получаем директорию скрипта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_DIR="/opt/face-comparison-service"
SERVICE_NAME="face-comparison.service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"

log_info "Директория проекта: $PROJECT_DIR"
log_info "Целевая директория: $SERVICE_DIR"

# Проверка зависимостей
check_dependencies() {
    log_info "Проверка зависимостей..."
    
    local missing_deps=()
    
    # Проверка Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    elif ! systemctl is-active --quiet docker; then
        log_warning "Docker установлен, но не запущен"
        systemctl start docker
        systemctl enable docker
    fi
    
    # Проверка Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Проверка NVIDIA Docker (если есть GPU)
    if nvidia-smi &> /dev/null; then
        log_info "Обнаружен NVIDIA GPU"
        if ! docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi &> /dev/null; then
            log_warning "NVIDIA Docker runtime не настроен корректно"
            missing_deps+=("nvidia-docker2")
        fi
    else
        log_warning "NVIDIA GPU не обнаружен"
    fi
    
    # Проверка curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Отсутствуют зависимости: ${missing_deps[*]}"
        log_info "Установите их командой:"
        log_info "apt update && apt install -y ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "Все зависимости присутствуют"
}

# Создание пользователя и группы
setup_user() {
    log_info "Настройка пользователя сервиса..."
    
    # Определяем пользователя для сервиса
    local service_user=${SUDO_USER:-ubuntu}
    
    if ! id "$service_user" &>/dev/null; then
        log_info "Создание пользователя $service_user..."
        useradd -r -s /bin/bash -m "$service_user"
    fi
    
    # Добавляем пользователя в группу docker
    if ! groups "$service_user" | grep -q docker; then
        log_info "Добавление пользователя $service_user в группу docker..."
        usermod -aG docker "$service_user"
    fi
    
    echo "SERVICE_USER=$service_user" > /tmp/face-comparison-install-config
    log_success "Пользователь настроен: $service_user"
}

# Копирование файлов
install_files() {
    log_info "Установка файлов сервиса..."
    
    # Создание целевой директории
    mkdir -p "$SERVICE_DIR"
    
    # Копирование всех файлов проекта
    log_info "Копирование файлов проекта..."
    cp -r "$PROJECT_DIR"/* "$SERVICE_DIR/"
    
    # Создание необходимых директорий
    mkdir -p "$SERVICE_DIR"/{models,logs,temp}
    mkdir -p /var/log/face-comparison
    
    # Установка прав выполнения для скриптов
    chmod +x "$SERVICE_DIR/startup.sh"
    chmod +x "$SERVICE_DIR/healthcheck.sh"
    chmod +x "$SERVICE_DIR/scripts"/*.sh
    
    # Получение пользователя из конфига
    local service_user=$(grep SERVICE_USER /tmp/face-comparison-install-config | cut -d'=' -f2)
    
    # Установка прав доступа
    chown -R "$service_user:$service_user" "$SERVICE_DIR"
    chown -R "$service_user:$service_user" /var/log/face-comparison
    
    log_success "Файлы установлены в $SERVICE_DIR"
}

# Создание .env файла
create_env_file() {
    log_info "Создание файла конфигурации..."
    
    local env_file="$SERVICE_DIR/.env"
    
    # Создаем .env файл если не существует
    if [ ! -f "$env_file" ]; then
        cat > "$env_file" << EOF
# Face Comparison Service Configuration

# Telegram уведомления
TELEGRAM_BOT_TOKEN=7267358011:AAGMaQNXOcRuizEijEzri0AumB-T5yetmy4
TELEGRAM_CHAT_ID=979363701

# Slack уведомления (опционально)
# SLACK_WEBHOOK_URL=

# Email уведомления (опционально)
# EMAIL_WEBHOOK_URL=

# Логирование
LOG_LEVEL=INFO

# GPU настройки
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true

# Модели
MODEL_CACHE_DIR=/app/models

# Прочие настройки
PYTHONUNBUFFERED=1
COMPOSE_PROJECT_NAME=face_comparison
DOCKER_BUILDKIT=1
EOF
        
        local service_user=$(grep SERVICE_USER /tmp/face-comparison-install-config | cut -d'=' -f2)
        chown "$service_user:$service_user" "$env_file"
        log_success "Создан файл конфигурации: $env_file"
    else
        log_info "Файл конфигурации уже существует: $env_file"
    fi
}

# Установка systemd сервиса
install_systemd_service() {
    log_info "Установка systemd сервиса..."
    
    local service_user=$(grep SERVICE_USER /tmp/face-comparison-install-config | cut -d'=' -f2)
    
    # Создание файла сервиса с правильным пользователем
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Face Comparison Service with GPU Support
After=network.target docker.service nvidia-container-runtime.service
Requires=docker.service
Wants=nvidia-container-runtime.service

[Service]
Type=exec
User=$service_user
Group=docker
WorkingDirectory=$SERVICE_DIR
Environment="HOME=/home/$service_user"
EnvironmentFile=$SERVICE_DIR/.env
ExecStartPre=/bin/bash -c 'nvidia-smi > /dev/null 2>&1 || (echo "GPU not available" && exit 1)'
ExecStartPre=/usr/bin/docker-compose down
ExecStart=/usr/bin/docker-compose up --build
ExecStop=/usr/bin/docker-compose down
ExecReload=/usr/bin/docker-compose restart
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=face-comparison
KillMode=mixed
TimeoutStartSec=300
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF
    
    # Перезагрузка systemd
    systemctl daemon-reload
    
    log_success "Systemd сервис установлен: $SERVICE_FILE"
}

# Установка cron задач
install_cron() {
    log_info "Установка cron задач для мониторинга..."
    
    local service_user=$(grep SERVICE_USER /tmp/face-comparison-install-config | cut -d'=' -f2)
    local cron_file="/etc/cron.d/face-comparison"
    
    cat > "$cron_file" << EOF
# Face Comparison Service Health Check
# Проверка каждые 5 минут
*/5 * * * * $service_user $SERVICE_DIR/healthcheck.sh check >> /var/log/face-comparison/healthcheck-cron.log 2>&1

# Ротация логов раз в день
0 2 * * * $service_user find $SERVICE_DIR/logs -name "*.log" -type f -size +100M -exec logrotate -f /dev/stdin {} \; << EOL
$SERVICE_DIR/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 $service_user $service_user
}
EOL
EOF
    
    chmod 644 "$cron_file"
    log_success "Cron задачи установлены: $cron_file"
}

# Тестирование установки
test_installation() {
    log_info "Тестирование установки..."
    
    # Проверка файлов
    if [ ! -f "$SERVICE_DIR/startup.sh" ]; then
        log_error "Скрипт startup.sh не найден"
        return 1
    fi
    
    if [ ! -f "$SERVICE_DIR/docker-compose.yml" ]; then
        log_error "Файл docker-compose.yml не найден"
        return 1
    fi
    
    if [ ! -f "$SERVICE_FILE" ]; then
        log_error "Systemd сервис не установлен"
        return 1
    fi
    
    # Проверка синтаксиса docker-compose
    cd "$SERVICE_DIR"
    if ! docker-compose config > /dev/null 2>&1; then
        log_error "Ошибка конфигурации docker-compose"
        return 1
    fi
    
    log_success "Тестирование прошло успешно"
}

# Главная функция установки
main() {
    log_info "Начало установки Face Comparison Service..."
    
    check_dependencies
    setup_user
    install_files
    create_env_file
    install_systemd_service
    install_cron
    test_installation
    
    # Очистка временных файлов
    rm -f /tmp/face-comparison-install-config
    
    log_success "✅ Face Comparison Service успешно установлен!"
    echo
    log_info "Команды управления сервисом:"
    echo "  sudo systemctl start $SERVICE_NAME      # Запуск"
    echo "  sudo systemctl stop $SERVICE_NAME       # Остановка"
    echo "  sudo systemctl restart $SERVICE_NAME    # Перезапуск"
    echo "  sudo systemctl enable $SERVICE_NAME     # Автозапуск"
    echo "  sudo systemctl disable $SERVICE_NAME    # Отключить автозапуск"
    echo "  sudo systemctl status $SERVICE_NAME     # Статус"
    echo "  sudo journalctl -u $SERVICE_NAME -f     # Логи"
    echo
    log_info "Файлы сервиса:"
    echo "  Директория: $SERVICE_DIR"
    echo "  Конфигурация: $SERVICE_DIR/.env"
    echo "  Логи: /var/log/face-comparison/"
    echo
    log_info "Мониторинг:"
    echo "  $SERVICE_DIR/healthcheck.sh check       # Проверка здоровья"
    echo "  Health URL: http://localhost:8000/api/v1/health"
    echo
    log_warning "Для запуска сервиса выполните:"
    echo "  sudo systemctl enable $SERVICE_NAME"
    echo "  sudo systemctl start $SERVICE_NAME"
}

# Запуск установки
main "$@" 