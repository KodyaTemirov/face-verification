# Автозапуск Face Verification Service

Данная инструкция описывает настройку автоматического запуска сервиса при включении сервера.

## ✅ Настройка выполнена

Автозапуск уже настроен для текущего сервера через **crontab**:

```bash
# Проверить настройку автозапуска
crontab -l
```

Должно отображаться:
```bash
@reboot /home/kodyatemirov/face-verification/start-face-verification.sh
```

## 📁 Файлы автозапуска

### 1. `start-face-verification.sh`
- **Назначение**: Основной скрипт автозапуска
- **Функции**: 
  - Ожидание готовности Docker
  - Автоматический запуск контейнеров
  - Логирование процесса запуска
  - Проверка состояния сервиса

### 2. `manage-service.sh` 
- **Назначение**: Скрипт управления сервисом
- **Команды**:
  ```bash
  ./manage-service.sh start    # Запустить сервис
  ./manage-service.sh stop     # Остановить сервис  
  ./manage-service.sh restart  # Перезапустить сервис
  ./manage-service.sh status   # Показать состояние
  ./manage-service.sh logs     # Показать логи
  ./manage-service.sh health   # Проверить API
  ```

### 3. `face-verification.service` (systemd)
- **Назначение**: systemd unit файл (альтернативный вариант)
- **Статус**: Подготовлен, но не установлен (требует sudo)

## 🚀 Как это работает

1. **При перезагрузке сервера**:
   - Система запускает Docker
   - cron выполняет скрипт `start-face-verification.sh`
   - Скрипт ожидает готовности Docker (до 30 попыток)
   - Запускается Face Verification Service
   - Логи записываются в `startup.log`

2. **Проверка состояния**:
   ```bash
   # Быстрая проверка
   ./manage-service.sh status
   
   # Детальная проверка API
   ./manage-service.sh health
   ```

## 📋 Логирование

### Логи автозапуска
```bash
tail -f /home/kodyatemirov/face-verification/startup.log
```

### Логи приложения
```bash
docker-compose logs -f
```

### Все логи через скрипт управления
```bash
./manage-service.sh logs
```

## 🔧 Настройка на новом сервере

Если нужно настроить автозапуск на другом сервере:

### Вариант 1: Через crontab (рекомендуется)
```bash
# Перейти в директорию проекта
cd /path/to/face-verification

# Сделать скрипты исполняемыми
chmod +x start-face-verification.sh
chmod +x manage-service.sh

# Добавить в crontab
(crontab -l 2>/dev/null; echo "@reboot $(pwd)/start-face-verification.sh") | crontab -

# Проверить
crontab -l
```

### Вариант 2: Через systemd (требует root)
```bash
# Скопировать unit файл
sudo cp face-verification.service /etc/systemd/system/

# Обновить пути в файле при необходимости
sudo nano /etc/systemd/system/face-verification.service

# Активировать сервис
sudo systemctl daemon-reload
sudo systemctl enable face-verification.service
sudo systemctl start face-verification.service

# Проверить статус
sudo systemctl status face-verification.service
```

## ⚠️ Важные заметки

1. **Docker должен быть настроен для автозапуска**:
   ```bash
   sudo systemctl enable docker
   ```

2. **Пользователь должен быть в группе docker**:
   ```bash
   sudo usermod -aG docker $USER
   ```

3. **Проверка после перезагрузки**:
   ```bash
   # Через 5-10 минут после перезагрузки
   ./manage-service.sh status
   curl http://localhost:8000/api/v1/health
   ```

## 🔍 Диагностика проблем

### Если сервис не запускается автоматически:

1. **Проверить crontab**:
   ```bash
   crontab -l
   ```

2. **Проверить логи автозапуска**:
   ```bash
   cat startup.log
   ```

3. **Проверить Docker**:
   ```bash
   docker info
   systemctl status docker
   ```

4. **Запустить вручную**:
   ```bash
   ./start-face-verification.sh
   ```

5. **Проверить разрешения**:
   ```bash
   ls -la start-face-verification.sh
   ls -la manage-service.sh
   ```

## 📞 Поддержка

Для решения проблем с автозапуском:
1. Проверьте логи: `./manage-service.sh logs`
2. Запустите диагностику: `./manage-service.sh health`
3. Проверьте настройки в данном файле 