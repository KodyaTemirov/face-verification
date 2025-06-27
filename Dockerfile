FROM python:3.9-slim

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Устанавливаем системные зависимости для InsightFace и OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libavcodec59 \
    libavformat59 \
    libswscale6 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    libatlas-base-dev \
    libtbb12 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app/ ./app/
COPY startup.sh .
COPY healthcheck.sh .

# Создаем необходимые директории
RUN mkdir -p /app/logs /app/models /app/temp

# Устанавливаем права на выполнение скриптов
RUN chmod +x startup.sh healthcheck.sh

# Открываем порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Команда запуска
CMD ["python", "app/main.py"] 