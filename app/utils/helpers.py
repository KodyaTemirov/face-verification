import re
import hashlib
from urllib.parse import urlparse
from typing import Optional, List
import structlog

logger = structlog.get_logger(__name__)


def validate_image_url(url: str) -> bool:
    """Валидация URL изображения"""
    try:
        # Проверка формата URL
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        
        # Проверка допустимых схем
        if result.scheme not in ('http', 'https'):
            return False
        
        # Проверка расширения файла
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
        path_lower = result.path.lower()
        
        # Проверка расширения в URL или content-type будет проверен при загрузке
        has_valid_extension = any(path_lower.endswith(ext) for ext in valid_extensions)
        
        return True  # Разрешаем URL без расширения, проверим при загрузке
        
    except Exception as e:
        logger.error(f"Ошибка валидации URL: {e}")
        return False


def sanitize_url(url: str) -> Optional[str]:
    """Санитизация URL для безопасности"""
    try:
        # Базовая очистка
        url = url.strip()
        
        # Удаление потенциально опасных символов
        dangerous_patterns = [
            r'[<>"\'\(\){}]',  # HTML/script injection
            r'javascript:',     # JavaScript injection
            r'data:',          # Data URLs
            r'file:',          # File URLs
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.warning(f"Обнаружен потенциально опасный URL: {url}")
                return None
        
        # Проверка на внутренние сети (опционально)
        parsed = urlparse(url)
        if parsed.hostname:
            # Блокировка localhost и внутренних IP
            internal_hosts = [
                'localhost', '127.0.0.1', '0.0.0.0',
                '::1', '[::]'
            ]
            
            if parsed.hostname.lower() in internal_hosts:
                logger.warning(f"Заблокирован внутренний хост: {parsed.hostname}")
                return None
            
            # Проверка на внутренние сети (10.x.x.x, 192.168.x.x, 172.16-31.x.x)
            if re.match(r'^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)', parsed.hostname):
                logger.warning(f"Заблокирован внутренний IP: {parsed.hostname}")
                return None
        
        return url
        
    except Exception as e:
        logger.error(f"Ошибка санитизации URL: {e}")
        return None


def generate_cache_key(data: str) -> str:
    """Генерация ключа для кэширования"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]


def truncate_url_for_logging(url: str, max_length: int = 100) -> str:
    """Сокращение URL для логирования"""
    if len(url) <= max_length:
        return url
    
    return f"{url[:max_length-3]}..."


def extract_filename_from_url(url: str) -> Optional[str]:
    """Извлечение имени файла из URL"""
    try:
        parsed = urlparse(url)
        path = parsed.path
        
        if '/' in path:
            filename = path.split('/')[-1]
            return filename if filename else None
        
        return None
        
    except Exception:
        return None


def is_safe_content_type(content_type: str) -> bool:
    """Проверка безопасности content-type"""
    safe_types = [
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/gif',
        'image/bmp',
        'image/webp'
    ]
    
    return content_type.lower() in safe_types


def format_file_size(size_bytes: int) -> str:
    """Форматирование размера файла"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def validate_threshold(threshold: float) -> bool:
    """Валидация порога схожести"""
    return 0.0 <= threshold <= 1.0


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Маскирование чувствительных данных для логов"""
    if len(data) <= visible_chars * 2:
        return mask_char * len(data)
    
    return f"{data[:visible_chars]}{mask_char * (len(data) - visible_chars * 2)}{data[-visible_chars:]}" 