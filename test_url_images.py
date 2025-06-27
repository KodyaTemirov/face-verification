#!/usr/bin/env python3
"""
Тест API Face Comparison Service с URL изображениями
"""
import requests
import json

def test_url_images():
    """Тестирует API с URL изображениями"""
    base_url = "http://localhost:8000"
    
    print("🔍 Тестирование Face Comparison Service с URL изображениями")
    print("=" * 60)
    
    # Тестовые URL изображений (публичные фото)
    # Используем httpbin для создания простых изображений
    test_urls = [
        "https://httpbin.org/image/jpeg",
        "https://httpbin.org/image/png",
    ]
    
    # Тест 1: Проверка загрузки по URL
    print("\n🎯 Тест 1: Сравнение изображений по URL")
    
    try:
        data = {
            "image1": test_urls[0],
            "image2": test_urls[1], 
            "image1_type": "url",
            "image2_type": "url",
            "model": "ArcFace",
            "threshold": 0.5
        }
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=30
        )
        
        print(f"Статус ответа: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Успешно: {result}")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    # Тест 2: Смешанный тест (URL + base64)
    print("\n🎯 Тест 2: Смешанный тест (URL + base64)")
    
    # Простое base64 изображение (красный квадрат 1x1)
    simple_base64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/AAAAAAAA=="
    
    try:
        data = {
            "image1": test_urls[0],
            "image2": simple_base64,
            "image1_type": "url", 
            "image2_type": "base64",
            "model": "ArcFace"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=30
        )
        
        print(f"Статус ответа: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Успешно: {result}")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    # Тест 3: Тест с несуществующим URL
    print("\n🎯 Тест 3: Тест с несуществующим URL")
    
    try:
        data = {
            "image1": "https://example.com/nonexistent.jpg",
            "image2": simple_base64,
            "image1_type": "url",
            "image2_type": "base64",
            "model": "ArcFace"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=30
        )
        
        print(f"Статус ответа: {response.status_code}")
        print(f"Ответ: {response.text}")
        
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_url_images() 