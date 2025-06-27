#!/usr/bin/env python3
"""
Тест API с URL изображениями от пользователя
"""
import requests
import json
import time

def test_platon_urls():
    """Тестирует API с предоставленными URL"""
    base_url = "http://localhost:8000"
    
    print("🔍 Тестирование Face Comparison Service с URL от пользователя")
    print("=" * 60)
    
    # URL изображений от пользователя
    url1 = "https://proctoring.platon.uz/media/images/user_image/ff8081814a91b327eb08c2bf.jpeg"
    url2 = "https://proctoring.platon.uz/media/images/user_image/ff808181ef43402eb9d72e3c.jpeg"
    
    # Проверка health сервиса
    print("\n🏥 Проверка статуса сервиса...")
    try:
        health_response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Сервис работает: {health_data['status']}")
            print(f"   Модель загружена: {health_data['models_loaded']}")
        else:
            print(f"❌ Сервис недоступен: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Ошибка подключения к сервису: {e}")
        return
    
    # Тест 1: Сравнение URL изображений
    print(f"\n🎯 Тест 1: Сравнение двух разных лиц")
    print(f"URL 1: {url1}")
    print(f"URL 2: {url2}")
    
    try:
        data = {
            "image1": url1,
            "image2": url2,
            "image1_type": "url",
            "image2_type": "url",
            "model": "ArcFace",
            "threshold": 0.6
        }
        
        print("Отправляю запрос...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=60
        )
        
        request_time = time.time() - start_time
        print(f"Время запроса: {request_time:.2f}с")
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Успешно!")
            print(f"   Верификация: {result['verified']}")
            print(f"   Сходство: {result['similarity_percentage']:.2f}%")
            print(f"   Дистанция: {result['distance']:.4f}")
            print(f"   Время обработки: {result['processing_time']}с")
            print(f"   Модель: {result['model']}")
            print(f"   Лица найдены: {result['faces_detected']}")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    # Тест 2: Сравнение одного изображения с собой
    print(f"\n🎯 Тест 2: Сравнение изображения с собой")
    
    try:
        data = {
            "image1": url1,
            "image2": url1,  # Тот же URL
            "image1_type": "url",
            "image2_type": "url",
            "model": "ArcFace"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=60
        )
        
        print(f"Статус ответа: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Успешно!")
            print(f"   Верификация: {result['verified']}")
            print(f"   Сходство: {result['similarity_percentage']:.2f}%")
            print(f"   Дистанция: {result['distance']:.4f}")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    # Тест 3: Смешанный тест (URL + base64)
    print(f"\n🎯 Тест 3: Смешанный тест (URL + base64)")
    
    # Простое тестовое base64 изображение (маленький квадрат)
    test_base64 = "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAACAAIDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/AAAAAAAA="
    
    try:
        data = {
            "image1": url1,
            "image2": test_base64,
            "image1_type": "url",
            "image2_type": "base64",
            "model": "ArcFace"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=60
        )
        
        print(f"Статус ответа: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Успешно!")
            print(f"   Результат: {result}")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_platon_urls() 