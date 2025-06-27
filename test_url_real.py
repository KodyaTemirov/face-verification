#!/usr/bin/env python3
"""
Тест API Face Comparison Service с реальными URL изображениями
"""
import requests
import json

def test_real_url_images():
    """Тестирует API с реальными URL изображениями"""
    base_url = "http://localhost:8000"
    
    print("🔍 Тестирование Face Comparison Service с реальными URL")
    print("=" * 55)
    
    # Реальные URL изображений лиц (публичные фото)
    face_urls = [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face",
        "https://images.unsplash.com/photo-1494790108755-2616b85eb8b6?w=400&h=400&fit=crop&crop=face"
    ]
    
    # Тест 1: Сравнение реальных лиц по URL
    print("\n🎯 Тест 1: Сравнение реальных лиц по URL")
    
    try:
        data = {
            "image1": face_urls[0],
            "image2": face_urls[1],
            "image1_type": "url",
            "image2_type": "url",
            "model": "ArcFace",
            "threshold": 0.5
        }
        
        print(f"URL 1: {face_urls[0]}")
        print(f"URL 2: {face_urls[1]}")
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=data,
            timeout=30
        )
        
        print(f"Статус ответа: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Успешно!")
            print(f"   Верификация: {result['verified']}")
            print(f"   Сходство: {result['similarity_percentage']:.2f}%")
            print(f"   Время обработки: {result['processing_time']}с")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    # Тест 2: Сравнение одного и того же лица
    print("\n🎯 Тест 2: Сравнение одного и того же URL")
    
    try:
        data = {
            "image1": face_urls[0],
            "image2": face_urls[0],  # Тот же URL
            "image1_type": "url",
            "image2_type": "url",
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
            print(f"✅ Успешно!")
            print(f"   Верификация: {result['verified']}")
            print(f"   Сходство: {result['similarity_percentage']:.2f}%")
        else:
            print(f"❌ Ошибка: {response.text}")
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
    
    # Тест 3: Некорректный URL
    print("\n🎯 Тест 3: Некорректный URL")
    
    try:
        data = {
            "image1": "https://httpbin.org/status/404",
            "image2": face_urls[0],
            "image1_type": "url",
            "image2_type": "url",
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
    test_real_url_images() 