#!/usr/bin/env python3
"""
Тест улучшенной детекции лиц
"""
import requests
import json

def test_enhanced_detection():
    """Простой тест улучшенной детекции"""
    base_url = "http://localhost:8000"
    
    print("🔍 Тестирование улучшенной детекции лиц")
    print("=" * 50)
    
    # URL от пользователя
    url1 = "https://proctoring.platon.uz/media/images/user_image/ff8081814a91b327eb08c2bf.jpeg"
    url2 = "https://proctoring.platon.uz/media/images/user_image/ff808181ef43402eb9d72e3c.jpeg"
    
    payload = {
        "image1": url1,
        "image2": url2,
        "image1_type": "url",
        "image2_type": "url",
        "model": "ArcFace",
        "metric": "cosine"
    }
    
    try:
        print("\n🎯 Тестирование пары URL изображений...")
        print(f"URL 1: {url1}")
        print(f"URL 2: {url2}")
        
        response = requests.post(
            f"{base_url}/api/v1/compare",
            json=payload,
            timeout=30
        )
        
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Успешно!")
            print(f"   Сходство: {result.get('similarity', 'N/A'):.3f}")
            print(f"   Расстояние: {result.get('distance', 'N/A'):.3f}")
            print(f"   Верифицировано: {result.get('verified', 'N/A')}")
            print(f"   Время обработки: {result.get('processing_time', 'N/A')}с")
        else:
            print("❌ Ошибка!")
            try:
                error = response.json()
                print(f"   Детали: {error}")
            except:
                print(f"   Текст ошибки: {response.text}")
                
    except Exception as e:
        print(f"❌ Исключение: {e}")

if __name__ == "__main__":
    test_enhanced_detection() 