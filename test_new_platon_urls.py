#!/usr/bin/env python3
"""
Тест API с новыми URL изображениями от пользователя
"""
import requests
import json
import time

def test_new_platon_urls():
    """Тестирует API с новыми предоставленными URL"""
    base_url = "http://localhost:8000"
    
    print("🔍 Тестирование Face Comparison Service с новыми URL")
    print("=" * 60)
    
    # Новые URL изображений от пользователя
    url1 = "https://proctoring.platon.uz/media/images/user_image/ff808181efa3b9499f2a071b.jpeg"
    url2 = "https://proctoring.platon.uz/media/images/user_image/ff8081812aebb26799faf70d.jpeg"
    
    # Проверка health сервиса
    print("\n🏥 Проверка статуса сервиса...")
    try:
        health_response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Сервис работает: {health_data.get('status', 'unknown')}")
            print(f"   Модель загружена: {health_data.get('models_loaded', False)}")
        else:
            print(f"❌ Проблема с сервисом: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Сервис недоступен: {e}")
        return
    
    # Тест 1: Сравнение двух новых URL
    print(f"\n🎯 Тест 1: Сравнение двух новых URL изображений")
    print(f"URL 1: {url1}")
    print(f"URL 2: {url2}")
    
    payload1 = {
        "image1": url1,
        "image2": url2,
        "image1_type": "url",
        "image2_type": "url",
        "model": "ArcFace",
        "metric": "cosine"
    }
    
    try:
        start_time = time.time()
        print("Отправляю запрос...")
        
        response1 = requests.post(
            f"{base_url}/api/v1/compare",
            json=payload1,
            timeout=60
        )
        
        request_time = time.time() - start_time
        print(f"Время запроса: {request_time:.2f}с")
        print(f"Статус ответа: {response1.status_code}")
        
        if response1.status_code == 200:
            data1 = response1.json()
            print("✅ Успешно!")
            print(f"   Верификация: {data1.get('verified', 'N/A')}")
            similarity = data1.get('similarity', 0)
            print(f"   Сходство: {similarity:.2%}")
            print(f"   Дистанция: {data1.get('distance', 'N/A'):.4f}")
            print(f"   Время обработки: {data1.get('processing_time', 'N/A'):.3f}с")
            print(f"   Модель: {data1.get('model_used', 'N/A')}")
            
            # Информация о найденных лицах
            faces_info = data1.get('faces_detected', {})
            print(f"   Лица найдены: {{'image1': {faces_info.get('image1', 0)}, 'image2': {faces_info.get('image2', 0)}}}")
            
        else:
            print("❌ Ошибка!")
            try:
                error_data = response1.json()
                print(f"   Детали: {error_data}")
            except:
                print(f"   HTTP Error: {response1.status_code}")
                print(f"   Ответ: {response1.text[:200]}")
                
    except requests.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
    
    # Тест 2: Сравнение первого URL с самим собой (должно дать 100% сходство)
    print(f"\n🎯 Тест 2: Сравнение изображения с самим собой")
    print(f"URL: {url1}")
    
    payload2 = {
        "image1": url1,
        "image2": url1,
        "image1_type": "url",
        "image2_type": "url",
        "model": "ArcFace",
        "metric": "cosine"
    }
    
    try:
        response2 = requests.post(
            f"{base_url}/api/v1/compare",
            json=payload2,
            timeout=60
        )
        
        print(f"Статус ответа: {response2.status_code}")
        
        if response2.status_code == 200:
            data2 = response2.json()
            print("✅ Успешно!")
            print(f"   Верификация: {data2.get('verified', 'N/A')}")
            similarity = data2.get('similarity', 0)
            print(f"   Сходство: {similarity:.2%}")
            print(f"   Дистанция: {data2.get('distance', 'N/A'):.4f}")
        else:
            print("❌ Ошибка!")
            try:
                error_data = response2.json()
                print(f"   Детали: {error_data}")
            except:
                print(f"   HTTP Error: {response2.status_code}")
                
    except requests.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

    print(f"\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_new_platon_urls() 