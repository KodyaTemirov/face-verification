#!/usr/bin/env python3
import requests
import json

# Простой тест с печатью всех данных
data = {
    "image1": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100",
    "image2": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAACAAIDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/AAAAAAAA=",
    "image1_type": "url",
    "image2_type": "base64",
    "model": "ArcFace"
}

print("Отправляю запрос с данными:")
print(json.dumps(data, indent=2))

response = requests.post(
    "http://localhost:8000/api/v1/compare",
    json=data,
    timeout=30
)

print(f"\nСтатус ответа: {response.status_code}")
print(f"Ответ: {response.text}") 