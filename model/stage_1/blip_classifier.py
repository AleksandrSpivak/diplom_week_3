"""
BLIP Image Captioning Classifier for Stage 1: Binary Classification

Призначення:
    Бінарна класифікація зображень для 4 бізнес-категорій з використанням
    підходу Image-to-Text (Image Captioning). Модель генерує текстовий опис
    зображення, який потім аналізується на наявність ключових слів.

Підхід:
    1. Генерація caption (текстового опису) для зображення через BLIP
    2. Пошук ключових слів (keywords) у згенерованому тексті
    3. Якщо знайдено хоча б одне ключове слово → relevance = 1 (True)
    4. Інакше → relevance = 0 (False)

Процес класифікації:
    Для кожної категорії:
    1. Завантаження списку релевантних об'єктів з available_business_categories.json
    2. Для кожного зображення:
       - Генерація caption (BLIP conditional generation)
       - Пошук keywords у caption (case-insensitive)
       - Визначення binary label (0 або 1)
    3. Збереження результатів з detected objects та caption

Вхідні дані:
    - data/stage_1/{category}/true/*.jpg - позитивні зразки 
    - data/stage_1/{category}/neg/*.jpg - негативні зразки 
    - available_business_categories.json - списки ключових слів для категорій

Вихідні дані:
    - results/stage_1/results_blip.json - результати класифікації

Структура результату:
    {
        "Catering": [
            {
                "image": "cat_12345.jpg",
                "detected_objects": ["wine", "food"],
                "caption": "a bottle of wine and a plate of food on a table",
                "label": 1
            },
            ...
        ],
        ...
    }

Модель:
    - BLIP (Salesforce/blip-image-captioning-base)

"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class BLIPClassifier:
    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base",
        model_path="weights/blip",
    ):
        """Ініціалізація BLIP моделі"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Завантаження моделі
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.processor = BlipProcessor.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model.to(self.device)

        # Завантаження категорій з JSON - тільки 4 категорії
        with open("available_business_categories.json", "r", encoding="utf-8") as f:
            all_categories = json.load(f)

        self.business_categories = {
            "Pet-Friendly Services": all_categories["Pet-Friendly Services"],
            "Catering": all_categories["Catering"],
            "Marine_Activities": all_categories["Marine_Activities"],
            "Cultural_Excursions": all_categories["Cultural_Excursions"],
        }

        self.output_path = "results/stage_1/results_blip.json"

    def classify_image(self, image_path, objects):
        """Класифікація зображення за наявністю об'єктів"""
        image = Image.open(image_path).convert("RGB")

        detected = []

        # Генеруємо caption для зображення
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).lower()

        # Перевіряємо чи згадуються об'єкти в caption
        for obj in objects:
            if obj.lower() in caption:
                detected.append(obj)

        return detected, caption

    def is_relevant(self, image_path, category):
        """Перевірка релевантності фото для категорії"""
        relevant_objects = self.business_categories.get(category, [])
        if not relevant_objects:
            return 0, [], ""

        detected, caption = self.classify_image(image_path, relevant_objects)
        label = 1 if len(detected) > 0 else 0

        return label, detected, caption

    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)

    def process_dataset(self, data_dir="data/stage_1"):
        """Обробка всього датасету"""
        results = {}

        for category in self.business_categories.keys():
            category_path = Path(data_dir) / category
            if not category_path.exists():
                print(f"Skipping {category} - directory not found")
                continue

            print(f"\nProcessing category: {category}")
            category_results = []
            image_files = list(category_path.glob("*.jpg")) + list(
                category_path.glob("*.png")
            )

            # Progress bar для фото в категорії
            for img_path in tqdm(image_files, desc=f"{category}", unit="img"):
                label, detected, caption = self.is_relevant(str(img_path), category)

                category_results.append(
                    {
                        "image": img_path.name,
                        "detected_objects": detected,
                        "caption": caption,
                        "label": label,
                    }
                )

                # Оновлюємо results та зберігаємо JSON після кожного фото
                results[category] = category_results
                self.save_results(results)

            print(f"Completed {category}: {len(category_results)} images processed")

        return results


if __name__ == "__main__":
    print("Starting BLIP classification...")
    classifier = BLIPClassifier()
    results = classifier.process_dataset()
    print("\nBLIP classification completed!")
    print(f"Results saved to: {classifier.output_path}")
