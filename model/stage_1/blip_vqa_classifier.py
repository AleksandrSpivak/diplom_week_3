"""
BLIP VQA Classifier for Stage 1: Binary Classification via Visual Question Answering

Призначення:
    Бінарна класифікація зображень для 4 бізнес-категорій з використанням
    підходу Visual Question Answering (VQA). Модель отримує зображення та
    питання, генерує текстову відповідь, яка аналізується на наявність
    ключових слів для визначення релевантності.

Підхід:
    1. Для кожної категорії визначено специфічне VQA-питання
    2. Модель отримує зображення + питання → генерує текстову відповідь
    3. Аналіз відповіді на наявність ключових слів (категорія-специфічних)
    4. Якщо знайдено ключове слово → relevance = 1 (True)
    5. Інакше → relevance = 0 (False)

Процес класифікації:
    Для кожної категорії:
    1. Використання предефінованого VQA-питання для категорії
    2. Для кожного зображення:
       - Генерація відповіді (max 20 tokens)
       - Пошук категорійних keywords у відповіді (case-insensitive)
       - Визначення binary label (0 або 1)
    3. Збереження результатів з питанням та відповіддю

VQA питання по категоріях:
    - Catering: "Does this photo show prepared food, drinks, or a restaurant setting?"
    - Marine_Activities: "What kind of marine or water activity is shown?"
    - Cultural_Excursions: "Does this photo show a museum, monument, statue, or castle?"
    - Pet-Friendly Services: "Is there a dog or a cat in this photo?"

Вхідні дані:
    - data/stage_1/{category}/true/*.jpg - позитивні зразки (100 на категорію)
    - data/stage_1/{category}/neg/*.jpg - негативні зразки (100 на категорію)

Вихідні дані:
    - results/stage_1/results_blip_vqa.json - результати класифікації

Структура результату:
    {
        "Catering": [
            {
                "image": "cat_12345.jpg",
                "vqa_question": "Does this photo show...",
                "vqa_answer": "yes food restaurant",
                "label": 1
            },
            ...
        ],
        ...
    }

Модель:
    - BLIP VQA (Salesforce/blip-vqa-base)

"""

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class BLIPVQAClassifier:
    def __init__(
        self, model_name="Salesforce/blip-vqa-base", model_path="weights/blip_vqa"
    ):
        """Ініціалізація BLIP VQA моделі"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Завантаження моделі
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.processor = BlipProcessor.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model.to(self.device)

        # VQA промпти для 4 категорій (v6.0: гібридна стратегія)
        self.VQA_PROMPTS = {
            "Catering": "Does this photo show prepared food, drinks, or a restaurant setting?",
            "Marine_Activities": "What kind of marine or water activity is shown?",
            "Cultural_Excursions": "Does this photo show a museum, monument, statue, or castle?",
            "Pet-Friendly Services": "Is there a dog or a cat in this photo?",
        }

        # Ключові слова для кожної категорії (v6.0)
        self.CATEGORY_KEYWORDS = {
            "Catering": [
                "yes",
                "food",
                "meal",
                "drink",
                "restaurant",
                "pizza",
                "burger",
                "sushi",
                "salad",
                "dessert",
                "cake",
                "ice cream",
                "wine",
                "beer",
                "coffee",
                "cocktail",
            ],
            "Marine_Activities": [
                "boat",
                "sailing",
                "fishing",
                "yacht",
                "surfboard",
                "surfing",
                "kayak",
                "jet ski",
                "snorkeling",
            ],
            "Cultural_Excursions": [
                "yes",
                "museum",
                "monument",
                "statue",
                "castle",
                "landmark",
            ],
            "Pet-Friendly Services": ["yes", "dog", "cat", "puppy", "kitten"],
        }

        self.output_path = "results/stage_1/results_blip_vqa.json"

    def is_relevant(self, image_path, category):
        """Перевірка релевантності фото для категорії через VQA"""
        question = self.VQA_PROMPTS.get(category)
        if not question:
            return 0, "", ""

        image = Image.open(image_path).convert("RGB")

        # Підготовка вхідних даних з питанням
        inputs = self.processor(image, text=question, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)

        answer = self.processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).lower()

        # Нова логіка: перевіряємо наявність ключових слів з категорії у відповіді
        keywords = self.CATEGORY_KEYWORDS.get(category, [])
        label = 1 if any(word.lower() in answer for word in keywords) else 0

        return label, question, answer

    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)

    def process_dataset(self, data_dir="data/stage_1"):
        """Обробка всього датасету"""
        results = {}

        for category in self.VQA_PROMPTS.keys():
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
                label, question, answer = self.is_relevant(str(img_path), category)

                category_results.append(
                    {
                        "image": img_path.name,
                        "vqa_question": question,
                        "vqa_answer": answer,
                        "label": label,
                    }
                )

                # Оновлюємо results та зберігаємо JSON після кожного фото
                results[category] = category_results
                self.save_results(results)

            print(f"Completed {category}: {len(category_results)} images processed")

        return results


if __name__ == "__main__":
    print("Starting BLIP VQA classification...")
    classifier = BLIPVQAClassifier()
    results = classifier.process_dataset()
    print("\nBLIP VQA classification completed!")
    print(f"Results saved to: {classifier.output_path}")
