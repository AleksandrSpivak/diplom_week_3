"""
CLIP Classifier for Stage 1: Binary Classification via Zero-Shot Scene Classification

Призначення:
    Бінарна класифікація зображень для 4 бізнес-категорій з використанням
    zero-shot підходу CLIP. використовує сцено-орієнтовану стратегію
    з абстрактними негативними промптами для контрастного порівняння.

Процес класифікації:
    Для кожної категорії:
    1. Використання пари [positive, negative] промптів
    2. Для кожного зображення:
       - Обчислення logits для обох промптів
       - Порівняння scores (argmax)
       - Визначення binary label (0 або 1)
    3. Збереження результатів

Пари промптів по категоріях:
    - Catering:
      [+] "a photo of prepared food, drinks, or a restaurant setting"
      [-] "a generic photo of an unrelated scene or object"
    - Marine_Activities:
      [+] "a photo of a marine or water activity, like a boat, sailing, or surfing"
      [-] "a generic photo of an unrelated scene or object"
    - Cultural_Excursions:
      [+] "a photo of a cultural landmark, monument, museum, or statue"
      [-] "a generic photo of an unrelated scene or object"
    - Pet-Friendly Services:
      [+] "a photo of a domestic pet, like a dog or a cat"
      [-] "a generic photo of an unrelated scene or object"

Вхідні дані:
    - data/stage_1/{category}/true/*.jpg - позитивні зразки (100 на категорію)
    - data/stage_1/{category}/neg/*.jpg - негативні зразки (100 на категорію)

Вихідні дані:
    - results/stage_1/results_clip.json - результати класифікації

Структура результату:
    {
        "Catering": [
            {
                "image": "cat_12345.jpg",
                "detected_objects": ["a photo of prepared food..."],
                "label": 1
            },
            ...
        ],
        ...
    }

Модель:
    - CLIP (openai/clip-vit-base-patch32)
    - Zero-shot image-text matching
    - Contrastive learning approach

"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class CLIPClassifier:
    def __init__(
        self, model_name="openai/clip-vit-base-patch32", model_path="weights/clip"
    ):
        """Ініціалізація CLIP моделі"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Завантаження моделі
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=str(model_dir))
        self.processor = CLIPProcessor.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model.to(self.device)

        # V4.0: Abstract Negative Prompts - тільки 4 категорії
        self.PROMPT_PAIRS = {
            "Catering": [
                "a photo of prepared food, drinks, or a restaurant setting",
                "a generic photo of an unrelated scene or object",
            ],
            "Marine_Activities": [
                "a photo of a marine or water activity, like a boat, sailing, or surfing",
                "a generic photo of an unrelated scene or object",
            ],
            "Cultural_Excursions": [
                "a photo of a cultural landmark, monument, museum, or statue",
                "a generic photo of an unrelated scene or object",
            ],
            "Pet-Friendly Services": [
                "a photo of a domestic pet, like a dog or a cat",
                "a generic photo of an unrelated scene or object",
            ],
        }

        self.output_path = "results/stage_1/results_clip.json"

    def classify_image(self, image_path, text_prompts):
        """Класифікація зображення"""
        image = Image.open(image_path)

        inputs = self.processor(
            text=text_prompts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Використовуємо "сирі" logits
            logits_per_image = outputs.logits_per_image

        return logits_per_image[0].cpu().numpy()

    def is_relevant(self, image_path, category):
        """Перевірка релевантності фото для категорії (v3.0 - Scene Classification)"""
        # 1. Отримуємо пару промптів
        text_prompts = self.PROMPT_PAIRS.get(category)
        if not text_prompts or len(text_prompts) != 2:
            return 0, []

        # 2. Отримуємо 2 scores (logits) для [Positive, Negative]
        scores = self.classify_image(image_path, text_prompts)

        score_positive = scores[0]
        score_negative = scores[1]

        # 3. Визначаємо лейбл
        label = 1 if score_positive > score_negative else 0

        # 4. Визначаємо "detected" (для сумісності зі звітом)
        detected = []
        if label == 1:
            detected.append(text_prompts[0])

        return label, detected

    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)

    def process_dataset(self, data_dir="data/stage_1"):
        """Обробка всього датасету"""
        results = {}

        for category in self.PROMPT_PAIRS.keys():
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
                label, detected = self.is_relevant(str(img_path), category)

                category_results.append(
                    {
                        "image": img_path.name,
                        "detected_objects": detected,
                        "label": label,
                    }
                )

                # Оновлюємо results та зберігаємо JSON після кожного фото
                results[category] = category_results
                self.save_results(results)

            print(f"Completed {category}: {len(category_results)} images processed")

        return results


if __name__ == "__main__":
    print("Starting CLIP classification (v3.0)...")
    classifier = CLIPClassifier()
    results = classifier.process_dataset()
    print("\nCLIP classification completed!")
    print(f"Results saved to: {classifier.output_path}")
