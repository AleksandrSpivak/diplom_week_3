"""
Grounding DINO Detector for Stage 1: Binary Classification via Zero-Shot Object Detection

Призначення:
    Бінарна класифікація зображень для 4 бізнес-категорій з використанням
    zero-shot object detection підходу. Модель виконує локалізацію та детекцію
    об'єктів на основі текстових промптів (keywords) без попереднього навчання.

Підхід:
    1. Для кожної категорії визначено список "золотих" keywords (конкретні об'єкти)
    2. Keywords об'єднуються в текстовий промпт: "pizza . burger . sushi . ..."
    3. Grounding DINO виконує детекцію з локалізацією (bounding boxes)
    4. Якщо знайдено хоча б один об'єкт → relevance = 1
    5. Інакше → relevance = 0

Процес класифікації:
    Для кожної категорії:
    1. Завантаження Golden Keywords для категорії
    2. Формування text prompt (keywords через " . ")
    3. Для кожного зображення:
       - Zero-shot object detection з bounding boxes
       - Фільтрація за box_threshold=0.35 та text_threshold=0.25
       - Збір унікальних detected labels
       - Визначення binary label (0 або 1)
    4. Збереження результатів

Golden Keywords по категоріях:
    - Catering: pizza, burger, sushi, salad, dessert, cake, wine glass, coffee cup, cocktail
    - Marine_Activities: boat, yacht, surfboard, jet ski, kayak
    - Cultural_Excursions: museum, monument, statue, castle, fountain
    - Pet-Friendly Services: dog, cat, puppy, kitten

Вхідні дані:
    - data/stage_1/{category}/true/*.jpg - позитивні зразки (100 на категорію)
    - data/stage_1/{category}/neg/*.jpg - негативні зразки (100 на категорію)

Вихідні дані:
    - results/stage_1/results_grounding_dino.json - результати класифікації

Структура результату:
    {
        "Catering": [
            {
                "image": "cat_12345.jpg",
                "detected_objects": ["pizza", "wine glass"],
                "label": 1
            },
            ...
        ],
        ...
    }

Модель:
    - Grounding DINO Tiny (IDEA-Research/grounding-dino-tiny)
    - Zero-shot open-vocabulary object detection
    - Локалізація з bounding boxes
    - Thresholds: box_threshold=0.35, text_threshold=0.25

"""

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class GroundingDINODetector:
    def __init__(
        self,
        model_name="IDEA-Research/grounding-dino-tiny",
        model_path="weights/grounding_dino_tiny",
    ):
        """Ініціалізація Grounding DINO моделі"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Константи для threshold
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25

        # Завантаження моделі
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model.to(self.device)

        # V2.0: "Golden Keywords" - тільки 4 категорії
        self.GOLDEN_KEYWORDS = {
            # Конкретні страви/напої, а не "dining"
            "Catering": [
                "pizza",
                "burger",
                "sushi",
                "salad",
                "dessert",
                "cake",
                "wine glass",
                "coffee cup",
                "cocktail",
            ],
            # Конкретний транспорт/обладнання
            "Marine_Activities": ["boat", "yacht", "surfboard", "jet ski", "kayak"],
            # Конкретні архітектурні об'єкти (БЕЗ "Bust")
            "Cultural_Excursions": [
                "museum",
                "monument",
                "statue",
                "castle",
                "fountain",
            ],
            # Конкретні тварини (БЕЗ "Canary")
            "Pet-Friendly Services": ["dog", "cat", "puppy", "kitten"],
        }

        self.output_path = "results/stage_1/results_grounding_dino.json"

    def is_relevant(self, image_path, category):
        """Перевірка релевантності фото для категорії через zero-shot detection (v2.0)"""
        # 1. Отримуємо нові ключові слова
        relevant_objects = self.GOLDEN_KEYWORDS.get(category, [])

        if not relevant_objects:
            return 0, []

        # 2. Формуємо промпт
        text_prompt = " . ".join([obj.lower() for obj in relevant_objects]) + " ."

        # Завантаження зображення
        image = Image.open(image_path).convert("RGB")

        # Підготовка inputs
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Пост-обробка результатів
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        # Аналіз результатів
        detected_objects = []
        if len(results["scores"]) > 0:
            # Витягуємо labels знайдених об'єктів
            detected_labels = results["labels"]
            detected_objects = list(set(detected_labels))  # Унікальні labels

        # Визначаємо релевантність
        label = 1 if len(detected_objects) > 0 else 0

        return label, detected_objects

    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)

    def process_dataset(self, data_dir="data/stage_1"):
        """Обробка всього датасету"""
        results = {}

        for category in self.GOLDEN_KEYWORDS.keys():
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
    print("Starting Grounding DINO detection (v2.0)...")
    detector = GroundingDINODetector()
    results = detector.process_dataset()
    print("\nGrounding DINO detection completed!")
    print(f"Results saved to: {detector.output_path}")
