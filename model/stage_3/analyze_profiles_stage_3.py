"""
Profile Analyzer for Stage 3: Raw Scores Generation

Призначення:
    Генерація "сирих" (raw) classification scores для всіх зображень у профілях
    користувачів. Ці scores використовуються для подальшого аналізу порогів та
    валідації ієрархічної тегізації.

Процес аналізу:
    1. Завантаження моделі SigLIP (розробленої на Етапі 2)
    2. Для кожного профілю:
       - Обробка всіх зображень профілю
       - Генерація scores для 5 класів (4 бізнес + Irrelevant)
       - Збереження детальних результатів
    3. Інкрементальне збереження після кожного профілю (resume capability)

Вхідні дані:
    - data/stage_3/dataset_gt.json - Ground Truth профілів
    - data/stage_3/images/*.jpg - зображення профілів

Вихідні дані:
    - results/stage_3/analysis_results.json - детальні scores для кожного зображення

Структура результату:
    {
        "user_id": "hobbyist_0001",
        "gt_vector": {"Catering": 0.55, "Marine_Activities": 0.20, ...},
        "image_details": [
            {
                "filename": "cat_12345.jpg",
                "scores": {"Catering": 0.456, "Marine_Activities": 0.012, ...},
                "gt_original_classes": ["Food", "Restaurant"]
            },
            ...
        ]
    }

Модель:
    - SigLIP (google/siglip-base-patch16-224)
    - Sigmoid activation для незалежних scores
    - Фіксовані промпти для 5 класів

Особливості:
    - Підтримка resume (продовження обробки після перерви)
    - Збереження прогресу після кожного профілю
    - GPU acceleration (якщо доступно)

"""

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class ProfileAnalyzer:
    def __init__(
        self, model_name="google/siglip-base-patch16-224", model_path="weights/siglip"
    ):
        """Ініціалізація ProfileAnalyzer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Завантаження SigLIP моделі
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model.to(self.device)

        # Промпти для 5 класів
        self.CLASS_PROMPTS = {
            "Catering": "a photo of food or drinks, restaurant, dining",
            "Marine_Activities": "a photo of watercraft, boat, jet ski, surfing, beach activities",
            "Cultural_Excursions": "a photo of cultural landmark, monument, museum, castle, sculpture",
            "Pet-Friendly Services": "a photo of domestic pet, dog, cat, puppy, kitten",
            "Irrelevant": "a photo unrelated to business services",
        }

    def get_image_scores(self, image_path):
        """
        Отримати scores для всіх 5 класів для одного зображення
        Повертає: dict {"Catering": 0.45, "Marine_Activities": 0.12, ...}
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return {class_name: 0.0 for class_name in self.CLASS_PROMPTS.keys()}

        scores = {}

        for class_name, prompt in self.CLASS_PROMPTS.items():
            inputs = self.processor(
                text=[prompt], images=image, padding="max_length", return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                prob = torch.sigmoid(logits_per_image)

            scores[class_name] = prob[0][0].cpu().item()

        return scores

    def process_profile(self, user_object, images_dir):
        """
        Обробка одного профілю користувача

        Повертає: dict з user_id, gt_vector, image_details
        """
        image_details = []

        # Обробка кожного зображення в профілі
        for image_item in user_object["image_data"]:
            filename = image_item["filename"]
            image_path = Path(images_dir) / filename

            if not image_path.exists():
                print(f"Warning: {filename} not found")
                continue

            # Отримання scores для зображення
            scores = self.get_image_scores(str(image_path))

            # Збереження детальної інформації
            image_details.append(
                {
                    "filename": filename,
                    "scores": {k: round(v, 8) for k, v in scores.items()},
                    "gt_original_classes": image_item["original_classes"],
                }
            )

        # Формування результату
        return {
            "user_id": user_object["user_id"],
            "gt_vector": user_object["gt_vector"],
            "image_details": image_details,
        }


def main():
    print("=" * 60)
    print("STAGE 3: Profile Analysis (Raw Scores Generation)")
    print("=" * 60)

    # Шляхи
    dataset_gt_path = "data/stage_3/dataset_gt.json"
    images_dir = "data/stage_3/images"
    output_path = "results/stage_3/analysis_results.json"

    # Створення директорії для результатів
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Ініціалізація аналізатора
    print("\nInitializing ProfileAnalyzer...")
    analyzer = ProfileAnalyzer()

    # Завантаження датасету
    print(f"\nLoading dataset from: {dataset_gt_path}")
    with open(dataset_gt_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Total profiles to analyze: {len(dataset)}")

    # Завантаження попереднього прогресу
    results = []
    processed_ids = set()

    try:
        if Path(output_path).exists():
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            for result in results:
                processed_ids.add(result["user_id"])
            print(f"Loaded {len(results)} previously processed profiles")
    except Exception as e:
        print(f"Warning: Could not load previous results: {e}")
        results = []
        processed_ids = set()

    # Обробка профілів
    print("\nAnalyzing profiles...")
    for user_object in tqdm(dataset, desc="Processing profiles", unit="profile"):
        if user_object["user_id"] in processed_ids:
            continue

        result = analyzer.process_profile(user_object, images_dir)
        results.append(result)

        # Зберігати після кожного профілю
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total profiles analyzed: {len(results)}")


if __name__ == "__main__":
    main()
