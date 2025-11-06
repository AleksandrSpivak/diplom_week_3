"""
Profile Analyzer for Stage 4: Sensitivity Analysis - Raw Scores Generation

Призначення:
    Генерація "сирих" (raw) classification scores для всіх зображень у профілях
    експерименту чутливості. Ці scores використовуються для побудови кривої
    чутливості та визначення Reliable N.

Процес аналізу:
    1. Завантаження моделі SigLIP (тієї самої, що використовувалась на Етапах 2-3)
    2. Для кожного профілю (N=0..10, 100 повторень):
       - Обробка всіх 50 зображень профілю
       - Генерація scores для 5 класів (4 бізнес + Irrelevant)
       - Збереження детальних результатів з прив'язкою до N
    3. Інкрементальне збереження після кожного профілю

Вхідні дані:
    - data/stage_4/dataset_gt_{prefix}_100.json - Ground Truth профілів з N
    - data/stage_4/images/*.jpg - зображення профілів

Вихідні дані:
    - results/stage_4/analysis_results_{prefix}.json - детальні scores з N для кожного профілю

Структура результату:
    {
        "user_id": "cat_N05_rep01",
        "target_category": "Catering",
        "n_signals": 5,
        "total_images": 50,
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
    - Ті самі промпти, що на Етапах 2-3 (для порівнянності)
    - Sigmoid activation для незалежних scores

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

        Повертає: dict з user_id, target_category, n_signals, image_details
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
            "target_category": user_object["target_category"],
            "n_signals": user_object["n_signals"],
            "total_images": user_object["total_images"],
            "image_details": image_details,
        }


def main():
    print("=" * 60)
    print("STAGE 4: Profile Analysis (Raw Scores Generation)")
    print("=" * 60)

    # Шляхи
    dataset_gt_path = "data/stage_4/dataset_gt.json"
    images_dir = "data/stage_4/images"
    output_path = "results/stage_4/analysis_results.json"

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

    # Обробка профілів
    results = []

    print("\nAnalyzing profiles...")
    for user_object in tqdm(dataset, desc="Processing profiles", unit="profile"):
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
