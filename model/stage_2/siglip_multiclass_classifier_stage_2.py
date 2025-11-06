"""
SigLIP Multiclass Classifier for Stage 2

Призначення:
    Мультикласова класифікація зображень на 5 класів (4 бізнес-категорії + Irrelevant)
    з використанням pretrained SigLIP моделі. Для кожного зображення обирається клас
    з найвищим sigmoid score (argmax підхід).

Процес класифікації:
    1. Завантаження моделі SigLIP (google/siglip-base-patch16-224)
    2. Для кожного зображення:
       - Генерація scores для всіх 5 промптів
       - Вибір класу з найвищим score (argmax)
       - Збереження predicted class та всіх scores
    3. Інкрементальне збереження результатів

Вхідні дані:
    - data/stage_2/ground_truth.json - Ground Truth анотації
    - data/stage_2/*.jpg - зображення датасету (2500 шт, 500 на клас)

Вихідні дані:
    - results/stage_2/results_siglip_multiclass.json - результати класифікації

Структура результату:
    [
        {
            "image": "cat_12345.jpg",
            "true_label": "Catering",
            "predicted_label": "Catering",
            "scores": {
                "Catering": 0.856234,
                "Marine_Activities": 0.012345,
                "Cultural_Excursions": 0.045678,
                "Pet-Friendly Services": 0.001234,
                "Irrelevant": 0.123456
            }
        },
        ...
    ]

Модель:
    - SigLIP base-patch16-224
    - Sigmoid activation (незалежні scores для кожного класу)
    - Фіксовані текстові промпти для 5 класів
    - Argmax для фінального рішення

"""

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class SigLIPMulticlassClassifier:
    def __init__(
        self, model_name="google/siglip-base-patch16-224", model_path="weights/siglip"
    ):
        """Ініціалізація SigLIP моделі"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Завантаження моделі
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

        self.output_path = "results/stage_2/results_siglip_multiclass.json"

    def classify_image(self, image_path, text_prompt):
        """
        Класифікація зображення з SigLIP
        Повертає sigmoid probability (0-1) для пари image-text
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return 0.0

        inputs = self.processor(
            text=[text_prompt], images=image, padding="max_length", return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            prob = torch.sigmoid(logits_per_image)

        return prob[0][0].cpu().item()

    def predict_class(self, image_path):
        """
        Передбачення класу для зображення
        Повертає: (predicted_class, scores_dict)
        """
        scores = {}

        for class_name, prompt in self.CLASS_PROMPTS.items():
            score = self.classify_image(image_path, prompt)
            scores[class_name] = score

        # Обрати клас з найвищим score
        predicted_class = max(scores, key=scores.get)

        return predicted_class, scores

    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def process_dataset(
        self,
        data_dir="data/stage_2",
        ground_truth_path="data/stage_2/ground_truth.json",
    ):
        """Обробка датасету"""
        # Завантажити ground truth
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        results = []
        data_path = Path(data_dir)

        print(f"\nProcessing {len(ground_truth)} images...")

        for filename, gt_data in tqdm(
            ground_truth.items(), desc="Classifying", unit="img"
        ):
            img_path = data_path / filename

            if not img_path.exists():
                print(f"Warning: {filename} not found")
                continue

            predicted_class, scores = self.predict_class(str(img_path))

            results.append(
                {
                    "image": filename,
                    "true_label": gt_data["label"],
                    "predicted_label": predicted_class,
                    "scores": {k: round(v, 6) for k, v in scores.items()},
                }
            )

            # Зберігати після кожного зображення
            self.save_results(results)

        return results


if __name__ == "__main__":
    print("Starting SigLIP Multiclass classification...")

    classifier = SigLIPMulticlassClassifier()
    results = classifier.process_dataset()

    print(f"\nClassification completed!")
    print(f"Results saved to: {classifier.output_path}")
    print(f"Total images processed: {len(results)}")
