"""
SigLIP Classifier V2 for Stage 1: Binary Classification with Prompt Ensembling

Призначення:
    Бінарна класифікація зображень для 4 бізнес-категорій з використанням
    SigLIP моделі та техніки Prompt Ensembling. V2 покращує Recall шляхом
    використання множинних промптів на категорію та вибору максимального score.

Підхід (V2 - Prompt Ensembling):
    1. Для кожної категорії визначено список з 2-4 варіативних промптів
    2. Для кожного зображення:
       - Обчислення sigmoid probability для КОЖНОГО промпту
       - Вибір максимального score: max_score = max(scores_list)
       - Порівняння max_score з threshold
    3. Якщо max_score > threshold → relevance = 1
    4. Інакше → relevance = 0

Еволюція версій:
    - V1: Один промпт на категорію, низький Recall через сцено-орієнтовані промпти
    - V2: Ансамбль об'єктно-орієнтованих промптів + max aggregation

Процес класифікації:
    Для кожної категорії:
    1. Завантаження ансамблю промптів з POSITIVE_PROMPTS
    2. Для кожного зображення:
       - Inference з кожним промптом → список scores
       - max_score = max(scores)
       - Визначення binary label через threshold
    3. Збереження результатів з max_score

Ансамблі промптів по категоріях (V2):
    - Catering: 4 промпти (food, drinks, desserts, kitchen items)
    - Marine_Activities: 3 промпти (watercraft, surfboard, swimwear)
    - Cultural_Excursions: 3 промпти (landmarks, castles, sculptures)
    - Pet-Friendly Services: 2 промпти (pets, puppies/kittens)

Вхідні дані:
    - data/stage_1/{category}/true/*.jpg - позитивні зразки (100 на категорію)
    - data/stage_1/{category}/neg/*.jpg - негативні зразки (100 на категорію)

Вихідні дані:
    - results/stage_1/results_siglip_v2.json - результати класифікації

Структура результату:
    {
        "Catering": [
            {
                "image": "cat_12345.jpg",
                "score": 0.000045,
                "label": 1
            },
            ...
        ],
        ...
    }

Модель:
    - SigLIP (google/siglip-base-patch16-224)
    - Sigmoid activation (незалежні image-text scores)
    - Padding: "max_length" (важливо для стабільності)
    - FP16 на GPU

Оптимальний threshold:
    threshold = 0.000010 (визначено експериментально)
    - За замовчуванням: 0.00005 (змінюється в main)

"""

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


class SigLIPClassifierV2:
    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        model_path="weights/siglip",
        threshold=0.00005,
    ):
        """Ініціалізація SigLIP моделі"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Threshold для класифікації
        self.threshold = threshold

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

        # V2: Prompt Ensembling - тільки 4 категорії
        self.POSITIVE_PROMPTS = {
            "Catering": [
                "a photo of food, such as pizza, pasta, hamburger, or sushi",
                "a photo of a drink, such as wine, beer, or a cocktail",
                "a photo of a dessert, cake, coffee, or croissant",
                "a photo of kitchen items, like a bowl, knife, or serving tray",
            ],
            "Marine_Activities": [
                "a photo of a watercraft, boat, or jet ski",
                "a photo of a surfboard or paddle",
                "a photo of swimwear",
            ],
            "Cultural_Excursions": [
                "a photo of a cultural landmark, monument, or museum",
                "a photo of a castle, tower, or lighthouse",
                "a photo of a sculpture or fountain",
            ],
            "Pet-Friendly Services": [
                "a photo of a domestic pet, like a dog or a cat",
                "a picture of a puppy or a kitten",
            ],
        }

        self.output_path = "results/stage_1/results_siglip_v2.json"

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

        # Підготовка inputs (важливо: padding="max_length")
        inputs = self.processor(
            text=[text_prompt], images=image, padding="max_length", return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            # SigLIP використовує sigmoid
            prob = torch.sigmoid(logits_per_image)

        return prob[0][0].cpu().item()

    def is_relevant(self, image_path, category):
        """
        Перевірка релевантності (V2 - Prompt Ensembling)
        Використовує max(scores) з усіх промптів для категорії
        """
        # Отримуємо список позитивних промптів
        text_prompts_list = self.POSITIVE_PROMPTS.get(category)
        if not text_prompts_list:
            return 0, 0.0

        all_scores = []
        for text_prompt in text_prompts_list:
            # Отримуємо sigmoid probability для кожного промпту
            score = self.classify_image(image_path, text_prompt)
            all_scores.append(score)

        # Беремо максимальний бал з усіх варіантів
        max_score = max(all_scores)

        # Визначаємо лейбл через threshold
        label = 1 if max_score > self.threshold else 0

        return label, max_score

    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def process_dataset(self, data_dir="data/stage_1"):
        """Обробка всього датасету"""
        results = {}

        for category in self.POSITIVE_PROMPTS.keys():
            category_path = Path(data_dir) / category
            if not category_path.exists():
                print(f"Skipping {category} - directory not found")
                continue

            print(f"\nProcessing category: {category}")
            category_results = []
            image_files = sorted(
                list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
            )

            # Progress bar для фото в категорії
            for img_path in tqdm(image_files, desc=f"{category}", unit="img"):
                label, score = self.is_relevant(str(img_path), category)

                category_results.append(
                    {"image": img_path.name, "score": round(score, 10), "label": label}
                )

                # Оновлюємо results та зберігаємо JSON після кожного фото
                results[category] = category_results
                self.save_results(results)

            print(f"Completed {category}: {len(category_results)} images processed")

        return results


if __name__ == "__main__":
    print("Starting SigLIP classification (V2 - Prompt Ensembling)...")

    # Встановлюємо поріг, знайдений на попередньому кроці тюнінгу
    optimal_threshold = 0.000030

    classifier = SigLIPClassifierV2(threshold=optimal_threshold)

    results = classifier.process_dataset()

    print("\nSigLIP (Ensemble) classification completed!")
    print(f"Results saved to: {classifier.output_path}")
    print(f"Threshold used: {classifier.threshold}")
