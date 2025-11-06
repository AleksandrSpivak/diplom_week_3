"""
Open Images Dataset v7 - Підготовка датасету для порівняльного аналізу моделей

Завантажує по N релевантних + N нерелевантних іміджів для кожної бізнес-категорії.
Створює ground_truth.json для кожної категорії з мітками relevant: 1/0.

Структура:
data/stage_1/
├── Catering/
│   ├── [іміджі.jpg]
│   └── ground_truth.json
├── Marine_Activities/
├── Cultural_Excursions/
├── Pet-Friendly_Services/
...

Ground truth формат:
{
  "image_id.jpg": {
    "classes": ["Class1", "Class2"],  # Класи з Open Images
    "relevant": 1  # 1 = релевантний для категорії, 0 = нерелевантний
  }
}

Нерелевантні іміджі: завантажуються випадково з датасету, перевіряється що вони
НЕ містять класів поточної категорії. Підходить для будь-якого розміру датасету.
"""

import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import shutil
import json

# Конфігурація
DATA_DIR = "datastage_1"
IMAGES_PER_TYPE = 100  # N релевантних + N нерелевантних
SPLIT = "train"

# Завантажити категорії з JSON - тільки 4 категорії
with open("available_business_categories.json", "r", encoding="utf-8") as f:
    ALL_CATEGORIES = json.load(f)

# Вибрати тільки 4 категорії
BUSINESS_CATEGORIES = {
    "Pet-Friendly Services": ALL_CATEGORIES["Pet-Friendly Services"],
    "Catering": ALL_CATEGORIES["Catering"],
    "Marine_Activities": ALL_CATEGORIES["Marine_Activities"],
    "Cultural_Excursions": ALL_CATEGORIES["Cultural_Excursions"],
}


def download_relevant_images(category_name, class_list, num_images, split):
    """Завантажує релевантні іміджі для категорії"""
    print(f"  Завантаження {num_images} релевантних іміджів...")

    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["classifications"],
            classes=class_list,
            max_samples=num_images * 3,  # Завантажуємо з запасом
            shuffle=True,
            dataset_name=f"oi_{category_name.lower()}_relevant",
        )

        return dataset
    except Exception as e:
        print(f"    ✗ Помилка: {e}")
        return None


def download_irrelevant_images(category_name, excluded_classes, num_images, split):
    """
    Завантажує нерелевантні іміджі (випадкова вибірка з датасету).
    Перевіряє що іміджі НЕ містять excluded_classes.

    МАСШТАБУЄТЬСЯ: не обмежує пошук певними класами, а завантажує
    випадкові іміджі і фільтрує їх по excluded_classes.
    """
    print(f"  Завантаження {num_images} нерелевантних іміджів...")

    try:
        # Завантажуємо випадкову вибірку іміджів БЕЗ обмеження по класах
        # FiftyOne завантажить рандомні іміджі з усього датасету
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["classifications"],
            max_samples=num_images * 5,  # Більший запас для фільтрації
            shuffle=True,
            dataset_name=f"oi_{category_name.lower()}_irrelevant",
        )

        return dataset
    except Exception as e:
        print(f"    ✗ Помилка: {e}")
        return None


def extract_image_classes(sample):
    """Витягує класи з FiftyOne sample"""
    classes = []
    if hasattr(sample, "positive_labels") and sample.positive_labels:
        for label in sample.positive_labels.classifications:
            classes.append(label.label)
    return classes


def process_category(category_name, class_list, data_dir, num_images, split):
    """Обробляє одну категорію: завантажує іміджі + створює ground_truth.json"""
    print(f"\n{'='*70}")
    print(f"Категорія: {category_name}")
    print(f"Класи ({len(class_list)}): {', '.join(class_list[:10])}...")
    print(f"{'='*70}")

    # Створити папку категорії
    category_dir = Path(data_dir) / category_name
    category_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}

    # 1. Завантажити релевантні іміджі
    relevant_dataset = download_relevant_images(
        category_name, class_list, num_images, split
    )

    if relevant_dataset:
        print(f"  Копіювання релевантних іміджів...")
        count = 0
        for sample in relevant_dataset:
            if count >= num_images:
                break

            src_path = sample.filepath
            filename = f"{sample.id}.jpg"
            dst_path = category_dir / filename

            try:
                shutil.copy2(src_path, dst_path)

                # Витягти класи
                classes = extract_image_classes(sample)

                ground_truth[filename] = {"classes": classes, "relevant": 1}

                count += 1

            except Exception as e:
                print(f"    ✗ Помилка {sample.id}: {e}")

        print(f"  ✓ Релевантних: {count}")
        fo.delete_dataset(f"oi_{category_name.lower()}_relevant")

    # 2. Завантажити нерелевантні іміджі
    irrelevant_dataset = download_irrelevant_images(
        category_name, class_list, num_images, split
    )

    if irrelevant_dataset:
        print(f"  Копіювання нерелевантних іміджів...")
        print(f"  Фільтрація: пропускаємо іміджі з класами категорії...")
        count = 0
        checked = 0

        for sample in irrelevant_dataset:
            if count >= num_images:
                break

            checked += 1

            # Витягти класи іміджу
            sample_classes = extract_image_classes(sample)

            # КРИТИЧНА ПЕРЕВІРКА: чи є хоч один клас з поточної категорії?
            has_category_class = any(cls in class_list for cls in sample_classes)

            if has_category_class:
                # Пропустити - цей імідж містить клас з категорії
                continue

            src_path = sample.filepath
            filename = f"{sample.id}_neg.jpg"
            dst_path = category_dir / filename

            try:
                shutil.copy2(src_path, dst_path)

                ground_truth[filename] = {"classes": sample_classes, "relevant": 0}

                count += 1

            except Exception as e:
                print(f"    ✗ Помилка {sample.id}: {e}")

        print(f"  ✓ Нерелевантних: {count} (з {checked} перевірених)")
        fo.delete_dataset(f"oi_{category_name.lower()}_irrelevant")

    # 3. Зберегти ground_truth.json
    gt_path = category_dir / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Ground truth збережено: {len(ground_truth)} іміджів")

    return len(ground_truth)


def main():
    print("=" * 70)
    print("OPEN IMAGES v7 - ПІДГОТОВКА ДАТАСЕТУ ДЛЯ ПОРІВНЯННЯ МОДЕЛЕЙ")
    print("=" * 70)
    print(f"Split: {SPLIT}")
    print(
        f"Іміджів на категорію: {IMAGES_PER_TYPE} релевантних + {IMAGES_PER_TYPE} нерелевантних"
    )
    print(f"Папка: {DATA_DIR}/")
    print(f"Категорій: {len(BUSINESS_CATEGORIES)}")
    print(f"Обрані категорії: {', '.join(BUSINESS_CATEGORIES.keys())}")
    print("=" * 70)

    # Створити головну папку
    Path(DATA_DIR).mkdir(exist_ok=True)

    # Обробити кожну категорію
    total_images = 0
    for category, class_list in BUSINESS_CATEGORIES.items():
        count = process_category(
            category_name=category,
            class_list=class_list,
            data_dir=DATA_DIR,
            num_images=IMAGES_PER_TYPE,
            split=SPLIT,
        )
        total_images += count

    # Фінальна статистика
    print(f"\n{'='*70}")
    print("ЗАВЕРШЕНО")
    print(f"{'='*70}")

    for category in BUSINESS_CATEGORIES.keys():
        category_dir = Path(DATA_DIR) / category
        if category_dir.exists():
            images = list(category_dir.glob("*.jpg"))
            gt_file = category_dir / "ground_truth.json"

            if gt_file.exists():
                with open(gt_file, "r", encoding="utf-8") as f:
                    gt = json.load(f)
                relevant = sum(1 for v in gt.values() if v["relevant"] == 1)
                irrelevant = sum(1 for v in gt.values() if v["relevant"] == 0)

                print(
                    f"✓ {category}: {len(images)} іміджів ({relevant} rel, {irrelevant} irrel)"
                )
            else:
                print(f"✗ {category}: ground_truth.json відсутній")

    print(f"\nЗагалом іміджів: {total_images}")
    print("=" * 70)


if __name__ == "__main__":
    main()
