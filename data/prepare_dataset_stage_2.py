"""
Dataset Preparation Script for Stage 2: Multiclass Classification

Призначення:
    Генерація чистого мультикласового датасету для валідації моделі SigLIP
    на задачі класифікації з 5 класами (4 бізнес-категорії + Irrelevant).

Структура датасету:
    - 500 зображень на кожен клас (загалом 2500 зображень)
    - Кожне зображення гарантовано "чисте" (без перехресних бізнес-класів)
    - Використовує розширені списки класів (base + extensions)

Вихідні дані:
    - data/stage_2/ground_truth.json - Ground Truth анотації
    - data/stage_2/*.jpg - зображення з префіксами (cat_, mar_, cul_, pet_, irr_)
"""

import fiftyone as fo
import fiftyone.zoo as foz
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Константи
TARGET_CATEGORIES = [
    "Pet-Friendly Services",
    "Catering",
    "Marine_Activities",
    "Cultural_Excursions",
]
PREFIX_MAP = {
    "Catering": "cat",
    "Marine_Activities": "mar",
    "Cultural_Excursions": "cul",
    "Pet-Friendly Services": "pet",
    "Irrelevant": "irr",
}
IMAGES_PER_CLASS = 500
BATCH_SIZE = 500

OUTPUT_DIR = Path("data/stage_2")
SPLIT = "train"


def extract_image_classes(sample):
    """Витягує всі класи classification з sample"""
    classes = set()
    if hasattr(sample, "positive_labels") and sample.positive_labels:
        for label in sample.positive_labels.classifications:
            classes.add(label.label)
    return classes


def main():
    # Етап 1 (Розширений): Завантаження та Об'єднання категорій
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    ground_truth = {}

    # Завантажуємо БАЗОВІ категорії
    with open("available_business_categories.json", "r") as f:
        base_categories = json.load(f)

    # Завантажуємо ДОДАТКОВІ категорії
    with open("category_extensions.json", "r") as f:
        extended_categories_data = json.load(f)

    # Створюємо фінальні списки класів в пам'яті
    final_categories = {}
    ALL_BUSINESS_CLASSES = set()

    for category in TARGET_CATEGORIES:
        base_set = set(base_categories.get(category, []))
        extended_set = set(extended_categories_data.get(category, []))

        # Об'єднуємо базові та розширені класи
        full_set = base_set.union(extended_set)

        final_categories[category] = list(full_set)
        ALL_BUSINESS_CLASSES.update(full_set)

    print(f"\nВсього об'єднаних бізнес-класів: {len(ALL_BUSINESS_CLASSES)}")

    # Етап 2: Обробка позитивних категорій
    for category_name in TARGET_CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Обробка категорії: {category_name}")
        print(f"{'='*60}")

        CURRENT_BASE_CLASSES = base_categories[category_name]
        CURRENT_EXPANDED_CLASSES = set(final_categories[category_name])
        OTHER_BUSINESS_CLASSES = ALL_BUSINESS_CLASSES - CURRENT_EXPANDED_CLASSES

        print(f"Базових класів: {len(CURRENT_BASE_CLASSES)}")
        print(f"Розширених класів: {len(CURRENT_EXPANDED_CLASSES)}")
        print(f"Інших бізнес-класів: {len(OTHER_BUSINESS_CLASSES)}")

        clean_samples_found = 0
        batch_counter = 0
        processed_ids = set()
        empty_batches_count = 0

        while clean_samples_found < IMAGES_PER_CLASS:
            batch_counter += 1
            dataset_name = f"temp_{category_name}_batch{batch_counter}"

            print(f"Завантаження батчу #{batch_counter} ({BATCH_SIZE} зразків)...")

            try:
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split=SPLIT,
                    label_types=["classifications"],
                    classes=CURRENT_BASE_CLASSES,
                    max_samples=BATCH_SIZE,
                    shuffle=True,
                    dataset_name=dataset_name,
                    download_if_necessary=False,
                )
            except Exception as e:
                print(f"⚠ Помилка завантаження батчу #{batch_counter}: {e}")
                print(f"Спроба повторного завантаження...")
                try:
                    fo.delete_dataset(dataset_name)
                except:
                    pass
                continue

            new_samples_in_batch = 0

            for sample in tqdm(dataset, desc=f"Фільтрація батчу #{batch_counter}"):
                if clean_samples_found >= IMAGES_PER_CLASS:
                    break

                if sample.id in processed_ids:
                    continue

                processed_ids.add(sample.id)
                sample_classes = extract_image_classes(sample)

                # Перевірка релевантності
                has_relevance = any(
                    cls in CURRENT_EXPANDED_CLASSES for cls in sample_classes
                )

                if not has_relevance:
                    continue

                # Перевірка чистоти
                has_overlap = any(
                    cls in OTHER_BUSINESS_CLASSES for cls in sample_classes
                )

                if not has_overlap:
                    filename = f"{PREFIX_MAP[category_name]}_{sample.id}.jpg"
                    src_path = sample.filepath
                    dst_path = OUTPUT_DIR / filename

                    shutil.copy2(src_path, dst_path)

                    ground_truth[filename] = {
                        "label": category_name,
                        "original_classes": list(sample_classes),
                    }

                    clean_samples_found += 1
                    new_samples_in_batch += 1

            fo.delete_dataset(dataset_name)

            print(
                f"Знайдено: {clean_samples_found}/{IMAGES_PER_CLASS} (нових у батчі: {new_samples_in_batch})"
            )

            if new_samples_in_batch == 0:
                empty_batches_count += 1
                print(f"⚠ Батч не дав нових зразків. Спроба {empty_batches_count}/5")

                if empty_batches_count >= 5:
                    print(f"⚠ 5 батчів підряд без нових зразків. Зупинка.")
                    break
            else:
                empty_batches_count = 0

        print(f"✓ Зібрано {clean_samples_found} чистих зображень для {category_name}")

    # Етап 3: Обробка негативної категорії
    print(f"\n{'='*60}")
    print("Обробка категорії: Irrelevant")
    print(f"{'='*60}")

    clean_samples_found = 0
    batch_counter = 0
    processed_ids = set()

    while clean_samples_found < IMAGES_PER_CLASS:
        batch_counter += 1
        dataset_name = f"temp_irrelevant_batch{batch_counter}"

        print(f"Завантаження батчу #{batch_counter} ({BATCH_SIZE} зразків)...")

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=SPLIT,
                label_types=["classifications"],
                max_samples=BATCH_SIZE,
                shuffle=True,
                dataset_name=dataset_name,
                download_if_necessary=False,
            )
        except Exception as e:
            print(f"⚠ Помилка завантаження батчу #{batch_counter}: {e}")
            try:
                fo.delete_dataset(dataset_name)
            except:
                pass
            continue

        new_samples_in_batch = 0

        for sample in tqdm(dataset, desc=f"Фільтрація батчу #{batch_counter}"):
            if clean_samples_found >= IMAGES_PER_CLASS:
                break

            if sample.id in processed_ids:
                continue

            processed_ids.add(sample.id)
            sample_classes = extract_image_classes(sample)

            has_business_class = any(
                cls in ALL_BUSINESS_CLASSES for cls in sample_classes
            )

            if not has_business_class:
                filename = f"{PREFIX_MAP['Irrelevant']}_{sample.id}.jpg"
                src_path = sample.filepath
                dst_path = OUTPUT_DIR / filename

                shutil.copy2(src_path, dst_path)

                ground_truth[filename] = {
                    "label": "Irrelevant",
                    "original_classes": list(sample_classes),
                }

                clean_samples_found += 1
                new_samples_in_batch += 1

        fo.delete_dataset(dataset_name)

        print(
            f"Знайдено: {clean_samples_found}/{IMAGES_PER_CLASS} (нових у батчі: {new_samples_in_batch})"
        )

        if new_samples_in_batch == 0:
            print(f"⚠ Батч не дав нових зразків. Зупинка.")
            break

    print(f"✓ Зібрано {clean_samples_found} негативних зображень")

    # Етап 4: Збереження
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Фінальна статистика
    print(f"\n{'='*60}")
    print("ФІНАЛЬНА СТАТИСТИКА")
    print(f"{'='*60}")

    stats = {}
    for filename, data in ground_truth.items():
        label = data["label"]
        stats[label] = stats.get(label, 0) + 1

    for label, count in sorted(stats.items()):
        print(f"{label}: {count} зображень")

    print(f"\nВсього: {len(ground_truth)} зображень")
    print(f"Ground truth збережено: {gt_path}")


if __name__ == "__main__":
    main()
