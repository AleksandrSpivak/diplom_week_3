"""
Dataset Preparation Script for Stage 4: Sensitivity Analysis

Призначення:
    Генерація синтетичних профілів для аналізу чутливості системи.
    Визначення мінімальної кількості релевантних зображень (N), необхідної
    для надійної ідентифікації бізнес-інтересу користувача.

Структура експерименту:
    - Одна цільова категорія за раз (TARGET_CATEGORY)
    - Фіксований розмір профілю: 50 зображень
    - Змінна N (кількість сигнальних зображень): від 0 до 10
    - 100 повторень для кожного N
    - Загалом: 1100 профілів × 50 зображень = 55,000 зображень на категорію

Дизайн профілю:
    - N сигнальних зображень (релевантні TARGET_CATEGORY)
    - (50-N) шумових зображень (Irrelevant)
    - N=0: базовий рівень шуму (baseline noise)
    - N=1..10: поступове збільшення сигналу

Вихідні дані:
    - data/stage_4/dataset_gt_{prefix}_100.json - Ground Truth профілів
    - data/stage_4/images/*.jpg - зображення з префіксами

Мета аналізу:
    Побудова "кривої чутливості" - залежності агрегованого score системи
    від кількості сигнальних зображень N, для визначення Reliable N
    (мінімальний N для надійного виявлення).
"""

import fiftyone as fo
import fiftyone.zoo as foz
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Константи для Етапу 4
TARGET_CATEGORY = "Pet-Friendly Services"
ALL_CATEGORIES = [TARGET_CATEGORY, "Irrelevant"]
PREFIX_MAP = {"Pet-Friendly Services": "pet", "Irrelevant": "irr"}

PROFILE_SIZE = 50
N_SIGNALS_RANGE = range(0, 11)
REPETITIONS_PER_N = 100

BATCH_SIZE = 1555
SPLIT = "train"
OUTPUT_DIR = Path("data/stage_4")
IMAGES_DIR = OUTPUT_DIR / "images"


class User:
    """Клас профілю користувача для Етапу 4"""

    def __init__(self, n_signals: int):
        self.user_id = None
        self.total_images = PROFILE_SIZE
        self.n_signals = n_signals
        self.image_needs = self._calculate_image_needs()
        self.image_data = []

    def _calculate_image_needs(self):
        """Розрахунок потреби: N сигналів + (M-N) шуму"""
        noise_images = self.total_images - self.n_signals
        needs = {TARGET_CATEGORY: self.n_signals, "Irrelevant": noise_images}
        return needs

    def to_dict(self):
        """Серіалізація для JSON"""
        return {
            "user_id": self.user_id,
            "target_category": TARGET_CATEGORY,
            "n_signals": self.n_signals,
            "total_images": self.total_images,
            "image_data": self.image_data,
            "image_needs": self.image_needs,
        }


def extract_image_classes(sample):
    """Витягує всі класи classification з sample"""
    classes = set()
    if hasattr(sample, "positive_labels") and sample.positive_labels:
        for label in sample.positive_labels.classifications:
            classes.add(label.label)
    return classes


def main():
    print("=" * 60)
    print("ЕТАП А: ПІДГОТОВКА (Генерація Потреби)")
    print("=" * 60)

    IMAGES_DIR.mkdir(exist_ok=True, parents=True)

    list_of_users = []

    print(
        f"Генерація {REPETITIONS_PER_N} профілів для кожного N в діапазоні {N_SIGNALS_RANGE}..."
    )

    for n_signals in N_SIGNALS_RANGE:
        for rep in range(REPETITIONS_PER_N):
            user = User(n_signals=n_signals)
            user.user_id = (
                f"{PREFIX_MAP[TARGET_CATEGORY]}_N{n_signals:02d}_rep{rep+1:02d}"
            )
            list_of_users.append(user)

    print(f"\nВсього профілів: {len(list_of_users)}")

    total_needs = {cat: 0 for cat in ALL_CATEGORIES}
    for user in list_of_users:
        for cat in ALL_CATEGORIES:
            total_needs[cat] += user.image_needs[cat]

    print("\nЗагальна потреба в зображеннях:")
    total_images_needed = 0
    for cat in ALL_CATEGORIES:
        print(f"  {cat}: {total_needs[cat]}")
        total_images_needed += total_needs[cat]
    print(f"  РАЗОМ: {total_images_needed}")

    image_pool = {cat: [] for cat in ALL_CATEGORIES}
    image_metadata = {}
    global_processed_ids = set()

    with open("available_business_categories.json", "r") as f:
        base_categories = json.load(f)

    with open("category_extensions.json", "r") as f:
        extended_categories_data = json.load(f)

    final_categories = {}
    ALL_BUSINESS_CLASSES = set()

    base_set = set(base_categories.get(TARGET_CATEGORY, []))
    extended_set = set(extended_categories_data.get(TARGET_CATEGORY, []))
    full_set = base_set.union(extended_set)
    final_categories[TARGET_CATEGORY] = list(full_set)
    ALL_BUSINESS_CLASSES.update(full_set)

    print(f"\nВсього об'єднаних бізнес-класів: {len(ALL_BUSINESS_CLASSES)}")

    print("\n" + "=" * 60)
    print("ЕТАП Б: ЗАВАНТАЖЕННЯ (Наповнення Пулу)")
    print("=" * 60)

    print(f"\n{'='*60}")
    print(f"Завантаження: {TARGET_CATEGORY}")
    print(f"{'='*60}")

    current_pool = image_pool[TARGET_CATEGORY]
    needed = total_needs[TARGET_CATEGORY]

    CURRENT_BASE_CLASSES = base_categories[TARGET_CATEGORY]
    CURRENT_EXPANDED_CLASSES = set(final_categories[TARGET_CATEGORY])

    batch_counter = 0
    empty_batches_count = 0

    while len(current_pool) < needed:
        batch_counter += 1
        dataset_name = f"temp_{TARGET_CATEGORY}_batch{batch_counter}"

        print(f"Батч #{batch_counter} ({BATCH_SIZE} зразків)...")

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=SPLIT,
                label_types=["classifications"],
                classes=CURRENT_BASE_CLASSES,
                max_samples=BATCH_SIZE,
                shuffle=True,
                dataset_name=dataset_name,
                download_if_necessary=True,
            )
        except Exception as e:
            print(f"⚠ Помилка завантаження: {e}")
            try:
                fo.delete_dataset(dataset_name)
            except:
                pass
            continue

        new_samples_in_batch = 0

        for sample in tqdm(dataset, desc=f"Фільтрація батчу #{batch_counter}"):
            if len(current_pool) >= needed:
                break

            if sample.id in global_processed_ids:
                continue

            sample_classes = extract_image_classes(sample)
            has_relevance = any(
                cls in CURRENT_EXPANDED_CLASSES for cls in sample_classes
            )

            if has_relevance:
                global_processed_ids.add(sample.id)

                filename = f"{PREFIX_MAP[TARGET_CATEGORY]}_{sample.id}.jpg"
                src_path = sample.filepath
                dst_path = IMAGES_DIR / filename

                shutil.copy2(src_path, dst_path)

                sample_classes_list = list(sample_classes)
                image_metadata[filename] = {
                    "classes": sample_classes_list,
                    "category": TARGET_CATEGORY,
                }

                current_pool.append(filename)
                new_samples_in_batch += 1

        fo.delete_dataset(dataset_name)

        print(
            f"Завантажено: {len(current_pool)}/{needed} (нових: {new_samples_in_batch})"
        )

        if new_samples_in_batch == 0:
            empty_batches_count += 1
            if empty_batches_count >= 5:
                print(f"⚠ 5 порожніх батчів. Зупинка.")
                break
        else:
            empty_batches_count = 0

    print(f"✓ Зібрано {len(current_pool)} зображень для {TARGET_CATEGORY}")

    print(f"\n{'='*60}")
    print("Завантаження: Irrelevant")
    print(f"{'='*60}")

    current_pool = image_pool["Irrelevant"]
    needed = total_needs["Irrelevant"]

    batch_counter = 0

    while len(current_pool) < needed:
        batch_counter += 1
        dataset_name = f"temp_irrelevant_batch{batch_counter}"

        print(f"Батч #{batch_counter} ({BATCH_SIZE} зразків)...")

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
            print(f"⚠ Помилка завантаження: {e}")
            try:
                fo.delete_dataset(dataset_name)
            except:
                pass
            continue

        new_samples_in_batch = 0

        for sample in tqdm(dataset, desc=f"Фільтрація батчу #{batch_counter}"):
            if len(current_pool) >= needed:
                break

            if sample.id in global_processed_ids:
                continue

            sample_classes = extract_image_classes(sample)
            has_business_class = any(
                cls in ALL_BUSINESS_CLASSES for cls in sample_classes
            )

            if not has_business_class:
                global_processed_ids.add(sample.id)

                filename = f"{PREFIX_MAP['Irrelevant']}_{sample.id}.jpg"
                src_path = sample.filepath
                dst_path = IMAGES_DIR / filename

                shutil.copy2(src_path, dst_path)

                sample_classes_list = list(sample_classes)
                image_metadata[filename] = {
                    "classes": sample_classes_list,
                    "category": "Irrelevant",
                }

                current_pool.append(filename)
                new_samples_in_batch += 1

        fo.delete_dataset(dataset_name)

        print(
            f"Завантажено: {len(current_pool)}/{needed} (нових: {new_samples_in_batch})"
        )

        if new_samples_in_batch == 0:
            print(f"⚠ Порожній батч. Зупинка.")
            break

    print(f"✓ Зібрано {len(current_pool)} irrelevant зображень")

    print("\n" + "=" * 60)
    print("ЕТАП В: РОЗПОДІЛ (Наповнення Профілів)")
    print("=" * 60)

    for user in tqdm(list_of_users, desc="Розподіл зображень по профілям"):
        for cat in ALL_CATEGORIES:
            needed_count = user.image_needs[cat]
            for _ in range(needed_count):
                if image_pool[cat]:
                    img_filename = image_pool[cat].pop()

                    user.image_data.append(
                        {
                            "filename": img_filename,
                            "original_classes": image_metadata[img_filename]["classes"],
                        }
                    )
                else:
                    print(
                        f"⚠ Недостатньо зображень для {cat} (користувач {user.user_id})"
                    )

        random.shuffle(user.image_data)

    print(f"✓ Розподіл завершено")

    print("\n" + "=" * 60)
    print("ЕТАП Г: ЗБЕРЕЖЕННЯ")
    print("=" * 60)

    dataset_gt = [user.to_dict() for user in list_of_users]

    gt_path = OUTPUT_DIR / "dataset_gt.json"
    with open(gt_path, "w") as f:
        json.dump(dataset_gt, f, indent=2)

    print(f"✓ Ground truth збережено: {gt_path}")

    print("\n" + "=" * 60)
    print("ФІНАЛЬНА СТАТИСТИКА")
    print("=" * 60)
    print(f"Профілів згенеровано: {len(list_of_users)}")
    print(f"Зображень завантажено: {len(global_processed_ids)}")
    print(f"Зображень розподілено: {sum(len(u.image_data) for u in list_of_users)}")

    remaining_in_pool = sum(len(pool) for pool in image_pool.values())
    if remaining_in_pool > 0:
        print(f"⚠ Залишок в пулі: {remaining_in_pool}")


if __name__ == "__main__":
    main()
