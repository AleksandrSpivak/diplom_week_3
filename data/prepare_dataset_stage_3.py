"""
Dataset Preparation Script for Stage 3: Hierarchical Profile Tagging

Призначення:
    Генерація синтетичних профілів користувачів для валідації ієрархічної
    мультилейблової тегізації бізнес-інтересів. Профілі імітують реалістичний
    розподіл інтересів з контрольованим рівнем "шуму".

Структура датасету:
    - 1500 профілів (500 Hobbyist + 500 User + 500 Noise)
    - Кожен профіль містить 40-50 зображень
    - Загалом ~67,650 зображень
    - Профілі можуть мати змішані теги (multi-label GT)

Типи профілів:
    1. Hobbyist: Домінантний інтерес (50-60%) + можливий вторинний (15-25%)
    2. User: Помірний інтерес у 1-4 категоріях (15-25% кожна)
    3. Noise: Низький інтерес (0-5% на категорію)

Вихідні дані:
    - data/stage_3/dataset_gt.json - Ground Truth профілів з векторами інтересів
    - data/stage_3/images/*.jpg - зображення з префіксами (cat_, mar_, cul_, pet_, irr_)
"""

import fiftyone as fo
import fiftyone.zoo as foz
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from math import floor

# Константи
TARGET_CATEGORIES = [
    "Pet-Friendly Services",
    "Catering",
    "Marine_Activities",
    "Cultural_Excursions",
]
ALL_CATEGORIES = TARGET_CATEGORIES + ["Irrelevant"]
PREFIX_MAP = {
    "Catering": "cat",
    "Marine_Activities": "mar",
    "Cultural_Excursions": "cul",
    "Pet-Friendly Services": "pet",
    "Irrelevant": "irr",
}

NUM_HOBBYISTS = 500
NUM_USERS = 500
NUM_NOISE = 500
TOTAL_PROFILES = NUM_HOBBYISTS + NUM_USERS + NUM_NOISE

BATCH_SIZE = 5000
SPLIT = "train"
OUTPUT_DIR = Path("data/stage_3")
IMAGES_DIR = OUTPUT_DIR / "images"


class UserVector:
    """Клас для генерації Ground Truth векторів інтересів"""

    @staticmethod
    def create_hobbyist_vector():
        """
        Hobbyist: Один домінуючий інтерес (50-60%) + можливо один вторинний (15-25%)
        """
        vector = {cat: 0.0 for cat in ALL_CATEGORIES}

        # Крок 1: Вибрати домінуючу категорію
        dominant_category = random.choice(TARGET_CATEGORIES)
        dominant_percent = random.uniform(0.50, 0.60)
        vector[dominant_category] = dominant_percent

        # Крок 2: 50/50 шанс додати вторинний інтерес
        if random.random() < 0.5:
            # Вибрати іншу категорію
            available_categories = [
                cat for cat in TARGET_CATEGORIES if cat != dominant_category
            ]
            secondary_category = random.choice(available_categories)
            secondary_percent = random.uniform(0.15, 0.25)
            vector[secondary_category] = secondary_percent

        # Крок 3: Залишок -> Irrelevant
        total_business = sum(vector[cat] for cat in TARGET_CATEGORIES)
        vector["Irrelevant"] = 1.0 - total_business

        return vector

    @staticmethod
    def create_user_vector():
        """
        User: Помірний інтерес у 1-4 сферах (по 15-25% кожна)
        """
        vector = {cat: 0.0 for cat in ALL_CATEGORIES}

        # Крок 1: Вибрати 1-4 категорії
        num_categories = random.randint(1, 4)
        selected_categories = random.sample(TARGET_CATEGORIES, num_categories)

        # Крок 2: Згенерувати відсотки для кожної (15-25%)
        for category in selected_categories:
            percent = random.uniform(0.15, 0.25)
            vector[category] = percent

        # Крок 3: Обробка переповнення
        total_business = sum(vector[cat] for cat in TARGET_CATEGORIES)

        if total_business > 1.0:
            # Нормалізувати тільки бізнес-категорії до 1.0
            scale_factor = 1.0 / total_business
            for cat in TARGET_CATEGORIES:
                vector[cat] *= scale_factor
            vector["Irrelevant"] = 0.0
        else:
            # Залишок -> Irrelevant
            vector["Irrelevant"] = 1.0 - total_business

        return vector

    @staticmethod
    def create_noise_vector():
        """
        Noise: Низький інтерес у 1-4 сферах (по 1-5% кожна)
        """
        vector = {cat: 0.0 for cat in ALL_CATEGORIES}

        # Крок 1: Вибрати 1-4 категорії
        num_categories = random.randint(1, 4)
        selected_categories = random.sample(TARGET_CATEGORIES, num_categories)

        # Крок 2: Згенерувати відсотки для кожної (1-5%)
        for category in selected_categories:
            percent = random.uniform(0.01, 0.05)
            vector[category] = percent

        # Крок 3: Залишок -> Irrelevant
        total_business = sum(vector[cat] for cat in TARGET_CATEGORIES)
        vector["Irrelevant"] = 1.0 - total_business

        return vector


class User:
    """Клас профілю користувача"""

    def __init__(self, simulation_type: str):
        self.user_id = None
        self.total_images = random.randint(40, 50)

        if simulation_type == "hobbyist":
            self.gt_vector = UserVector.create_hobbyist_vector()
        elif simulation_type == "user":
            self.gt_vector = UserVector.create_user_vector()
        elif simulation_type == "noise":
            self.gt_vector = UserVector.create_noise_vector()
        else:
            raise ValueError(f"Unknown simulation_type: {simulation_type}")

        self.image_needs = self._calculate_image_needs()
        self.image_data = []

    def _calculate_image_needs(self):
        """Largest Remainder Method для точного розподілу зображень"""
        raw_values = {
            cat: self.gt_vector[cat] * self.total_images for cat in ALL_CATEGORIES
        }

        needs = {cat: floor(raw_values[cat]) for cat in ALL_CATEGORIES}
        remainders = {cat: raw_values[cat] - needs[cat] for cat in ALL_CATEGORIES}

        current_sum = sum(needs.values())
        missing_images = self.total_images - current_sum

        sorted_categories = sorted(
            remainders.keys(), key=lambda c: remainders[c], reverse=True
        )

        for i in range(missing_images):
            cat = sorted_categories[i]
            needs[cat] += 1

        return needs

    def to_dict(self):
        """Серіалізація для JSON"""
        return {
            "user_id": self.user_id,
            "gt_vector": self.gt_vector,
            "total_images": self.total_images,
            "image_data": self.image_data,
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

    print(f"Генерація {NUM_HOBBYISTS} hobbyists...")
    for i in range(NUM_HOBBYISTS):
        user = User("hobbyist")
        user.user_id = f"hobbyist_{i+1:04d}"
        list_of_users.append(user)

    print(f"Генерація {NUM_USERS} users...")
    for i in range(NUM_USERS):
        user = User("user")
        user.user_id = f"user_{i+1:04d}"
        list_of_users.append(user)

    print(f"Генерація {NUM_NOISE} noise users...")
    for i in range(NUM_NOISE):
        user = User("noise")
        user.user_id = f"noise_{i+1:04d}"
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

    for category in TARGET_CATEGORIES:
        base_set = set(base_categories.get(category, []))
        extended_set = set(extended_categories_data.get(category, []))
        full_set = base_set.union(extended_set)
        final_categories[category] = list(full_set)
        ALL_BUSINESS_CLASSES.update(full_set)

    print(f"\nВсього об'єднаних бізнес-класів: {len(ALL_BUSINESS_CLASSES)}")

    print("\n" + "=" * 60)
    print("ЕТАП Б: ЗАВАНТАЖЕННЯ (Наповнення Пулу)")
    print("=" * 60)

    for category_name in TARGET_CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Завантаження: {category_name}")
        print(f"{'='*60}")

        current_pool = image_pool[category_name]
        needed = total_needs[category_name]

        CURRENT_BASE_CLASSES = base_categories[category_name]
        CURRENT_EXPANDED_CLASSES = set(final_categories[category_name])

        batch_counter = 0
        empty_batches_count = 0

        while len(current_pool) < needed:
            batch_counter += 1
            dataset_name = f"temp_{category_name}_batch{batch_counter}"

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

                    filename = f"{PREFIX_MAP[category_name]}_{sample.id}.jpg"
                    src_path = sample.filepath
                    dst_path = IMAGES_DIR / filename

                    shutil.copy2(src_path, dst_path)

                    sample_classes_list = list(sample_classes)
                    image_metadata[filename] = {
                        "classes": sample_classes_list,
                        "category": category_name,
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

        print(f"✓ Зібрано {len(current_pool)} зображень для {category_name}")

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
