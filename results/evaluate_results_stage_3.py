"""
Stage 3 Grid Search Analysis

Призначення:
    Пошук оптимальних порогів для фільтрації та тегізації профілів користувачів.
    Перебір комбінацій 3 порогів (filter, hobbyist, user) та розрахунок
    мультилейблових метрик для кожної конфігурації.

Підхід:
    1. Завантаження analysis_results.json з image-level scores для кожного профілю
    2. Grid Search по 3 параметрах:
       - filter_threshold: фільтрація низько-впевнених передбачень
       - hobbyist_threshold: поріг для Hobbyist тегів
       - user_threshold: поріг для User тегів
    3. Для кожної комбінації порогів:
       - Агрегація зображень через WTA + Filter
       - Генерація тегів за ієрархічним алгоритмом
       - Розрахунок мультилейблових метрик
    4. Вибір конфігурації з найвищим Sample F1-Score

Метрики:
    Sample-Based:
    - Sample F1: середнє F1 по всіх профілях (основна метрика)
    - EMR (Exact Match Ratio): % профілів з точним збігом тегів

    Label-Based:
    - Hobbyist Precision/Recall: метрики для Hobbyist_* тегів
    - User Precision/Recall: метрики для User_* тегів

Простір пошуку:
    - filter_threshold: [0.0, 0.000001, 0.00001, 0.00005, 0.0001]
    - hobbyist_threshold: [0.25, 0.275, ..., 0.50] (крок 0.025, 11 значень)
    - user_threshold: [0.075, 0.0825, ..., 0.15] (крок 0.0075, 11 значень)
    - Всього: 5 × 11 × 11 = 605 комбінацій

Ground Truth константи:
    - GT_HOBBYIST_THRESHOLD: 0.50
    - GT_USER_THRESHOLD: 0.15

Вхідні дані:
    - results/stage_3/analysis_results.json - image-level scores для профілів

Вихідні дані:
    - results/stage_3/full_analysis_report.json - результати Grid Search

Структура результату:
    {
        "best_configuration": {
            "filter_threshold": 0.00001,
            "hobbyist_threshold": 0.45,
            "user_threshold": 0.12,
            "sample_f1": 0.87,
            "emr": 0.65,
            "hobbyist_precision": 0.92,
            "hobbyist_recall": 0.88,
            "user_precision": 0.85,
            "user_recall": 0.82
        },
        "all_results": [...]
    }

Категорії:
    Бізнес-категорії (4):
    - Catering
    - Marine_Activities
    - Cultural_Excursions
    - Pet-Friendly Services

    Всі категорії (5):
    - Бізнес-категорії + Irrelevant

Виведення:
    - Progress bar для Grid Search
    - Best configuration з порогами та метриками
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
from tqdm import tqdm


class Stage3Analyzer:
    # Константи для GT
    GT_HOBBYIST_THRESHOLD = 0.50
    GT_USER_THRESHOLD = 0.15

    BUSINESS_CATEGORIES = [
        "Catering",
        "Marine_Activities",
        "Cultural_Excursions",
        "Pet-Friendly Services",
    ]
    ALL_CATEGORIES = BUSINESS_CATEGORIES + ["Irrelevant"]

    def __init__(self, analysis_results_path: str):
        """Ініціалізація аналізатора"""
        print(f"Loading analysis results from: {analysis_results_path}")
        with open(analysis_results_path, "r", encoding="utf-8") as f:
            self.profiles = json.load(f)

        print(f"Loaded {len(self.profiles)} profiles")

        # Підготовка GT тегів один раз
        print("Preparing Ground Truth tags...")
        self.gt_tags = self._prepare_gt_tags()

    def _prepare_gt_tags(self) -> Dict[str, Set[str]]:
        """Підготовка еталонних тегів для всіх профілів"""
        gt_tags = {}

        for profile in self.profiles:
            user_id = profile["user_id"]
            gt_vector = profile["gt_vector"]
            tags = self._generate_tags(
                gt_vector, self.GT_HOBBYIST_THRESHOLD, self.GT_USER_THRESHOLD
            )
            gt_tags[user_id] = tags

        return gt_tags

    def _generate_tags(
        self, vector: Dict[str, float], hobbyist_threshold: float, user_threshold: float
    ) -> Set[str]:
        """
        Генерація тегів з вектора за ієрархічним алгоритмом

        Args:
            vector: Вектор інтересів
            hobbyist_threshold: Поріг для Hobbyist
            user_threshold: Поріг для User

        Returns:
            Набір тегів (наприклад, {'Hobbyist_Catering', 'User_Marine_Activities'})
        """
        tags = set()

        for category in self.BUSINESS_CATEGORIES:
            percent = vector.get(category, 0.0)

            if percent >= hobbyist_threshold:
                tags.add(f"Hobbyist_{category}")
            elif percent >= user_threshold:
                tags.add(f"User_{category}")

        return tags

    def _aggregate_with_filter(
        self, image_details: List[Dict], filter_threshold: float
    ) -> Dict[str, float]:
        """
        Агрегація з фільтрацією (WTA + Filter)

        Args:
            image_details: Список зображень з scores
            filter_threshold: Поріг фільтрації

        Returns:
            Агрегований вектор
        """
        counts = {cat: 0 for cat in self.ALL_CATEGORIES}

        for image_detail in image_details:
            scores = image_detail["scores"]

            # WTA
            max_class = max(scores, key=scores.get)
            max_score = scores[max_class]

            # Фільтрація
            if max_score < filter_threshold and max_class != "Irrelevant":
                final_class = "Irrelevant"
            else:
                final_class = max_class

            counts[final_class] += 1

        # Нормалізація
        total = len(image_details)
        vector = {
            cat: counts[cat] / total if total > 0 else 0.0
            for cat in self.ALL_CATEGORIES
        }

        return vector

    def _calculate_multilabel_metrics(
        self, all_gt_tags: Dict[str, Set[str]], all_pred_tags: Dict[str, Set[str]]
    ) -> Dict:
        """
        Розрахунок мультилейблових метрик

        Args:
            all_gt_tags: GT теги для всіх профілів
            all_pred_tags: Predicted теги для всіх профілів

        Returns:
            Словник з метриками
        """
        # Sample-Based метрики
        f1_scores = []
        exact_matches = 0

        for user_id in all_gt_tags.keys():
            gt = all_gt_tags[user_id]
            pred = all_pred_tags[user_id]

            # Exact Match
            if gt == pred:
                exact_matches += 1

            # F1 для профілю
            if len(gt) == 0 and len(pred) == 0:
                f1 = 1.0
            elif len(gt) == 0 or len(pred) == 0:
                f1 = 0.0
            else:
                intersection = len(gt & pred)
                precision = intersection / len(pred) if len(pred) > 0 else 0.0
                recall = intersection / len(gt) if len(gt) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

            f1_scores.append(f1)

        sample_f1 = np.mean(f1_scores)
        emr = exact_matches / len(all_gt_tags)

        # Label-Based метрики (окремо для Hobbyist та User)
        hobbyist_metrics = self._calculate_label_metrics(
            all_gt_tags, all_pred_tags, "Hobbyist"
        )
        user_metrics = self._calculate_label_metrics(all_gt_tags, all_pred_tags, "User")

        return {
            "sample_f1": sample_f1,
            "emr": emr,
            "hobbyist_precision": hobbyist_metrics["precision"],
            "hobbyist_recall": hobbyist_metrics["recall"],
            "user_precision": user_metrics["precision"],
            "user_recall": user_metrics["recall"],
        }

    def _calculate_label_metrics(
        self,
        all_gt_tags: Dict[str, Set[str]],
        all_pred_tags: Dict[str, Set[str]],
        prefix: str,
    ) -> Dict:
        """Розрахунок Precision/Recall для тегів з певним префіксом"""
        tp = 0
        fp = 0
        fn = 0

        for user_id in all_gt_tags.keys():
            gt = {tag for tag in all_gt_tags[user_id] if tag.startswith(prefix)}
            pred = {tag for tag in all_pred_tags[user_id] if tag.startswith(prefix)}

            tp += len(gt & pred)
            fp += len(pred - gt)
            fn += len(gt - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {"precision": precision, "recall": recall}

    def run_grid_search(self) -> List[Dict]:
        """
        Основний метод Grid Search

        Returns:
            Список результатів для всіх комбінацій порогів
        """
        # Визначення простору пошуку
        filter_thresholds = [0.0, 0.000001, 0.00001, 0.00005, 0.0001]

        # Hobbyist: від 50% до 100% від 0.50, крок 5% (0.025)
        hobbyist_thresholds = [0.25 + i * 0.025 for i in range(11)]  # 0.25 до 0.50

        # User: від 50% до 100% від 0.15, крок 5% (0.0075)
        user_thresholds = [0.075 + i * 0.0075 for i in range(11)]  # 0.075 до 0.15

        total_combinations = (
            len(filter_thresholds) * len(hobbyist_thresholds) * len(user_thresholds)
        )

        print(f"\n{'='*70}")
        print("GRID SEARCH")
        print(f"{'='*70}")
        print(f"Filter thresholds: {len(filter_thresholds)}")
        print(f"Hobbyist thresholds: {len(hobbyist_thresholds)}")
        print(f"User thresholds: {len(user_thresholds)}")
        print(f"Total combinations: {total_combinations}")
        print(f"{'='*70}\n")

        results = []

        # Grid Search
        with tqdm(total=total_combinations, desc="Grid Search") as pbar:
            for filter_t in filter_thresholds:
                for hobbyist_t in hobbyist_thresholds:
                    for user_t in user_thresholds:
                        # Агрегація з фільтрацією
                        pred_vectors = {}
                        for profile in self.profiles:
                            user_id = profile["user_id"]
                            image_details = profile["image_details"]
                            pred_vector = self._aggregate_with_filter(
                                image_details, filter_t
                            )
                            pred_vectors[user_id] = pred_vector

                        # Тегізація
                        pred_tags = {}
                        for user_id, pred_vector in pred_vectors.items():
                            tags = self._generate_tags(pred_vector, hobbyist_t, user_t)
                            pred_tags[user_id] = tags

                        # Розрахунок метрик
                        metrics = self._calculate_multilabel_metrics(
                            self.gt_tags, pred_tags
                        )

                        # Збереження результату
                        results.append(
                            {
                                "filter_threshold": filter_t,
                                "hobbyist_threshold": hobbyist_t,
                                "user_threshold": user_t,
                                "sample_f1": metrics["sample_f1"],
                                "emr": metrics["emr"],
                                "hobbyist_precision": metrics["hobbyist_precision"],
                                "hobbyist_recall": metrics["hobbyist_recall"],
                                "user_precision": metrics["user_precision"],
                                "user_recall": metrics["user_recall"],
                            }
                        )

                        pbar.update(1)

        return results

    def find_best_configuration(self, results: List[Dict]) -> Dict:
        """Пошук найкращої конфігурації за Sample F1"""
        best = max(results, key=lambda x: x["sample_f1"])
        return best

    def print_best_configuration(self, best: Dict):
        """Вивід найкращої конфігурації"""
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION")
        print(f"{'='*70}")
        print(f"Filter Threshold:    {best['filter_threshold']:.6f}")
        print(f"Hobbyist Threshold:  {best['hobbyist_threshold']:.4f}")
        print(f"User Threshold:      {best['user_threshold']:.4f}")
        print(f"\n{'='*70}")
        print("METRICS")
        print(f"{'='*70}")
        print(f"Sample F1-Score:     {best['sample_f1']:.4f}")
        print(f"Exact Match Ratio:   {best['emr']:.4f}")
        print(f"\nHobbyist Tags:")
        print(f"  Precision:         {best['hobbyist_precision']:.4f}")
        print(f"  Recall:            {best['hobbyist_recall']:.4f}")
        print(f"\nUser Tags:")
        print(f"  Precision:         {best['user_precision']:.4f}")
        print(f"  Recall:            {best['user_recall']:.4f}")
        print(f"{'='*70}")


def main():
    print("=" * 70)
    print("STAGE 3: GRID SEARCH ANALYSIS")
    print("=" * 70)

    # Шляхи
    analysis_results_path = "results/stage_3/analysis_results.json"
    output_path = "results/stage_3/full_analysis_report.json"

    # Створення директорії для результатів
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Ініціалізація аналізатора
    analyzer = Stage3Analyzer(analysis_results_path)

    # Запуск Grid Search
    results = analyzer.run_grid_search()

    # Пошук найкращої конфігурації
    best = analyzer.find_best_configuration(results)

    # Збереження результатів
    print(f"\nSaving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_configuration": best, "all_results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Вивід найкращої конфігурації
    analyzer.print_best_configuration(best)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
