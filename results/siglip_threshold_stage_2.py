"""
SigLIP Threshold Tuning Script

Призначення:
    Знаходить оптимальний threshold для SigLIP мультикласової моделі
    на основі збережених scores у results_siglip_multiclass.json.
    Threshold застосовується для фільтрації низько-впевнених передбачень
    у клас "Irrelevant".

Підхід:
    1. Завантаження results_siglip_multiclass.json з scores для кожного класу
    2. Для кожного threshold з діапазону:
       - Якщо predicted_label != "Irrelevant" AND score < threshold → "Irrelevant"
       - Інакше залишити predicted_label
    3. Розрахунок метрик для кожного threshold:
       - Macro F1, Overall Accuracy
       - Average Precision/Recall для 4 бізнес-категорій
    4. Вибір threshold з найвищим Macro F1

Метрики для оцінки:
    - Macro F1: середнє F1 по всіх 5 класах (основна метрика для вибору)
    - Overall Accuracy: загальна точність
    - Average Precision: середня Precision для 4 бізнес-категорій (без Irrelevant)
    - Average Recall: середній Recall для 4 бізнес-категорій (без Irrelevant)

Вхідні дані:
    - results/stage_2/results_siglip_multiclass.json - scores та labels

Вихідні дані:
    - results/stage_2/siglip_threshold_tuning.json - оптимальний threshold та всі результати

Структура результату:
    {
        "best_threshold": 0.00005,
        "all_results": [
            {
                "threshold": 0.00001,
                "metrics": {
                    "macro_f1": 0.85,
                    "overall_accuracy": 0.87,
                    "avg_precision": 0.90,
                    "avg_recall": 0.88,
                    "per_class_metrics": {...}
                }
            },
            ...
        ]
    }

Класи:
    - Catering
    - Marine_Activities
    - Cultural_Excursions
    - Pet-Friendly Services
    - Irrelevant

Виведення:
    - Metrics для кожного threshold (F1, Acc, P, R)
    - Best threshold з найвищим Macro F1
    - Per-class metrics для найкращого threshold
"""

import json
from pathlib import Path
import numpy as np


class SigLIPThresholdTuner:
    def __init__(self, results_dir="results/stage_2"):
        self.results_dir = Path(results_dir)
        self.siglip_results_path = self.results_dir / "results_siglip_multiclass.json"
        self.classes = [
            "Catering",
            "Marine_Activities",
            "Cultural_Excursions",
            "Pet-Friendly Services",
            "Irrelevant",
        ]

    def load_siglip_results(self):
        """Завантажити результати SigLIP"""
        if not self.siglip_results_path.exists():
            print(f"Error: {self.siglip_results_path} not found!")
            return []

        with open(self.siglip_results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_confusion_matrix(self, predictions, ground_truth):
        """Побудова confusion matrix для 5 класів"""
        n_classes = len(self.classes)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for true_label, pred_label in zip(ground_truth, predictions):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            confusion_matrix[true_idx][pred_idx] += 1

        return confusion_matrix

    def calculate_metrics_per_class(self, confusion_matrix):
        """Розрахунок Precision/Recall/F1 для кожного класу"""
        metrics = {}

        for idx, class_name in enumerate(self.classes):
            tp = confusion_matrix[idx][idx]
            fp = confusion_matrix[:, idx].sum() - tp
            fn = confusion_matrix[idx, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[class_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            }

        return metrics

    def calculate_macro_f1(self, per_class_metrics):
        """Розрахунок Macro F1"""
        f1_scores = [metrics["f1_score"] for metrics in per_class_metrics.values()]
        return round(np.mean(f1_scores), 4)

    def calculate_overall_accuracy(self, confusion_matrix):
        """Розрахунок Overall Accuracy"""
        correct = np.trace(confusion_matrix)
        total = confusion_matrix.sum()
        return round(correct / total, 4) if total > 0 else 0

    def calculate_business_metrics(self, per_class_metrics):
        """Розрахунок Average Precision/Recall для бізнес-категорій"""
        business_classes = [cls for cls in self.classes if cls != "Irrelevant"]

        precisions = [per_class_metrics[cls]["precision"] for cls in business_classes]
        recalls = [per_class_metrics[cls]["recall"] for cls in business_classes]

        avg_precision = round(np.mean(precisions), 4)
        avg_recall = round(np.mean(recalls), 4)

        return avg_precision, avg_recall

    def evaluate_threshold(self, threshold, siglip_results):
        """Оцінка метрик для конкретного порогу"""
        predictions = []
        ground_truth = []

        for result in siglip_results:
            true_label = result["true_label"]
            predicted_label = result["predicted_label"]
            scores = result["scores"]

            # Якщо predicted != Irrelevant, отримуємо його score
            if predicted_label != "Irrelevant":
                predicted_score = scores.get(predicted_label, 0)

                # Застосовуємо threshold
                if predicted_score < threshold:
                    predicted_label = "Irrelevant"

            predictions.append(predicted_label)
            ground_truth.append(true_label)

        # Побудова confusion matrix
        confusion_matrix = self.build_confusion_matrix(predictions, ground_truth)

        # Розрахунок метрик
        per_class_metrics = self.calculate_metrics_per_class(confusion_matrix)
        macro_f1 = self.calculate_macro_f1(per_class_metrics)
        overall_accuracy = self.calculate_overall_accuracy(confusion_matrix)
        avg_precision, avg_recall = self.calculate_business_metrics(per_class_metrics)

        return {
            "macro_f1": macro_f1,
            "overall_accuracy": overall_accuracy,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "per_class_metrics": per_class_metrics,
        }

    def tune_threshold(self, start=0.000001, end=0.9, step=0.05):
        """Пошук оптимального порогу"""
        print("Loading SigLIP results...")
        siglip_results = self.load_siglip_results()

        if not siglip_results:
            return

        print(f"\nTuning threshold from {start} to {end}...")
        print("=" * 100)

        # Створюємо список порогів
        thresholds_mega_fine = np.arange(0.000001, 0.00001, 0.000001)
        thresholds_ultra_hyper_fine = np.arange(0.00001, 0.0001, 0.00001)
        thresholds_hyper_fine = np.arange(0.0001, 0.001, 0.0001)
        thresholds_ultra_fine = np.arange(0.001, 0.01, 0.001)
        thresholds_fine = np.arange(0.01, 0.1, 0.01)
        thresholds_coarse = np.arange(0.1, end + step, step)

        thresholds = thresholds_ultra_hyper_fine

        results = []

        for threshold in thresholds:
            metrics = self.evaluate_threshold(threshold, siglip_results)

            results.append({"threshold": round(threshold, 6), "metrics": metrics})

            print(
                f"Threshold: {threshold:.6f} | "
                f"F1={metrics['macro_f1']:.4f} | "
                f"Acc={metrics['overall_accuracy']:.4f} | "
                f"P={metrics['avg_precision']:.4f} | "
                f"R={metrics['avg_recall']:.4f}"
            )

        print("=" * 100)

        # Знайти найкращий threshold за Macro F1-Score
        best_result = max(results, key=lambda x: x["metrics"]["macro_f1"])
        best_threshold = best_result["threshold"]

        print(f"\n{'='*100}")
        print(f"BEST THRESHOLD: {best_threshold}")
        print(f"{'='*100}")
        print(f"Macro F1:  {best_result['metrics']['macro_f1']:.4f}")
        print(f"Accuracy:  {best_result['metrics']['overall_accuracy']:.4f}")
        print(
            f"Average Precision (4 business): {best_result['metrics']['avg_precision']:.4f}"
        )
        print(
            f"Average Recall (4 business):    {best_result['metrics']['avg_recall']:.4f}"
        )
        print(f"\nPer-class metrics:")
        for class_name, class_metrics in best_result["metrics"][
            "per_class_metrics"
        ].items():
            print(
                f"  {class_name:<25} P={class_metrics['precision']:.4f} "
                f"R={class_metrics['recall']:.4f} F1={class_metrics['f1_score']:.4f}"
            )
        print(f"{'='*100}")

        # Збереження результатів
        output_path = self.results_dir / "siglip_threshold_tuning.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"best_threshold": best_threshold, "all_results": results},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n✓ Detailed results saved to: {output_path}")

        return best_threshold, best_result


if __name__ == "__main__":
    print("Starting SigLIP Threshold Tuning...")

    tuner = SigLIPThresholdTuner()

    # Пошук оптимального порогу
    best_threshold, best_result = tuner.tune_threshold()

    print("\nThreshold tuning completed!")
    print(f"Recommended threshold: {best_threshold}")
