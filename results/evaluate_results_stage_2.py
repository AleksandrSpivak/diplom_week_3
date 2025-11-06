"""
Evaluation Script for Multiclass Classification

Призначення:
    Обчислює метрики для мультикласової класифікації (5 класів):
    - Overall Accuracy, Macro F1
    - Precision/Recall/F1 per class
    - Confusion Matrix
    - Average Precision/Recall для 4 бізнес-категорій

Підхід:
    1. Завантаження results_{model}.json з predicted та true labels
    2. Побудова confusion matrix (5x5)
    3. Розрахунок метрик для кожного класу:
       - TP: діагональний елемент [i][i]
       - FP: сума колонки - TP
       - FN: сума рядка - TP
    4. Overall Accuracy: trace(CM) / sum(CM)
    5. Macro F1: середнє F1 по всіх класах

Метрики:
    - Overall Accuracy: (sum діагоналі CM) / (sum всієї CM)
    - Macro F1: середнє F1-Score по всіх 5 класах
    - Per-class Precision: TP / (TP + FP)
    - Per-class Recall: TP / (TP + FN)
    - Per-class F1: 2 * (P * R) / (P + R)

Вхідні дані:
    - results/stage_2/results_{model}.json - predicted та true labels

Вихідні дані:
    - results/stage_2/multiclass_metrics.json - метрики по моделях

Структура результату:
    {
        "siglip_multiclass": {
            "overall_accuracy": 0.85,
            "macro_f1": 0.82,
            "per_class_metrics": {
                "Catering": {
                    "precision": 0.90,
                    "recall": 0.85,
                    "f1_score": 0.87,
                    "support": 100
                },
                ...
            },
            "confusion_matrix": [[...], ...]
        },
        ...
    }

Класи:
    - Catering
    - Marine_Activities
    - Cultural_Excursions
    - Pet-Friendly Services
    - Irrelevant

Моделі для оцінки:
    - siglip_multiclass
    - siglip_multiclass_with_filtration

Виведення:
    - Per-class metrics table: Accuracy, Precision, Recall, F1, Support
    - Confusion Matrix: Ground Truth (rows) vs Predicted (columns)
    - Average Precision/Recall для 4 бізнес-категорій
"""

import json
from pathlib import Path
import numpy as np


class MulticlassEvaluator:
    def __init__(self, results_dir="results/stage_2"):
        self.results_dir = Path(results_dir)
        self.classes = [
            "Catering",
            "Marine_Activities",
            "Cultural_Excursions",
            "Pet-Friendly Services",
            "Irrelevant",
        ]
        self.output_path = self.results_dir / "multiclass_metrics.json"

    def load_results(self, model_name):
        """Завантажити результати моделі"""
        results_path = self.results_dir / f"results_{model_name}.json"
        if not results_path.exists():
            return []

        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_confusion_matrix(self, results):
        """Побудова confusion matrix"""
        n_classes = len(self.classes)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for result in results:
            true_label = result["true_label"]
            pred_label = result["predicted_label"]

            true_idx = class_to_idx.get(true_label)
            pred_idx = class_to_idx.get(pred_label)

            if true_idx is not None and pred_idx is not None:
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
                "support": int(confusion_matrix[idx, :].sum()),
            }

        return metrics

    def calculate_overall_accuracy(self, confusion_matrix):
        """Розрахунок Overall Accuracy"""
        correct = np.trace(confusion_matrix)
        total = confusion_matrix.sum()
        return round(correct / total, 4) if total > 0 else 0

    def calculate_macro_f1(self, per_class_metrics):
        """Розрахунок Macro F1"""
        f1_scores = [metrics["f1_score"] for metrics in per_class_metrics.values()]
        return round(np.mean(f1_scores), 4)

    def calculate_business_metrics(self, per_class_metrics):
        """Розрахунок Average Precision/Recall для бізнес-категорій (без Irrelevant)"""
        business_classes = [cls for cls in self.classes if cls != "Irrelevant"]

        precisions = [per_class_metrics[cls]["precision"] for cls in business_classes]
        recalls = [per_class_metrics[cls]["recall"] for cls in business_classes]

        avg_precision = round(np.mean(precisions), 4)
        avg_recall = round(np.mean(recalls), 4)

        return avg_precision, avg_recall

    def evaluate_model(self, model_name):
        """Оцінка однієї моделі"""
        results = self.load_results(model_name)

        if not results:
            print(f"No results found for {model_name}")
            return None

        # Confusion Matrix
        confusion_matrix = self.build_confusion_matrix(results)

        # Per-class metrics
        per_class_metrics = self.calculate_metrics_per_class(confusion_matrix)

        # Overall Accuracy
        overall_accuracy = self.calculate_overall_accuracy(confusion_matrix)

        # Macro F1
        macro_f1 = self.calculate_macro_f1(per_class_metrics)

        return {
            "overall_accuracy": overall_accuracy,
            "macro_f1": macro_f1,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix.tolist(),
        }

    def save_results(self, all_metrics):
        """Збереження результатів"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {self.output_path}")

    def print_metrics(self, model_name, metrics):
        """Виведення метрик"""
        print(f"\n{'='*90}")
        print(f"Model: {model_name}")
        print(f"{'='*90}")

        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")

        avg_precision, avg_recall = self.calculate_business_metrics(
            metrics["per_class_metrics"]
        )
        print(f"\nAverage Precision for 4 business categories: {avg_precision:.4f}")
        print(f"Average Recall for 4 business categories: {avg_recall:.4f}")

        print(
            f"\n{'Class':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
        )
        print("-" * 90)

        confusion_matrix = np.array(metrics["confusion_matrix"])
        total_samples = confusion_matrix.sum()

        for idx, (class_name, class_metrics) in enumerate(
            metrics["per_class_metrics"].items()
        ):
            tp = confusion_matrix[idx][idx]
            tn = (
                confusion_matrix.sum()
                - confusion_matrix[idx, :].sum()
                - confusion_matrix[:, idx].sum()
                + tp
            )
            accuracy = (tp + tn) / total_samples if total_samples > 0 else 0

            print(
                f"{class_name:<25} "
                f"{accuracy:>10.4f} "
                f"{class_metrics['precision']:>10.4f} "
                f"{class_metrics['recall']:>10.4f} "
                f"{class_metrics['f1_score']:>10.4f} "
                f"{class_metrics['support']:>10}"
            )

        print(f"\n{'='*90}")

    def print_confusion_matrix(self, model_name, confusion_matrix):
        """Виведення Confusion Matrix"""
        print(f"\nConfusion Matrix for {model_name}:")
        print(f"Ground Truth (rows) vs Predicted (columns)\n")
        print(f"{'':<25}", end="")
        for class_name in self.classes:
            print(f"{class_name[:10]:>12}", end="")
        print()
        print("-" * (25 + 12 * len(self.classes)))

        for idx, class_name in enumerate(self.classes):
            print(f"{class_name:<25}", end="")
            for val in confusion_matrix[idx]:
                print(f"{val:>12}", end="")
            print()
        print()


if __name__ == "__main__":
    print("Starting multiclass evaluation...")

    evaluator = MulticlassEvaluator()

    # Список моделей для оцінки
    models = ["siglip_multiclass", "siglip_multiclass_with_filtration"]

    all_metrics = {}

    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        metrics = evaluator.evaluate_model(model_name)

        if metrics:
            all_metrics[model_name] = metrics
            evaluator.print_metrics(model_name, metrics)
            evaluator.print_confusion_matrix(model_name, metrics["confusion_matrix"])

    if all_metrics:
        evaluator.save_results(all_metrics)

    print("\nEvaluation completed!")
