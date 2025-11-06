"""
Evaluation Script for Business Object Detection Models

Призначення:
    Обчислює метрики (Accuracy, Precision, Recall, F1-Score) для 6 моделей
    на основі results_*.json та ground_truth.json з кожної категорії.

Підхід:
    1. Завантаження ground_truth.json для кожної категорії
    2. Завантаження results_{model}.json для кожної моделі
    3. Порівняння predicted labels з true labels
    4. Обчислення confusion matrix (TP, TN, FP, FN)
    5. Розрахунок метрик з confusion matrix

Метрики:
    - Accuracy: (TP + TN) / total
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

Вхідні дані:
    - data/stage_1/{category}/ground_truth.json - true labels
    - results/stage_1/results_{model}.json - predicted labels

Вихідні дані:
    - results/stage_1/summary_metrics.json - метрики по моделях

Структура результату:
    {
        "per_category": {
            "yolo": {
                "Catering": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.98,
                    "f1_score": 0.95,
                    "confusion_matrix": {"TP": 98, "TN": 92, "FP": 8, "FN": 2}
                },
                ...
            },
            ...
        },
        "averages": {
            "yolo": {
                "accuracy": 0.93,
                "precision": 0.91,
                "recall": 0.95,
                "f1_score": 0.93
            },
            ...
        }
    }

Моделі для оцінки:
    - yolo
    - grounding_dino (gr_dino)
    - clip
    - blip
    - blip_vqa
    - siglip_v2 (siglip)

Виведення:
    - Summary table: середні метрики по всіх категоріях (сортовано за F1)
    - Category breakdown: TP та TN по категоріях для кожної моделі
"""

import json
from pathlib import Path
from collections import defaultdict


class ModelEvaluator:
    def __init__(self, data_dir='data/stage_1', results_dir='results/stage_1'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Моделі для оцінки
        self.models = [
            'yolo',
            'grounding_dino',
            'clip',
            'blip',
            'blip_vqa',
            'siglip_v2'
        ]
        
        self.output_path = self.results_dir / 'summary_metrics.json'
    
    def load_ground_truth(self, category):
        """Завантажити ground truth для категорії"""
        gt_path = self.data_dir / category / 'ground_truth.json'
        if not gt_path.exists():
            return {}
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Конвертувати у простий dict {filename: label}
        return {img: info['relevant'] for img, info in data.items()}
    
    def load_model_results(self, model_name):
        """Завантажити результати моделі"""
        results_path = self.results_dir / f'results_{model_name}.json'
        
        if not results_path.exists():
            return {}
        
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_metrics(self, tp, tn, fp, fn):
        """Розрахунок метрик на основі confusion matrix"""
        total = tp + tn + fp + fn
        
        # Accuracy
        accuracy = (tp + tn) / total if total > 0 else 0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': {
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }
        }
    
    def evaluate_model_on_category(self, model_name, category, ground_truth, model_results):
        """Оцінка однієї моделі на одній категорії"""
        tp = tn = fp = fn = 0
        
        category_results = model_results.get(category, [])
        
        for result in category_results:
            image_name = result['image']
            predicted_label = result['label']
            
            # Отримати true label з ground truth
            true_label = ground_truth.get(image_name)
            
            if true_label is None:
                continue  # Пропустити, якщо немає в ground truth
            
            # Підрахунок confusion matrix
            if predicted_label == 1 and true_label == 1:
                tp += 1
            elif predicted_label == 0 and true_label == 0:
                tn += 1
            elif predicted_label == 1 and true_label == 0:
                fp += 1
            elif predicted_label == 0 and true_label == 1:
                fn += 1
        
        return self.calculate_metrics(tp, tn, fp, fn)
    
    def evaluate_all(self):
        """Оцінка всіх моделей на всіх категоріях"""
        summary = {}
        
        # Отримати список категорій з data/stage_1/
        categories = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Скорочення назв моделей
        model_display_names = {
            'grounding_dino': 'gr_dino',
            'siglip_v2': 'siglip'
        }
        
        for model_name in self.models:
            display_name = model_display_names.get(model_name, model_name)
            print(f"\nEvaluating model: {display_name}")
            model_results = self.load_model_results(model_name)
            
            if not model_results:
                print(f"  ⚠ No results found for {display_name}")
                continue
            
            summary[model_name] = {}
            
            for category in categories:
                ground_truth = self.load_ground_truth(category)
                
                if not ground_truth:
                    print(f"  ⚠ No ground truth for category: {category}")
                    continue
                
                metrics = self.evaluate_model_on_category(
                    model_name, category, ground_truth, model_results
                )
                
                summary[model_name][category] = metrics
                
                print(f"  {category:<25} F1={metrics['f1_score']:<7.4f} "
                      f"Acc={metrics['accuracy']:<7.4f} "
                      f"P={metrics['precision']:<7.4f} "
                      f"R={metrics['recall']:<7.4f}")
        
        return summary
    
    def calculate_average_metrics(self, summary):
        """Розрахунок середніх метрик по всіх категоріях для кожної моделі"""
        averages = {}
        
        for model_name, categories in summary.items():
            if not categories:
                continue
            
            total_metrics = defaultdict(float)
            count = len(categories)
            
            for category, metrics in categories.items():
                total_metrics['accuracy'] += metrics['accuracy']
                total_metrics['precision'] += metrics['precision']
                total_metrics['recall'] += metrics['recall']
                total_metrics['f1_score'] += metrics['f1_score']
            
            averages[model_name] = {
                'accuracy': round(total_metrics['accuracy'] / count, 4),
                'precision': round(total_metrics['precision'] / count, 4),
                'recall': round(total_metrics['recall'] / count, 4),
                'f1_score': round(total_metrics['f1_score'] / count, 4)
            }
        
        return averages
    
    def save_results(self, summary, averages):
        """Збереження результатів"""
        output = {
            'per_category': summary,
            'averages': averages
        }
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {self.output_path}")
    
    def print_summary_table(self, averages, summary):
        """Виведення зведеної таблиці"""
        # Збір загальних TP, TN, FP, FN для кожної моделі
        totals = {}
        for model_name in averages.keys():
            tp_total = tn_total = fp_total = fn_total = 0
            for category_metrics in summary[model_name].values():
                cm = category_metrics['confusion_matrix']
                tp_total += cm['TP']
                tn_total += cm['TN']
                fp_total += cm['FP']
                fn_total += cm['FN']
            totals[model_name] = {
                'TP': tp_total,
                'TN': tn_total,
                'FP': fp_total,
                'FN': fn_total
            }
        
        # Скорочення назв моделей
        model_display_names = {
            'grounding_dino': 'gr_dino',
            'siglip_v2': 'siglip'
        }
        
        # Сортування за F1-Score (спадання)
        sorted_models = sorted(averages.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        width = 97
        print("\n" + "="*width)
        print("AVERAGE METRICS ACROSS ALL CATEGORIES")
        print("="*width)
        
        # Заголовок
        print(f"{'Model':<25} {'F1-Score':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
        print("-"*width)
        
        for model_name, metrics in sorted_models:
            display_name = model_display_names.get(model_name, model_name)
            t = totals[model_name]
            print(f"{display_name:<25} "
                  f"{metrics['f1_score']:>10.4f} "
                  f"{metrics['accuracy']:>10.4f} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{t['TP']:>6} "
                  f"{t['TN']:>6} "
                  f"{t['FP']:>6} "
                  f"{t['FN']:>6}")
        
        print("="*width)
    
    def print_category_breakdown(self, summary):
        """Виведення таблиці метрик по категоріях"""
        if not summary:
            return
        
        # Отримати всі категорії
        all_categories = set()
        for model_results in summary.values():
            all_categories.update(model_results.keys())
        
        categories = sorted(all_categories)
        
        # Скорочення назв моделей
        model_display_names = {
            'grounding_dino': 'gr_dino',
            'siglip_v2': 'siglip'
        }
        
        # Сортування моделей за середнім F1-Score
        model_f1_scores = {}
        for model_name, model_results in summary.items():
            total_f1 = sum(metrics['f1_score'] for metrics in model_results.values())
            avg_f1 = total_f1 / len(model_results) if model_results else 0
            model_f1_scores[model_name] = avg_f1
        
        models = sorted(model_f1_scores.keys(), key=lambda x: model_f1_scores[x], reverse=True)
        
        if not categories or not models:
            return
        
        # Розрахувати ширину колонок
        category_width = 25
        col_width = 12
        total_width = category_width + len(models) * col_width
        
        print("\n" + "="*total_width)
        print("TRUE POSITIVES (TP) BY CATEGORY")
        print("="*total_width)
        
        # Заголовок
        header = f"{'Category':<{category_width}}"
        for model in models:
            display_name = model_display_names.get(model, model)
            header += f"{display_name:>{col_width}}"
        print(header)
        print("-"*total_width)
        
        # TP для кожної категорії
        for category in categories:
            row = f"{category:<{category_width}}"
            for model in models:
                tp = summary.get(model, {}).get(category, {}).get('confusion_matrix', {}).get('TP', 0)
                row += f"{tp:>{col_width}}"
            print(row)
        
        print("="*total_width)
        
        print("\n" + "="*total_width)
        print("TRUE NEGATIVES (TN) BY CATEGORY")
        print("="*total_width)
        
        # Заголовок
        header = f"{'Category':<{category_width}}"
        for model in models:
            display_name = model_display_names.get(model, model)
            header += f"{display_name:>{col_width}}"
        print(header)
        print("-"*total_width)
        
        # TN для кожної категорії
        for category in categories:
            row = f"{category:<{category_width}}"
            for model in models:
                tn = summary.get(model, {}).get(category, {}).get('confusion_matrix', {}).get('TN', 0)
                row += f"{tn:>{col_width}}"
            print(row)
        
        print("="*total_width)


if __name__ == '__main__':
    print("Starting evaluation...")
    
    evaluator = ModelEvaluator()
    summary = evaluator.evaluate_all()
    averages = evaluator.calculate_average_metrics(summary)
    
    evaluator.save_results(summary, averages)
    evaluator.print_summary_table(averages, summary)
    evaluator.print_category_breakdown(summary)
    
    print("\nEvaluation completed!")