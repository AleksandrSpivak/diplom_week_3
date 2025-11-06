"""
Image-Level Threshold Analysis for Stage 3

Призначення:
    Аналіз впливу порогу фільтрації на image-level класифікацію.
    Перебір порогів та розрахунок Precision/Recall/F1 для кожної категорії.
    Пошук оптимального порогу за Macro F1-Score.

Підхід:
    1. Завантаження analysis_results.json з image-level scores
    2. Для кожного порогу фільтрації:
       - WTA: max_class = argmax(scores)
       - Filter: якщо max_score < threshold AND max_class != "Irrelevant" → "Irrelevant"
       - Інакше залишити max_class
    3. Порівняння predicted labels з GT labels
    4. Розрахунок Precision/Recall/F1 для кожної категорії через sklearn
    5. Вибір порогу з найвищим Macro F1

Метрики:
    Per-class:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (P * R) / (P + R)
    - Support: кількість зображень класу в GT
    
    Macro Average:
    - Macro Precision: середня Precision по всіх класах
    - Macro Recall: середній Recall по всіх класах
    - Macro F1: середній F1 по всіх класах (основна метрика для вибору)

Простір пошуку:
    filter_thresholds: [0.0, 0.000010, 0.000020, 0.000030, 0.000040, 
                        0.000050, 0.000075, 0.000100]

Вхідні дані:
    - results/stage_3/analysis_results.json - image-level scores

Вихідні дані:
    - results/stage_3/image_level_threshold_analysis.json - результати аналізу

Структура результату:
    {
        "best_threshold": {
            "threshold": 0.000050,
            "metrics": {
                "Catering": {
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1": 0.90,
                    "support": 150
                },
                ...
                "macro_avg": {
                    "precision": 0.87,
                    "recall": 0.85,
                    "f1": 0.86
                }
            }
        },
        "all_results": [...]
    }

Категорії:
    - Catering
    - Marine_Activities
    - Cultural_Excursions
    - Pet-Friendly Services
    - Irrelevant

Виведення:
    - Precision vs Threshold table (по всіх категоріях)
    - Recall vs Threshold table (по всіх категоріях)
    - F1-Score vs Threshold table (по всіх категоріях)
    - Best threshold з найвищим Macro F1
"""

import json
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


class ImageThresholdAnalyzer:
    # Константи
    FILTER_THRESHOLDS = [0.0, 0.000010, 0.000020, 0.000030, 0.000040, 
                         0.000050, 0.000075, 0.000100]
    
    CATEGORIES = ["Catering", "Marine_Activities", "Cultural_Excursions", 
                  "Pet-Friendly Services", "Irrelevant"]
    
    PREFIX_TO_CATEGORY = {
        'cat': 'Catering',
        'mar': 'Marine_Activities',
        'cul': 'Cultural_Excursions',
        'pet': 'Pet-Friendly Services',
        'irr': 'Irrelevant'
    }
    
    def __init__(self, analysis_results_path: str):
        """Ініціалізація аналізатора"""
        print(f"Loading analysis results from: {analysis_results_path}")
        with open(analysis_results_path, 'r', encoding='utf-8') as f:
            self.profiles = json.load(f)
        
        print(f"Loaded {len(self.profiles)} profiles")
        
        # Підрахунок загальної кількості зображень
        total_images = sum(len(profile['image_details']) for profile in self.profiles)
        print(f"Total images: {total_images}")
        
        # Підготовка GT один раз
        print("Preparing Ground Truth labels...")
        self.all_gt_labels, self.all_image_scores = self._prepare_gt_and_scores()
    
    def _prepare_gt_and_scores(self) -> tuple:
        """Підготовка GT міток та scores для всіх зображень"""
        gt_labels = []
        image_scores = []
        
        for profile in self.profiles:
            for image_detail in profile['image_details']:
                filename = image_detail['filename']
                scores = image_detail['scores']
                
                # GT
                prefix = filename.split('_')[0]
                gt_label = self.PREFIX_TO_CATEGORY.get(prefix, 'Irrelevant')
                
                gt_labels.append(gt_label)
                image_scores.append(scores)
        
        return gt_labels, image_scores
    
    def _predict_with_filter(self, scores: Dict[str, float], threshold: float) -> str:
        """
        Прогноз з фільтрацією (WTA + Filter)
        
        Args:
            scores: Словник scores для зображення
            threshold: Поріг фільтрації
        
        Returns:
            Predicted категорія
        """
        # WTA
        max_class = max(scores, key=scores.get)
        max_score = scores[max_class]
        
        # Фільтрація
        if max_score < threshold and max_class != 'Irrelevant':
            return 'Irrelevant'
        else:
            return max_class
    
    def _calculate_metrics(self, gt_labels: List[str], 
                          pred_labels: List[str]) -> Dict:
        """Розрахунок метрик для всіх категорій"""
        # sklearn precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_labels, 
            pred_labels, 
            labels=self.CATEGORIES,
            average=None,
            zero_division=0
        )
        
        metrics = {}
        for i, category in enumerate(self.CATEGORIES):
            metrics[category] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Macro average
        metrics['macro_avg'] = {
            'precision': float(precision.mean()),
            'recall': float(recall.mean()),
            'f1': float(f1.mean())
        }
        
        return metrics
    
    def run_threshold_analysis(self) -> List[Dict]:
        """
        Основний метод аналізу порогів
        
        Returns:
            Список результатів для всіх порогів
        """
        print(f"\n{'='*70}")
        print("THRESHOLD ANALYSIS")
        print(f"{'='*70}")
        print(f"Testing {len(self.FILTER_THRESHOLDS)} thresholds")
        print(f"{'='*70}\n")
        
        results = []
        
        for threshold in tqdm(self.FILTER_THRESHOLDS, desc="Testing thresholds"):
            # Генерація прогнозів для поточного порогу
            pred_labels = []
            for scores in self.all_image_scores:
                pred = self._predict_with_filter(scores, threshold)
                pred_labels.append(pred)
            
            # Розрахунок метрик
            metrics = self._calculate_metrics(self.all_gt_labels, pred_labels)
            
            # Збереження результату
            results.append({
                'threshold': threshold,
                'metrics': metrics
            })
        
        return results
    
    def print_summary_table(self, results: List[Dict]):
        """Вивід зведеної таблиці"""
        print("\n" + "="*120)
        print("PRECISION vs THRESHOLD")
        print("="*120)
        
        # Заголовок
        header = f"{'Threshold':<12}"
        for cat in self.CATEGORIES:
            cat_short = cat[:15]
            header += f"{cat_short:>16}"
        header += f"{'Macro Avg':>16}"
        print(header)
        print("-"*120)
        
        # Рядки
        for result in results:
            threshold = result['threshold']
            metrics = result['metrics']
            
            row = f"{threshold:<12.6f}"
            for cat in self.CATEGORIES:
                precision = metrics[cat]['precision']
                row += f"{precision:>16.4f}"
            row += f"{metrics['macro_avg']['precision']:>16.4f}"
            print(row)
        
        print("\n" + "="*120)
        print("RECALL vs THRESHOLD")
        print("="*120)
        
        # Заголовок
        print(header)
        print("-"*120)
        
        # Рядки
        for result in results:
            threshold = result['threshold']
            metrics = result['metrics']
            
            row = f"{threshold:<12.6f}"
            for cat in self.CATEGORIES:
                recall = metrics[cat]['recall']
                row += f"{recall:>16.4f}"
            row += f"{metrics['macro_avg']['recall']:>16.4f}"
            print(row)
        
        print("\n" + "="*120)
        print("F1-SCORE vs THRESHOLD")
        print("="*120)
        
        # Заголовок
        print(header)
        print("-"*120)
        
        # Рядки
        for result in results:
            threshold = result['threshold']
            metrics = result['metrics']
            
            row = f"{threshold:<12.6f}"
            for cat in self.CATEGORIES:
                f1 = metrics[cat]['f1']
                row += f"{f1:>16.4f}"
            row += f"{metrics['macro_avg']['f1']:>16.4f}"
            print(row)
        
        print("="*120)
    
    def find_best_threshold(self, results: List[Dict]) -> Dict:
        """Пошук найкращого порогу за Macro F1"""
        best = max(results, key=lambda x: x['metrics']['macro_avg']['f1'])
        return best
    
    def print_best_threshold(self, best: Dict):
        """Вивід найкращого порогу"""
        print(f"\n{'='*70}")
        print("BEST THRESHOLD")
        print(f"{'='*70}")
        print(f"Threshold: {best['threshold']:.6f}")
        print(f"Macro F1: {best['metrics']['macro_avg']['f1']:.4f}")
        print(f"Macro Precision: {best['metrics']['macro_avg']['precision']:.4f}")
        print(f"Macro Recall: {best['metrics']['macro_avg']['recall']:.4f}")
        print(f"{'='*70}")


def main():
    print("="*70)
    print("STAGE 3: IMAGE-LEVEL THRESHOLD ANALYSIS")
    print("="*70)
    
    # Шляхи
    analysis_results_path = 'results/stage_3/analysis_results.json'
    output_path = 'results/stage_3/image_level_threshold_analysis.json'
    
    # Створення директорії для результатів
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Ініціалізація аналізатора
    analyzer = ImageThresholdAnalyzer(analysis_results_path)
    
    # Запуск аналізу
    results = analyzer.run_threshold_analysis()
    
    # Вивід зведеної таблиці
    analyzer.print_summary_table(results)
    
    # Пошук найкращого порогу
    best = analyzer.find_best_threshold(results)
    analyzer.print_best_threshold(best)
    
    # Збереження результатів
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_threshold': best,
            'all_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()