"""
Sensitivity Curve Analysis for Stage 4

Призначення:
    Визначення мінімальної кількості сигнальних зображень (Reliable N)
    для надійного виявлення бізнес-категорії у профілі користувача.
    Побудова кривих чутливості та статистичний аналіз впливу рівня шуму.

Підхід:
    1. Для кожної категорії завантажити analysis_results_{prefix}.json
    2. Застосувати filter_threshold для підрахунку пройдених зображень
    3. Для кожного N (0 до 10):
       - Зібрати passed_filter_count по всіх профілях з N сигнальними зображеннями
       - Розрахувати mean, std, min, max, p10, p90
    4. Визначити Reliable N за двома критеріями:
       - mean±std: N=X(mean-std) > N=0(mean+std)
       - p10/p90: N=X(p10) > N=0(p90)
    5. Побудова sensitivity curve з error bars

Критерії Reliable N:
    mean±std підхід (84% confidence):
    - Noise threshold: N=0(mean + std)
    - Signal threshold: N=X(mean - std)
    - Reliable N: мінімальне X, де signal_threshold > noise_threshold
    
    p10/p90 підхід (90% confidence):
    - Noise threshold: N=0(p90)
    - Signal threshold: N=X(p10)
    - Reliable N: мінімальне X, де signal_threshold > noise_threshold

Константи:
    - FILTER_THRESHOLD: 0.000010
    - N діапазон: 0 до 10 сигнальних зображень
    - Профілі: 50 зображень (N сигнальних + (50-N) шумових)

Вхідні дані:
    - results/stage_4/analysis_results_{prefix}.json - image-level scores
    - data/stage_4/dataset_gt_{prefix}_100.json - GT з n_signals

Вихідні дані:
    - results/stage_4/sensitivity_stats_{prefix}.json - статистика по N
    - results/stage_4/sensitivity_curve_{prefix}.png - графік

Структура sensitivity_stats:
    {
        "0": {
            "mean": 5.2,
            "std": 2.1,
            "min": 0.0,
            "max": 12.0,
            "p10": 2.0,
            "p90": 8.0,
            "mean_minus_std": 3.1,
            "mean_plus_std": 7.3,
            "count": 100
        },
        ...
    }

Категорії:
    - Catering (cat)
    - Marine_Activities (mar)
    - Cultural_Excursions (cul)
    - Pet-Friendly Services (pet)

Виведення:
    - Sensitivity curve plot з error bars (mean ± std)
    - Статистика для кожного N
    - TABLE 1: Reliable N за mean±std підходом
    - TABLE 2: Reliable N за p10/p90 підходом
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# Константи
FILTER_THRESHOLD = 0.000010
CATEGORIES = {
    "Catering": "cat",
    "Marine_Activities": "mar",
    "Cultural_Excursions": "cul",
    "Pet-Friendly Services": "pet"
}


def calculate_passed_images_count(profile_data, target_category):
    """
    Розрахунок кількості зображень, що пройшли фільтр
    
    Returns:
        int: passed_filter_count
    """
    image_details = profile_data['image_details']
    passed_filter_count = 0
    
    for img in image_details:
        score = img['scores'].get(target_category, 0.0)
        if score >= FILTER_THRESHOLD:
            passed_filter_count += 1
    
    return passed_filter_count


def main():
    print("="*60)
    print("STAGE 4: Sensitivity Curve Analysis (Batch Mode)")
    print("="*60)
    
    # Список для фінальної таблиці
    summary_data = []
    
    for category_name, prefix in CATEGORIES.items():
        print(f"\n{'='*20} PROCESSING: {category_name} {'='*20}")
        
        # Динамічні шляхи
        analysis_results_path = f'results/stage_4/analysis_results_{prefix}.json'
        dataset_gt_path = f'data/stage_4/dataset_gt_{prefix}_100.json'
        output_stats_path = f'results/stage_4/sensitivity_stats_{prefix}.json'
        output_plot_path = f'results/stage_4/sensitivity_curve_{prefix}.png'
        
        TARGET_CATEGORY = category_name
        
        # Створення директорії для результатів
        Path(output_stats_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Завантаження даних
        print(f"\nLoading analysis results from: {analysis_results_path}")
        try:
            with open(analysis_results_path, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
        except FileNotFoundError:
            print(f"  Error: File not found - {analysis_results_path}")
            continue
        
        print(f"Loading ground truth from: {dataset_gt_path}")
        try:
            with open(dataset_gt_path, 'r', encoding='utf-8') as f:
                dataset_gt = json.load(f)
        except FileNotFoundError:
            print(f"  Error: File not found - {dataset_gt_path}")
            continue
        
        # Створення словника user_id -> n_signals
        user_to_n = {user['user_id']: user['n_signals'] for user in dataset_gt}
        
        print(f"\nTotal profiles: {len(analysis_results)}")
        print(f"Filter threshold: {FILTER_THRESHOLD}")
        print(f"Target category: {TARGET_CATEGORY}")
        
        # Розрахунок кількості пройдених зображень
        print("\nCalculating passed images count...")
        scores_by_n = defaultdict(list)
        
        for profile in analysis_results:
            user_id = profile['user_id']
            n_signals = user_to_n.get(user_id)
            
            if n_signals is None:
                print(f"Warning: user_id {user_id} not found in ground truth")
                continue
            
            passed_count = calculate_passed_images_count(profile, category_name)
            scores_by_n[n_signals].append(passed_count)
        
        # Розрахунок статистики
        print("\nCalculating statistics...")
        sensitivity_stats = {}
        n_values = []
        mean_values = []
        std_values = []
        
        for n in sorted(scores_by_n.keys()):
            scores = scores_by_n[n]
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))
            p10_score = float(np.percentile(scores, 10))
            p90_score = float(np.percentile(scores, 90))
            mean_minus_std = mean_score - std_score
            mean_plus_std = mean_score + std_score
            
            sensitivity_stats[str(n)] = {
                "mean": round(mean_score, 6),
                "std": round(std_score, 6),
                "min": round(min_score, 6),
                "max": round(max_score, 6),
                "p10": round(p10_score, 6),
                "p90": round(p90_score, 6),
                "mean_minus_std": round(mean_minus_std, 6),
                "mean_plus_std": round(mean_plus_std, 6),
                "count": int(len(scores))
            }
            
            n_values.append(n)
            mean_values.append(mean_score)
            std_values.append(std_score)
            
            print(f"  N={n:2d}: mean={mean_score:5.2f}, std={std_score:4.2f}, mean-std={mean_minus_std:5.2f}, mean+std={mean_plus_std:5.2f}, p10={p10_score:5.2f}, p90={p90_score:5.2f}, n={len(scores):3d}")
        
        # Збереження статистики
        print(f"\nSaving statistics to: {output_stats_path}")
        with open(output_stats_path, 'w', encoding='utf-8') as f:
            json.dump(sensitivity_stats, f, indent=2, ensure_ascii=False)
        
        # Побудова графіку
        print(f"Generating sensitivity curve: {output_plot_path}")
        
        plt.figure(figsize=(12, 7))
        
        # Основна крива з error bars
        plt.errorbar(n_values, mean_values, yerr=std_values, 
                     marker='o', markersize=8, linewidth=2, 
                     capsize=5, capthick=2,
                     label='Mean passed images count ± std')
        
        # Форматування
        plt.xlabel('Number of Signal Images (N)', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Passed Images Count (out of 50)', fontsize=12, fontweight='bold')
        plt.title(f'Sensitivity Curve: {TARGET_CATEGORY}\n(Filter Threshold: {FILTER_THRESHOLD})', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Встановлення діапазону осей
        plt.xlim(-0.5, max(n_values) + 0.5)
        plt.ylim(bottom=0)
        
        # Додавання міток для кожної точки
        for n, mean in zip(n_values, mean_values):
            plt.annotate(f'{mean:.2f}', 
                        xy=(n, mean), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nCalculating reliable N for {category_name}...")
        
        # 1. Отримуємо поріг шуму (mean + 1×std)
        noise_stats = sensitivity_stats.get('0')
        if not noise_stats:
            print("  Error: N=0 (noise) data not found.")
            continue
            
        noise_threshold = noise_stats['mean'] + noise_stats['std']
        print(f"  Noise Threshold (N=0 mean+std): {noise_threshold:.2f}")
        print(f"    (mean={noise_stats['mean']:.2f} + std={noise_stats['std']:.2f})")
        
        # 2. Шукаємо X
        reliable_n = -1
        reliable_n_stats = {}
        
        # Сортуємо N, окрім N=0
        signal_n_values = sorted([int(k) for k in sensitivity_stats.keys() if int(k) > 0])
        
        for n in signal_n_values:
            n_str = str(n)
            signal_threshold = sensitivity_stats[n_str]['mean'] - sensitivity_stats[n_str]['std']
            
            if signal_threshold > noise_threshold:
                reliable_n = n
                reliable_n_stats = sensitivity_stats[n_str]
                print(f"  Found Reliable N: {reliable_n}")
                print(f"    N={reliable_n} (mean-std)={signal_threshold:.2f} > N=0 (mean+std)={noise_threshold:.2f}")
                break
        
        if reliable_n == -1:
            print("  Warning: Reliable N not found (no signal mean-std > noise mean+std).")
        
        # 3. Зберігаємо дані для зведеної таблиці
        noise_threshold_mean_std = noise_stats['mean'] + noise_stats['std']
        signal_threshold_mean_std = reliable_n_stats.get('mean', 0) - reliable_n_stats.get('std', 0) if reliable_n != -1 else 0
        
        # Для p10/p90 підходу
        reliable_n_p10_p90 = -1
        reliable_n_stats_p10_p90 = {}
        noise_threshold_p90 = noise_stats['p90']
        
        for n in signal_n_values:
            n_str = str(n)
            signal_threshold_p10 = sensitivity_stats[n_str]['p10']
            
            if signal_threshold_p10 > noise_threshold_p90:
                reliable_n_p10_p90 = n
                reliable_n_stats_p10_p90 = sensitivity_stats[n_str]
                break
        
        summary_data.append({
            "category": category_name,
            # mean±std підхід
            "n0_mean": noise_stats['mean'],
            "n0_mean_plus_std": noise_threshold_mean_std,
            "reliable_n_mean_std": reliable_n,
            "nx_mean_minus_std": signal_threshold_mean_std,
            "nx_mean": reliable_n_stats.get('mean', 0),
            # p10/p90 підхід
            "n0_p90": noise_stats['p90'],
            "reliable_n_p10_p90": reliable_n_p10_p90,
            "nx_p10": reliable_n_stats_p10_p90.get('p10', 0),
            "nx_mean_p10_p90": reliable_n_stats_p10_p90.get('mean', 0)
        })
    
    # Друк фінальних таблиць
    print(f"\n{'='*100}")
    print("FINAL SUMMARY TABLES")
    print(f"{'='*100}")
    
    # Таблиця 1: mean ± std підхід
    print(f"\n{'='*100}")
    print("TABLE 1: mean ± std approach")
    print(f"{'='*100}")
    print(f"{'Category':<25} | {'N=0 Mean':>10} | {'N=0 +std':>10} | {'Reliable N':>12} | {'N=X -std':>10} | {'N=X Mean':>10}")
    print("-" * 100)
    
    for item in summary_data:
        reliable_n_str = str(item['reliable_n_mean_std']) if item['reliable_n_mean_std'] != -1 else "Not Found"
        print(f"{item['category']:<25} | {item['n0_mean']:>10.2f} | {item['n0_mean_plus_std']:>10.2f} | {reliable_n_str:>12} | {item['nx_mean_minus_std']:>10.2f} | {item['nx_mean']:>10.2f}")
    
    print(f"{'='*100}")
    print("Criterion: N=X(mean-std) > N=0(mean+std)")
    print("Statistical confidence: ~84% (assumes normal distribution, ~16% risk)")
    
    # Таблиця 2: p10/p90 підхід
    print(f"\n{'='*100}")
    print("TABLE 2: p10/p90 approach")
    print(f"{'='*100}")
    print(f"{'Category':<25} | {'N=0 p90':>10} | {'Reliable N':>12} | {'N=X p10':>10} | {'N=X Mean':>10}")
    print("-" * 100)
    
    for item in summary_data:
        reliable_n_str = str(item['reliable_n_p10_p90']) if item['reliable_n_p10_p90'] != -1 else "Not Found"
        print(f"{item['category']:<25} | {item['n0_p90']:>10.2f} | {reliable_n_str:>12} | {item['nx_p10']:>10.2f} | {item['nx_mean_p10_p90']:>10.2f}")
    
    print(f"{'='*100}")
    print("Criterion: N=X(p10) > N=0(p90)")
    print("Statistical confidence: ~90% (ignores 10% outliers on each side, ~10% risk)")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()