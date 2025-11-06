"""
Normality Check for Stage 4 Data

Призначення:
    Перевірка гіпотези про нормальність розподілу метрики "passed_filter_count"
    для валідації статистичного підходу mean±std у визначенні Reliable N.

Підхід:
    1. Для кожної категорії завантажити analysis_results та dataset_gt
    2. Розрахувати passed_filter_count для кожного профілю
    3. Згрупувати дані за N (кількість сигнальних зображень)
    4. Для кожного N виконати Shapiro-Wilk тест
    5. Знайти максимальний p-value серед усіх N
    6. Інтерпретувати результат за alpha=0.05

Shapiro-Wilk Test:
    - Null hypothesis (H0): дані мають нормальний розподіл
    - Alternative (H1): дані НЕ мають нормальний розподіл
    - Alpha: 0.05 (рівень значущості)
    - Якщо p-value ≤ alpha → відхилити H0 (НЕ нормальний)
    - Якщо p-value > alpha → не відхилити H0 (можливо нормальний)

Константи:
    - FILTER_THRESHOLD: 0.000010
    - ALPHA: 0.05 (5% рівень значущості)

Вхідні дані:
    - results/stage_4/analysis_results_{prefix}.json - image-level scores
    - data/stage_4/dataset_gt_{prefix}_100.json - GT з n_signals

Вихідні дані:
    Тільки консольний вивід (без збереження JSON)

Категорії:
    - Catering (cat)
    - Marine_Activities (mar)
    - Cultural_Excursions (cul)
    - Pet-Friendly Services (pet)

Виведення:
    - Для кожної категорії: max p-value та інтерпретація
    - Висновок: якщо всі p-values ≤ 0.05 → використовувати p10/p90 замість mean±std

Інтерпретація результатів:
    - Якщо НЕ нормальний → mean±std підхід НЕ валідний, використовувати p10/p90
    - Якщо можливо нормальний → можна використовувати обидва підходи
"""

import json
from collections import defaultdict
from scipy import stats


# Константи
FILTER_THRESHOLD = 0.000010
ALPHA = 0.05

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
    print("NORMALITY CHECK: Shapiro-Wilk Test")
    print("="*60)
    print(f"Alpha = {ALPHA}")
    print(f"Якщо p-value <= {ALPHA}, розподіл НЕ є нормальним\n")
    
    for category_name, prefix in CATEGORIES.items():
        print(f"{category_name:<25}", end=" | ")
        
        # Шляхи до файлів
        analysis_results_path = f'results/stage_4/analysis_results_{prefix}.json'
        dataset_gt_path = f'data/stage_4/dataset_gt_{prefix}_100.json'
        
        # Завантаження даних
        try:
            with open(analysis_results_path, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
            
            with open(dataset_gt_path, 'r', encoding='utf-8') as f:
                dataset_gt = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: File not found")
            continue
        
        # Створення мапінгу user_id -> n_signals
        user_to_n = {user['user_id']: user['n_signals'] for user in dataset_gt}
        
        # Збір даних по N
        counts_by_n = defaultdict(list)
        
        for profile in analysis_results:
            user_id = profile['user_id']
            n_signals = user_to_n.get(user_id)
            
            if n_signals is None:
                continue
            
            passed_count = calculate_passed_images_count(profile, category_name)
            counts_by_n[n_signals].append(passed_count)
        
        # Знаходження максимального p-value
        max_p_value = 0.0
        max_n = -1
        
        for n in sorted(counts_by_n.keys()):
            data = counts_by_n[n]
            
            if len(data) < 3:
                continue
            
            # Тест Шапіро-Вілка
            statistic, p_value = stats.shapiro(data)
            
            if p_value > max_p_value:
                max_p_value = p_value
                max_n = n
        
        # Інтерпретація
        if max_p_value <= ALPHA:
            interpretation = "НЕ нормальний"
        else:
            interpretation = "можливо нормальний"
        
        print(f"Max p-value = {max_p_value:.4f} (N={max_n}) → {interpretation}")
    
    print(f"\n{'='*60}")
    print(f"Висновок: якщо всі p-values <= {ALPHA}, використовуємо тільки p10/p90")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()