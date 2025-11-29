import numpy as np


def print_statistics(rates_data):
    """Вывод статистики данных"""
    rates_data = rates_data * 100
    print("Статистика данных:")
    stats = [
        f"• Объем данных: {len(rates_data)} наблюдений",
        f"• Минимальная ставка: {rates_data.min():.4f}%",
        f"• Максимальная ставка: {rates_data.max():.4f}%",
        f"• Средняя ставка: {rates_data.mean():.4f}%",
        f"• Медиана: {np.median(rates_data):.4f}%",
        f"• Стандартное отклонение: {rates_data.std():.4f}%",
        f"• Коэффициент вариации: {(rates_data.std() / rates_data.mean()):.4f}",
    ]
    print("\n".join(stats))

    q25, q75 = np.percentile(rates_data, [25, 75])
    daily_changes = np.diff(rates_data)

    print(f"\n• 25-й перцентиль: {q25:.4f}%")
    print(f"• 75-й перцентиль: {q75:.4f}%")
    print(f"• IQR: {q75 - q25:.4f}%")

    print("\nАнализ изменений ставок:")
    print(f"• Среднее дневное изменение: {np.mean(daily_changes):.4f}%")
    print(f"• Волатильность изменений: {np.std(daily_changes):.4f}%")
    print(f"• Макс. рост за день: {daily_changes.max():.4f}%")
    print(f"• Макс. падение за день: {daily_changes.min():.4f}%")


def print_best_params(models, best_type):
    """Вывод параметров лучшей модели"""
    params = models[best_type]
    print(f"\nПараметры лучшей модели ({best_type}):")
    print(f"• Alpha (α): {params['alpha']:.4f}")
    print(f"• Sigma (σ): {params['sigma']:.4f}")

    if best_type == "constant":
        print(f"• Theta (θ): {params['theta']:.4f}")
    elif best_type == "linear":
        print(f"• Theta(t): {params['a']:.4f} + {params['b']:.4f}·t")
    else:
        print(f"• Theta(t): {params['a']:.4f} + {params['b']:.4f}·sin(2π·{params['freq']:.4f}·t)")
