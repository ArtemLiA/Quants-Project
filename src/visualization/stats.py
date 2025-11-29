import numpy as np


def print_statistics(data, data_type="rate", log_returns=None):
    """
    Универсальная функция вывода статистики данных

    Parameters
    ----------
    data : pd.Series или np.ndarray
        Массив данных (ставки или курсы)
    data_type : str
        Тип данных: "rate" для ставок, "fx" для обменного курса
    log_returns : pd.Series или np.ndarray, optional
        Лог-доходности (только для data_type="fx")
    """

    if data_type == "rate":
        # Для ставок: преобразуем в проценты
        data_display = data * 100
        unit = "%"
        asset_name = "ставка"
        changes_name = "ставок"
    else:
        # Для курса: оставляем как есть
        data_display = data
        unit = "RUB/USD"
        asset_name = "курс"
        changes_name = "курса"

    print(f"СТАТИСТИКА ДАННЫХ ({asset_name.upper()}):")
    print("=" * 50)

    # Основная статистика
    stats = [
        f"• Объем данных: {len(data)} наблюдений",
        f"• Минимальный {asset_name}: {data_display.min():.4f} {unit}",
        f"• Максимальный {asset_name}: {data_display.max():.4f} {unit}",
        f"• Средний {asset_name}: {data_display.mean():.4f} {unit}",
        f"• Медиана: {np.median(data_display):.4f} {unit}",
        f"• Стандартное отклонение: {data_display.std():.4f} {unit}",
    ]

    # Добавляем коэффициент вариации только если среднее не нулевое
    if data_display.mean() != 0:
        cv = data_display.std() / data_display.mean()
        stats.append(f"• Коэффициент вариации: {cv:.4f}")

    print("\n".join(stats))

    # Квартили
    q25, q75 = np.percentile(data_display, [25, 75])
    print(f"\n• 25-й перцентиль: {q25:.4f} {unit}")
    print(f"• 75-й перцентиль: {q75:.4f} {unit}")
    print(f"• IQR: {q75 - q25:.4f} {unit}")

    # Анализ изменений
    if data_type == "rate":
        # Для ставок: абсолютные изменения
        daily_changes = np.diff(data_display)
        changes_unit = "%"
    else:
        # Для курса: используем переданные лог-доходности
        if log_returns is not None:
            daily_changes = log_returns
            changes_unit = ""
        else:
            # Если лог-доходности не переданы, рассчитываем абсолютные изменения
            daily_changes = np.diff(data_display)
            changes_unit = unit

    print(f"\nАнализ изменений {changes_name}:")
    print(f"• Среднее дневное изменение: {np.mean(daily_changes):.4f}{changes_unit}")
    print(f"• Волатильность изменений: {np.std(daily_changes):.4f}{changes_unit}")
    print(f"• Макс. рост за день: {daily_changes.max():.4f}{changes_unit}")
    print(f"• Макс. падение за день: {daily_changes.min():.4f}{changes_unit}")


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
