import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def cir_log_likelihood(params, rates, dt=1 / 252):
    """
    Функция правдоподобия для CIR модели с постоянной theta

    Parameters
    ----------
    params : array-like
        Параметры модели [alpha, sigma, theta]
    rates : array-like
        Исторические данные ставок
    dt : float
        Временной шаг в годах (по умолчанию 1/252 для торговых дней)

    Returns
    -------
    float
        Отрицательное логарифмическое правдоподобие (для минимизации)
    """
    # Извлекаем параметры модели
    alpha, sigma, theta = params

    # Проверка положительности параметров - необходимое условие для CIR модели
    if alpha <= 0 or sigma <= 0 or theta <= 0:
        return 1e10  # Большое значение для отклонения невалидных параметров

    n = len(rates)
    log_likelihood = 0

    # Вычисляем правдоподобие для каждого перехода между наблюдениями
    for i in range(1, n):
        r_prev, r_curr = rates[i - 1], rates[i]

        # Ожидаемое изменение согласно CIR модели: dr = α(θ - r)dt
        expected_change = alpha * (theta - r_prev) * dt

        # Фактическое изменение ставки
        actual_change = r_curr - r_prev

        # Дисперсия изменения: Var[dr] = σ²r dt
        variance = sigma**2 * max(r_prev, 1e-8) * dt

        # Логарифм плотности нормального распределения для перехода
        if variance > 0:
            log_likelihood += -0.5 * np.log(2 * np.pi * variance) - (
                actual_change - expected_change
            ) ** 2 / (2 * variance)

    return -log_likelihood  # Возвращаем отрицательное значение для минимизации


def calibrate_cir(rates, dt=1 / 252, mode="sofr"):
    """
    Калибровка параметров CIR модели с постоянной theta

    Parameters
    ----------
    rates : array-like
        Исторические данные ставок
    dt : float
        Временной шаг в годах
    mode : str
        Режим калибровки: "auto", "sofr" или "rub"

    Returns
    -------
    scipy.optimize.OptimizeResult
        Результат оптимизации с оптимальными параметрами
    """
    # Базовая статистика для начального приближения
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)

    # Автоматическое определение типа ставок по уровню значений
    if mode == "auto":
        if mean_rate > 0.05:  # Эмпирический порог для разделения рублевых и SOFR ставок
            mode = "rub"
        else:
            mode = "sofr"

    # Настройки калибровки для разных типов ставок
    if mode == "rub":
        # Рублевые ставки: более высокие значения и волатильность
        initial_guess = [1.0, std_rate * 0.5, mean_rate]
        bounds = [(0.001, 10.0), (0.0001, 0.30), (0.0001, 1.0)]
    else:
        # SOFR ставки: низкие значения и волатильность
        initial_guess = [1.0, std_rate * 0.5, mean_rate]
        bounds = [(0.001, 10.0), (0.0001, 0.5), (0.0001, 1.0)]

    # Минимизация отрицательного правдоподобия для нахождения оптимальных параметров
    result = minimize(
        lambda params: cir_log_likelihood(params, rates, dt),
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B",
    )
    return result


def calibrate_theta_from_g_curve(g_curve_data, method="spline"):
    """
    Калибровка θ(t) по G-кривой

    G-кривая представляет собой кривую бескупонной доходности,
    которая используется как долгосрочное среднее в модели CIR

    Parameters
    ----------
    g_curve_data : pd.DataFrame
        DataFrame с колонками ['Date', 'Rate'] - G-кривая доходности
    method : str
        Метод интерполяции: 'spline' или 'piecewise'

    Returns
    -------
    dict
        Словарь с функцией theta(t) и метаданными
    """
    # Преобразуем DataFrame в массив времен и ставок
    # Время в годах от начальной даты
    start_date = g_curve_data["Date"].min()

    # Вычисляем время в годах для каждой даты
    times = (g_curve_data["Date"] - start_date).dt.days / 365.0
    g_rates = g_curve_data["Rate"].values

    if method == "spline":
        # Кубическая сплайн-интерполяция для гладкой функции theta(t)
        theta_spline = CubicSpline(times, g_rates)

        def theta_function(t):
            # Защита от отрицательных значений ставки
            return max(theta_spline(t), 1e-6)

        return {
            "theta_function": theta_function,
            "method": "spline",
            "times": times,
            "rates": g_rates,
        }

    elif method == "piecewise":
        # Кусочно-линейная интерполяция между узлами G-кривой
        def theta_function(t):
            # Поиск ближайших узлов интерполяции
            idx = np.searchsorted(times, t)
            if idx == 0:
                return g_rates[0]  # Экстраполяция влево - первое значение
            elif idx == len(times):
                return g_rates[-1]  # Экстраполяция вправо - последнее значение
            else:
                # Линейная интерполяция между узлами
                t1, t2 = times[idx - 1], times[idx]
                r1, r2 = g_rates[idx - 1], g_rates[idx]
                return r1 + (r2 - r1) * (t - t1) / (t2 - t1)

        return {
            "theta_function": theta_function,
            "method": "piecewise",
            "times": times,
            "rates": g_rates,
        }


def check_feller_condition(alpha, sigma, theta):
    """
    Проверка условия Феллера для CIR модели

    Условие Феллера гарантирует, что процесс ставки никогда не достигнет нуля
    при выполнении неравенства: 2αθ > σ²

    Parameters
    ----------
    alpha : float
        Скорость возврата к среднему
    sigma : float
        Волатильность
    theta : float
        Долгосрочное среднее

    Returns
    -------
    tuple
        (условие_выполнено, подробное_сообщение, краткое_сообщение)
    """
    # Вычисление обеих частей неравенства Феллера
    left_side = 2 * alpha * theta
    right_side = sigma**2

    feller_condition = left_side > right_side

    # Форматирование подробного сообщения с вычислениями
    detailed_message = (
        f"Условие Феллера (2αθ > σ²):\n"
        f"  2 * α * θ = 2 * {alpha:.6f} * {theta:.6f} = {left_side:.6f}\n"
        f"  σ² = {sigma:.6f}² = {right_side:.6f}\n"
        f"  {left_side:.6f} > {right_side:.6f} = {feller_condition}"
    )

    short_message = f"Условие Феллера: {'ВЫПОЛНЕНО' if feller_condition else 'НЕ ВЫПОЛНЕНО'}"
    print(short_message)

    return feller_condition, detailed_message, short_message
