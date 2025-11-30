import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def cir_log_likelihood(params, rates, dt=1 / 252, theta_func=None):
    """
    Функция правдоподобия для CIR модели (поддерживает постоянную и изменяющуюся theta)

    Parameters
    ----------
    params : array-like
        Параметры модели [alpha, sigma] или [alpha, sigma, theta]
    rates : array-like
        Исторические данные ставок
    dt : float
        Временной шаг в годах (по умолчанию 1/252 для торговых дней)
    theta_func : callable, optional
        Функция theta(t). Если None, используется постоянная theta из params

    Returns
    -------
    float
        Отрицательное логарифмическое правдоподобие (для минимизации)
    """
    if theta_func is None:
        # Постоянная theta: params = [alpha, sigma, theta]
        alpha, sigma, theta_val = params
    else:
        # Изменяющаяся theta: params = [alpha, sigma]
        alpha, sigma = params

    # Проверка положительности параметров
    if alpha <= 0 or sigma <= 0:
        return 1e10
    if theta_func is None and theta_val <= 0:
        return 1e10

    n = len(rates)
    log_likelihood = 0

    # Вычисляем правдоподобие для каждого перехода между наблюдениями
    for i in range(1, n):
        r_prev, r_curr = rates[i - 1], rates[i]

        # Определяем theta для текущего момента времени
        if theta_func is not None:
            # Время в годах от начала наблюдений
            t = i * dt
            current_theta = theta_func(t)
        else:
            current_theta = theta_val

        # Ожидаемое изменение согласно CIR модели: dr = α(θ - r)dt
        expected_change = alpha * (current_theta - r_prev) * dt

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


def calibrate_cir(rates, dt=1 / 252, mode="sofr", theta_func=None):
    """
    Калибровка параметров CIR модели (поддерживает постоянную и изменяющуюся theta)

    Parameters
    ----------
    rates : array-like
        Исторические данные ставок
    dt : float
        Временной шаг в годах
    mode : str
        Режим калибровки: "auto", "sofr" или "rub"
    theta_func : callable, optional
        Функция theta(t). Если None - калибруется постоянная theta

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

    if theta_func is not None:
        # Калибровка с изменяющейся theta: оптимизируем только alpha и sigma
        if mode == "rub":
            initial_guess = [1.0, std_rate * 0.5]
            bounds = [(0.001, 10.0), (0.0001, 0.30)]
        else:
            initial_guess = [1.0, std_rate * 0.5]
            bounds = [(0.001, 10.0), (0.0001, 0.5)]

        # Минимизация с передачей theta_func
        result = minimize(
            lambda params: cir_log_likelihood(params, rates, dt, theta_func),
            initial_guess,
            bounds=bounds,
            method="L-BFGS-B",
        )
    else:
        # Калибровка с постоянной theta: оптимизируем alpha, sigma и theta
        if mode == "rub":
            initial_guess = [1.0, std_rate * 0.5, mean_rate]
            bounds = [(0.001, 10.0), (0.0001, 0.30), (0.0001, 1.0)]
        else:
            initial_guess = [1.0, std_rate * 0.5, mean_rate]
            bounds = [(0.001, 10.0), (0.0001, 0.5), (0.0001, 1.0)]

        # Минимизация без theta_func
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

    # Вычисляем время в годах для каждой даты и преобразуем в numpy array
    times = ((g_curve_data["Date"] - start_date).dt.days / 365.0).values
    g_rates = g_curve_data["Rate"].values

    if method == "spline":
        # Кубическая сплайн-интерполяция для гладкой функции theta(t)
        theta_spline = CubicSpline(times, g_rates)

        def spline_theta_function(t):
            # Защита от отрицательных значений ставки
            return max(theta_spline(t), 1e-8)

        return {
            "theta_function": spline_theta_function,
            "method": method,
            "times": times,
            "rates": g_rates,
            "start_date": start_date,
        }

    elif method == "piecewise":
        # Кусочно-линейная интерполяция между узлами G-кривой
        def piecewise_theta_function(t):
            # Поиск ближайших узлов интерполяции
            idx = np.searchsorted(times, t)
            if idx == 0:
                return max(g_rates[0], 1e-8)  # Защита от отрицательных значений
            elif idx == len(times):
                return max(g_rates[-1], 1e-8)  # Защита от отрицательных значений
            else:
                # Линейная интерполяция между узлами
                t1, t2 = times[idx - 1], times[idx]
                r1, r2 = g_rates[idx - 1], g_rates[idx]
                interpolated_value = r1 + (r2 - r1) * (t - t1) / (t2 - t1)
                return max(interpolated_value, 1e-8)  # Защита от отрицательных значений

        return {
            "theta_function": piecewise_theta_function,
            "method": method,
            "times": times,
            "rates": g_rates,
            "start_date": start_date,
        }


def check_feller_condition(alpha, sigma, theta, time_points=None):
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
    theta : float or callable
        Долгосрочное среднее (постоянное или функция theta(t))
    time_points : array-like, optional
        Временные точки для проверки (если theta - функция)
    """

    # Если theta - функция, проверяем на всех временных точках
    if callable(theta):
        if time_points is None:
            # По умолчанию проверяем на горизонте 1 года
            time_points = np.linspace(0, 1, 50)

        conditions = []
        feller_info = []
        for t in time_points:
            theta_val = theta(t)
            left_side = 2 * alpha * theta_val
            right_side = sigma**2
            condition_met = left_side > right_side
            conditions.append(condition_met)
            feller_info.append((t, theta_val, left_side, right_side, condition_met))

        feller_condition = all(conditions)

        if feller_condition:
            print("Условие Феллера: ВЫПОЛНЕНО ДЛЯ ВСЕХ t")
        else:
            print("Условие Феллера: НАРУШЕНО В НЕКОТОРЫХ ТОЧКАХ")

    else:
        # Постоянная theta
        left_side = 2 * alpha * theta
        right_side = sigma**2
        feller_condition = left_side > right_side

        if feller_condition:
            print(f"Условие Феллера: ВЫПОЛНЕНО (2αθ = {left_side:.6f} > σ² = {right_side:.6f})")
        else:
            print(f"Условие Феллера: НЕ ВЫПОЛНЕНО (2αθ = {left_side:.6f} ≤ σ² = {right_side:.6f})")
