import numpy as np
from scipy.optimize import minimize

from src.theta.theta_factory import ThetaFactory


def time_dependent_cir_log_likelihood(params, rates, dt=1 / 252, factory=None):
    """
    Функция правдоподобия для CIR модели с зависящим от времени theta

    Параметры
    ----------
    params: array-like -> [alpha, sigma, theta_param1, theta_param2, ...]
    rates: array-like -> исторические данные ставок
    dt: float -> временной шаг
    factory: ThetaFactory -> экземпляр класса ThetaFactory для получения функции theta(t)
    """
    # Выбираем функцию theta
    if factory is None:
        factory = ThetaFactory(params, {})

    factory.params_source = params
    theta_func = factory.get_theta_func()

    # Извлекаем параметры alpha, sigma
    alpha, sigma = params[:2]

    # Проверка положительности параметров
    if alpha <= 0 or sigma <= 0:
        return 1e10

    n = len(rates)
    log_likelihood = 0
    time_points = np.arange(n) * dt  # Временные точки в годах

    for i in range(1, n):
        r_prev, r_curr = rates[i - 1], rates[i]
        t_prev = time_points[i - 1]  # Время для предыдущего наблюдения

        # Получаем theta для текущего момента времени
        theta_t = theta_func(t_prev)

        # Проверяем, что theta положительна
        if theta_t <= 0:
            return 1e10

        # Ожидаемое изменение ставки
        expected_change = alpha * (theta_t - r_prev) * dt
        actual_change = r_curr - r_prev
        variance = sigma**2 * max(r_prev, 1e-8) * dt

        # Вычисляем логарифмическое правдоподобие
        if variance > 0:
            log_likelihood += -0.5 * np.log(2 * np.pi * variance) - (
                actual_change - expected_change
            ) ** 2 / (2 * variance)

    return -log_likelihood  # Возвращаем отрицательное значение для минимизации


def calibrate_time_dependent_cir(
    rates,
    dt=1 / 252,
    theta_func_type="constant",
    initial_guess=None,
    theta_kwargs=None,
    mode="auto",  # "auto", "sofr", "rub"
):
    """
    Калибровка параметров CIR модели с зависящим от времени theta
    Универсальная функция для SOFR и рублевых ставок

    Параметры
    ----------
    rates: array-like -> исторические данные ставок
    dt: float -> временной шаг
    theta_func_type: str -> тип функции theta(t)
    initial_guess: array-like -> начальное приближение параметров
    theta_kwargs: dict -> дополнительные параметры для функции theta
    mode: str -> режим калибровки: "auto", "sofr" или "rub"
    """
    theta_kwargs = theta_kwargs or {}

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)

    # Автоматическое определение режима по данным
    if mode == "auto":
        if mean_rate > 0.0005:  # если средняя ставка > 0.05%, считаем что это рубль
            mode = "rub"
        else:
            mode = "sofr"

    # Конфигурации для разных режимов
    if mode == "rub":
        # Рублевый режим: [kappa, sigma, theta] для constant, [kappa, sigma, a, b...] для time-dependent
        config = {
            "constant": (
                [1.0, std_rate * 0.5, mean_rate],
                [(0.001, 10.0), (0.0001, 0.30), (0.0001, 1.0)],
            ),
            "linear": (
                [1.0, std_rate * 0.5, mean_rate, 0.01],
                [(0.001, 10.0), (0.0001, 1.0), (0.0001, 0.30), (-0.1, 0.1)],
            ),
            "periodic": (
                [1.0, std_rate * 0.5, mean_rate, 0.01, 1.0],
                [(0.001, 10.0), (0.0001, 1.0), (0.0001, 0.30), (0.0001, 0.1), (0.001, 10.0)],
            ),
        }
    else:
        # SOFR режим (по умолчанию)
        config = {
            "constant": (
                [1.0, std_rate * 0.5, mean_rate],
                [(0.001, 10.0), (0.0001, 0.5), (0.0001, 1.0)],
            ),
            "linear": (
                [1.0, std_rate * 0.5, mean_rate, 0.01],
                [(0.001, 10.0), (0.0001, 0.5), (0.0001, 1.0), (None, None)],
            ),
            "periodic": (
                [1.0, std_rate * 0.5, mean_rate, 0.01, 1.0],
                [(0.001, 10.0), (0.0001, 0.5), (0.0001, 1.0), (0.0001, 1.0), (0.001, 10.0)],
            ),
        }

    initial_guess, bounds = config.get(theta_func_type, (None, None))

    factory = ThetaFactory({"type": theta_func_type}, theta_kwargs)

    def likelihood_wrapper(params):
        return time_dependent_cir_log_likelihood(params, rates, dt, factory)

    result = minimize(likelihood_wrapper, initial_guess, bounds=bounds, method="L-BFGS-B")
    return result


def check_feller_condition(alpha, sigma, theta, model_type="constant"):
    """
    Проверка условия Феллера для CIR модели

    Parameters
    ----------
    alpha : float
        Скорость возврата к среднему
    sigma : float
        Волатильность
    theta : float
        Параметр theta (может быть постоянным, начальным или средним значением)
    model_type : str
        Тип модели: "constant", "linear", "periodic"

    Returns
    -------
    tuple : (bool, str, str)
        (условие_выполнено, подробное_сообщение, краткое_сообщение)
    """

    # Вычисляем обе части неравенства
    left_side = 2 * alpha * theta
    right_side = sigma**2

    feller_condition = left_side > right_side

    # Формируем сообщения в зависимости от типа модели
    if model_type == "constant":
        condition_desc = "2αθ > σ²"
        theta_desc = "θ"
    elif model_type == "linear":
        condition_desc = "2αθ(t₀) > σ²"
        theta_desc = "θ(t₀)"
    else:  # periodic
        condition_desc = "2αθ_сред > σ²"
        theta_desc = "θ_сред"

    detailed_message = (
        f"Условие Феллера ({condition_desc}):\n"
        f"  2 * α * {theta_desc} = 2 * {alpha:.6f} * {theta:.6f} = {left_side:.6f}\n"
        f"  σ² = {sigma:.6f}² = {right_side:.6f}\n"
        f"  {left_side:.6f} > {right_side:.6f} = {feller_condition}"
    )

    short_message = f"Условие Феллера ({condition_desc}): {'ВЫПОЛНЕНО' if feller_condition else 'НЕ ВЫПОЛНЕНО'}"
    print(short_message)

    return feller_condition, detailed_message, short_message
