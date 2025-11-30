import numpy as np

from src.calibration.calibrating import (
    calibrate_cir,
    calibrate_theta_from_g_curve,
    check_feller_condition,
)
from src.theta.theta_factory import ThetaFactory


def calibrate_models(rates_data, mode="auto", g_curve_data=None):
    """
    Калибровка моделей CIR с постоянной theta и на основе G-кривой

    Parameters
    ----------
    rates_data : array-like
        Исторические данные ставок для калибровки
    mode : str
        Режим калибровки: "auto", "sofr" или "rub"
    g_curve_data : pd.DataFrame, optional
        DataFrame с колонками ['Date', 'Rate'] - данные G-кривой

    Returns
    -------
    dict
        Словарь с параметрами калиброванных моделей
    """
    print("\nКалибровка моделей...")
    models = {}

    # Калибровка модели с постоянной theta
    print("1. Модель с постоянной theta...")
    result = calibrate_cir(rates_data, mode=mode)
    alpha, sigma, theta = result.x
    models["constant"] = {
        "alpha": alpha,
        "sigma": sigma,
        "theta": theta,
        "logl": -result.fun,
    }
    print(f"   α={alpha:.4f}, σ={sigma:.4f}, θ={theta:.4f}")
    check_feller_condition(alpha, sigma, theta)

    # Калибровка модели с theta на основе G-кривой
    if g_curve_data is not None:
        print("2. Модель с theta на основе G-кривой ...")
        theta_result = calibrate_theta_from_g_curve(g_curve_data, method="spline")

        models["g_curve_spline"] = {
            "alpha": alpha,
            "sigma": sigma,
            "theta_function": theta_result["theta_function"],
            "logl": models["constant"]["logl"],
            "method": theta_result["method"],
            "times": theta_result["times"],
            "rates": theta_result["rates"],
        }
        print(f"   На основе G-кривой, метод: {theta_result['method']}")
        check_feller_condition(alpha, sigma, theta_result["theta_function"])

        print("3. Модель с theta на основе G-кривой ...")
        theta_result = calibrate_theta_from_g_curve(g_curve_data, method="piecewise")

        models["g_curve_piecewise"] = {
            "alpha": alpha,
            "sigma": sigma,
            "theta_function": theta_result["theta_function"],
            "logl": models["constant"]["logl"],
            "method": theta_result["method"],
            "times": theta_result["times"],
            "rates": theta_result["rates"],
        }
        print(f"   На основе G-кривой, метод: {theta_result['method']}")
        check_feller_condition(alpha, sigma, theta_result["theta_function"])

    return models


def create_g_curve_model(g_curve_data, alpha=None, sigma=None):
    """
    Создание модели CIR на основе G-кривой

    Parameters
    ----------
    g_curve_data : pd.DataFrame
        DataFrame с колонками ['Date', 'Rate'] - данные G-кривой
    alpha : float, optional
        Скорость возврата к среднему
    sigma : float, optional
        Волатильность

    Returns
    -------
    dict
        Параметры модели CIR с theta на основе G-кривой
    """
    theta_factory = ThetaFactory.from_g_curve(g_curve_data)

    # Значения по умолчанию если не указаны
    if alpha is None:
        alpha = 1.0
    if sigma is None:
        rates = g_curve_data["Rate"].values
        sigma = np.std(rates) * 0.5

    return {
        "alpha": alpha,
        "sigma": sigma,
        "theta_function": theta_factory.get_theta_func(),
        "type": "g_curve",
        "g_curve_info": {
            "date_range": f"{g_curve_data['Date'].min()} - {g_curve_data['Date'].max()}",
            "rate_range": f"{g_curve_data['Rate'].min() * 100:.2f}% - {g_curve_data['Rate'].max() * 100:.2f}%",
            "data_points": len(g_curve_data),
        },
    }


def calibrate_fx_model(log_returns):
    """
    Калибровка параметров FX модели из лог-доходностей

    Parameters
    ----------
    log_returns : array-like
        Логарифмические доходности обменного курса

    Returns
    -------
    dict
        Параметры FX модели (дневные и годовые)
    """
    print("\nКалибровка FX модели...")

    # Расчет параметров из лог-доходностей
    mu_daily = log_returns.mean()  # Средняя дневная доходность
    sigma_daily = log_returns.std()  # Дневная волатильность
    trading_days = 252  # Количество торговых дней в году

    # Приведение к годовым значениям
    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)

    fx_params = {
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
    }

    print(f"   μ_дневной={mu_daily:.6f}, σ_дневной={sigma_daily:.6f}")
    print(f"   μ_годовой={mu_annual:.6f}, σ_годовой={sigma_annual:.6f}")

    return fx_params
