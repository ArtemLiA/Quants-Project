import numpy as np

from src.calibration.calibrating import calibrate_time_dependent_cir, check_feller_condition


def calibrate_models(rates_data, mode="auto"):
    """Калибровка всех моделей"""
    print("\nКалибровка моделей...")
    models = {}

    # Постоянная theta
    print("1. Модель с постоянной theta...")
    result = calibrate_time_dependent_cir(rates_data, theta_func_type="constant", mode=mode)
    alpha, sigma, theta = result.x
    models["constant"] = {"alpha": alpha, "sigma": sigma, "theta": theta, "logl": -result.fun}
    print(f"   α={alpha:.4f}, σ={sigma:.4f}, θ={theta:.4f}")
    check_feller_condition(alpha, sigma, theta, model_type="constant")

    # Линейная theta
    print("2. Модель с линейной theta...")
    result = calibrate_time_dependent_cir(rates_data, theta_func_type="linear", mode=mode)
    alpha, sigma, a, b = result.x
    models["linear"] = {"alpha": alpha, "sigma": sigma, "a": a, "b": b, "logl": -result.fun}
    print(f"   α={alpha:.4f}, σ={sigma:.4f}, θ(t)={a:.4f} + {b:.4f}·t")
    check_feller_condition(alpha, sigma, theta, model_type="linear")

    # Периодическая theta
    print("3. Модель с периодической theta...")
    result = calibrate_time_dependent_cir(rates_data, theta_func_type="periodic", mode=mode)
    alpha, sigma, a, b, freq = result.x
    models["periodic"] = {
        "alpha": alpha,
        "sigma": sigma,
        "a": a,
        "b": b,
        "freq": freq,
        "logl": -result.fun,
    }
    print(f"   α={alpha:.4f}, σ={sigma:.4f}, θ(t)={a:.4f} + {b:.4f}·sin(2π·{freq:.4f}·t)")
    check_feller_condition(alpha, sigma, theta, model_type="periodic")

    return models


def calibrate_fx_model(log_returns):
    """Калибровка параметров FX модели"""
    print("\nКалибровка FX модели...")

    # Расчет параметров из лог-доходностей
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()
    trading_days = 252

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
