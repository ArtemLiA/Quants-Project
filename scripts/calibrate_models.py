from src.calibration.calibrating import calibrate_time_dependent_cir


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

    # Линейная theta
    print("2. Модель с линейной theta...")
    result = calibrate_time_dependent_cir(rates_data, theta_func_type="linear", mode=mode)
    alpha, sigma, a, b = result.x
    models["linear"] = {"alpha": alpha, "sigma": sigma, "a": a, "b": b, "logl": -result.fun}
    print(f"   α={alpha:.4f}, σ={sigma:.4f}, θ(t)={a:.4f} + {b:.4f}·t")

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

    return models
