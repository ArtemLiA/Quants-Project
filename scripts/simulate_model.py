import pandas as pd

from src.models.cir import CIRModel
from src.theta.theta_factory import ThetaFactory


def create_model(params, model_type):
    """Создание CIR модели"""
    if model_type == "constant":
        theta_params = {"type": "constant", "value": params["theta"]}
    elif model_type == "linear":
        theta_params = {"type": "linear", "a": params["a"], "b": params["b"]}
    else:  # periodic
        theta_params = {
            "type": "periodic",
            "a": params["a"],
            "b": params["b"],
            "freq": params["freq"],
        }

    theta_func = ThetaFactory(theta_params).get_theta_func()
    return CIRModel(alpha=params["alpha"], sigma=params["sigma"], theta_func=theta_func)


def simulate_model(model, initial_rate, sofr_df, n_days=252, n_paths=50):
    """Симуляция траекторий"""
    dates = pd.date_range(start=sofr_df["Date"].max(), periods=n_days + 1, freq="B")
    trajectories = model(
        start_date=dates[0],
        end_date=dates[-1],
        n_trajectories=n_paths,
        r0=initial_rate,
        freq="B",
    )
    return dates, trajectories
