import numpy as np
import pandas as pd

from src.models.cir import CIRModel
from src.models.logfx import FXLogModel
from src.theta.theta_factory import ThetaFactory


def create_model(params, model_type):
    """
    Универсальное создание модели

    Parameters
    ----------
    params : dict
        Параметры модели
    model_type : str
        Тип модели: "constant", "g_curve", "fx"

    Returns
    -------
    model : CIRModel или FXLogModel
        Созданная модель
    """
    if model_type in ["constant", "g_curve"]:
        # Создание CIR модели для ставок
        if model_type == "constant":
            theta_func = ThetaFactory.from_constant(params["theta"]).get_theta_func()
        else:  # g_curve
            theta_func = params["theta_function"]

        return CIRModel(theta_func=theta_func, alpha=params["alpha"], sigma=params["sigma"])

    elif model_type == "fx":
        # Создание FX модели
        return FXLogModel(sigma=params["sigma_annual"])

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def simulate_model(
    model, initial_value, start_date, end_date, n_paths=50, freq="B", model_type=None, **kwargs
):
    """
    Универсальная симуляция траекторий

    Parameters
    ----------
    model : CIRModel или FXLogModel
        Модель для симуляции
    initial_value : float
        Начальное значение (ставка или курс)
    start_date : str или pd.Timestamp
        Дата начала симуляции
    end_date : str или pd.Timestamp
        Дата окончания симуляции
    n_paths : int
        Количество траекторий
    freq : str
        Частота данных ('B' для бизнес-дней)
    model_type : str, optional
        Тип модели ("rate" для ставок, "fx" для курса)
    **kwargs : dict
        Дополнительные параметры:
        - Для ставок: не требуются
        - Для FX: rf_rates, rd_rates или mu_annual

    Returns
    -------
    dates : pd.DatetimeIndex
        Даты симуляции
    trajectories : pd.DataFrame
        Траектории симуляции
    """

    # Автоматическое определение типа модели если не указан
    if model_type is None:
        if isinstance(model, CIRModel):
            model_type = "rate"
        elif isinstance(model, FXLogModel):
            model_type = "fx"
        else:
            raise ValueError("Не удалось определить тип модели")

    if model_type == "rate":
        # Симуляция ставок
        trajectories = model(
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            n_trajectories=n_paths,
            r0=initial_value,
            return_df=True,
        )
        return trajectories.index, trajectories

    elif model_type == "fx":
        # Симуляция обменного курса
        rf_rates = kwargs.get("rf_rates")
        rd_rates = kwargs.get("rd_rates")
        mu_annual = kwargs.get("mu_annual")

        # Создаем временные метки для расчета количества шагов
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_dates = len(dates)

        if rf_rates is not None and rd_rates is not None:
            # Симуляция с заданными ставками
            if len(rf_rates) != n_dates or len(rd_rates) != n_dates:
                raise ValueError(
                    f"Размеры массивов ставок должны соответствовать количеству дней. "
                    f"rf: {len(rf_rates)}, rd: {len(rd_rates)}, days: {n_dates}"
                )
            rf_array = rf_rates
            rd_array = rd_rates
        elif mu_annual is not None:
            # Упрощенная симуляция с постоянными ставками
            # mu_annual = rf - rd, для простоты rd = 0
            rf_array = np.full(n_dates, mu_annual)
            rd_array = np.zeros(n_dates)
        else:
            raise ValueError("Для FX модели необходимо указать rf_rates/rd_rates или mu_annual")

        trajectories = model(
            start_date=start_date,
            end_date=end_date,
            freq=freq,
            n_trajectories=n_paths,
            rf=rf_array,
            rd=rd_array,
            fx0=initial_value,
            return_df=True,
        )
        return trajectories.index, trajectories

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
