import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_historical_data(sofr_df):
    """График исторических данных"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(sofr_df["Date"], sofr_df["Rate"] * 100, linewidth=1)
    plt.title("Исторические данные")
    plt.ylabel("Ставка (%)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.hist(sofr_df["Rate"] * 100, bins=30, alpha=0.7)
    plt.title("Распределение ставок")
    plt.xlabel("Ставка (%)")
    plt.ylabel("Частота")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_simulation(
    dates, trajectories, model_type=None, params=None, title=None, asset_type="rate"
):
    """
    Универсальная функция для построения графиков симуляции

    Parameters
    ----------
    dates : array-like
        Массив дат или временных меток
    trajectories : pd.DataFrame или np.ndarray
        Матрица траекторий (время x траектории)
    model_type : str, optional
        Тип модели ("constant", "linear", "periodic", "fx")
    params : dict, optional
        Параметры модели
    title : str, optional
        Заголовок графика
    asset_type : str
        Тип актива ("rate" для ставок, "fx" для обменного курса)
    """
    plt.figure(figsize=(12, 8))

    # Цвета для разных моделей
    colors = {
        "constant": "red",
        "linear": "orange",
        "periodic": "purple",
        "fx": "blue",
        None: "blue",  # Цвет по умолчанию
    }
    color = colors.get(model_type, "blue")

    # Преобразуем trajectories в DataFrame если это массив
    if isinstance(trajectories, np.ndarray):
        if len(trajectories.shape) == 1:
            trajectories = pd.DataFrame(trajectories, index=dates)
        else:
            trajectories = pd.DataFrame(trajectories.T, index=dates)

    if asset_type == "rate":
        trajectories = trajectories * 100

    # Верхний график: траектории
    plt.subplot(2, 1, 1)
    for i in range(min(20, trajectories.shape[1])):
        plt.plot(dates, trajectories.iloc[:, i], alpha=0.3, linewidth=1.5)

    # Добавляем линии параметров если они указаны
    if params is not None:
        if model_type == "constant" and "theta" in params:
            theta_val = params["theta"] * 100
            plt.axhline(
                y=theta_val,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"θ={theta_val:.4f}%",
            )
        elif model_type in ["linear", "periodic"] and "a" in params:
            theta_val = params["a"] * 100
            label = (
                f"θ(t₀)={theta_val:.4f}%" if model_type == "linear" else f"θ(t₀)={theta_val:.4f}%"
            )
            plt.axhline(y=theta_val, color=color, linestyle="--", linewidth=2, label=label)
        elif model_type == "fx" and "initial_rate" in params:
            initial_rate = params["initial_rate"]
            plt.axhline(
                y=initial_rate,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Начальный курс={initial_rate:.4f}",
            )

    # Настройки заголовка и осей
    if title:
        plt.title(title)
    elif model_type:
        model_names = {
            "constant": "постоянная theta",
            "linear": "линейная theta",
            "periodic": "периодическая theta",
            "fx": "обменный курс",
        }
        plt.title(f"Симуляция ({model_names.get(model_type, model_type)})")
    else:
        plt.title("Симуляция траекторий")

    ylabel = "Курс (RUB/USD)" if asset_type == "fx" else "Ставка (%)"
    plt.ylabel(ylabel)
    if params or model_type:
        plt.legend()
    plt.grid(True, alpha=0.3)

    # Нижний график: статистики
    plt.subplot(2, 1, 2)
    mean_traj = trajectories.mean(axis=1)
    std_traj = trajectories.std(axis=1)

    plt.plot(dates, mean_traj, "b-", linewidth=2, label="Средняя траектория")
    plt.fill_between(
        dates, mean_traj - std_traj, mean_traj + std_traj, alpha=0.3, color="blue", label="±1σ"
    )

    # Добавляем параметры на нижний график
    if params is not None:
        if model_type == "constant" and "theta" in params:
            theta_val = params["theta"] * 100
            plt.axhline(
                y=theta_val, color=color, linestyle="--", linewidth=2, label=f"θ={theta_val:.4f}%"
            )
        elif model_type in ["linear", "periodic"] and "a" in params:
            theta_val = params["a"] * 100
            plt.axhline(
                y=theta_val,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"θ(t₀)={theta_val:.4f}%",
            )
        elif model_type == "fx" and "initial_rate" in params:
            initial_rate = params["initial_rate"]
            plt.axhline(
                y=initial_rate,
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"Начальный курс={initial_rate:.4f}",
            )

    plt.title("Статистики симуляции")
    plt.xlabel("Дата")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_fx_simulation(dates, trajectories, title="Симуляция курса RUB/USD"):
    """
    Построение графика симуляции курса RUB/USD
    """
    plt.figure(figsize=(12, 8))

    # Верхний график: траектории
    plt.subplot(2, 1, 1)
    for i in range(min(20, trajectories.shape[1])):
        plt.plot(dates, trajectories.iloc[:, i], alpha=0.3, linewidth=0.8, color="blue")

    # Средняя траектория
    mean_traj = trajectories.mean(axis=1)
    plt.plot(dates, mean_traj, "r-", linewidth=2, label="Средняя траектория")

    plt.title(f"{title} - Траектории")
    plt.ylabel("Курс RUB/USD")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Нижний график: статистики
    plt.subplot(2, 1, 2)
    std_traj = trajectories.std(axis=1)

    plt.plot(dates, mean_traj, "r-", linewidth=2, label="Средняя траектория")
    plt.fill_between(
        dates, mean_traj - std_traj, mean_traj + std_traj, alpha=0.3, color="red", label="±1σ"
    )

    plt.title(f"{title} - Статистики")
    plt.xlabel("Дата")
    plt.ylabel("Курс RUB/USD")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_fx_analysis(fx_hist, log_returns):
    """
    Построение графиков анализа курса RUB/USD
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # График 1: Исторический курс
    axes[0, 0].plot(fx_hist["Date"], fx_hist["Rate"], linewidth=1, color="blue")
    axes[0, 0].set_title("Исторический курс RUB/USD")
    axes[0, 0].set_xlabel("Дата")
    axes[0, 0].set_ylabel("Курс (RUB за 1 USD)")
    axes[0, 0].grid(True, alpha=0.3)

    # График 2: Распределение курса
    axes[0, 1].hist(fx_hist["Rate"], bins=50, alpha=0.7, color="green")
    axes[0, 1].set_title("Распределение курса RUB/USD")
    axes[0, 1].set_xlabel("Курс (RUB за 1 USD)")
    axes[0, 1].set_ylabel("Частота")
    axes[0, 1].grid(True, alpha=0.3)

    # График 3: Лог-доходности
    axes[1, 0].plot(fx_hist["Date"].iloc[1:], log_returns, linewidth=0.5, color="red", alpha=0.7)
    axes[1, 0].set_title("Дневные лог-доходности курса RUB/USD")
    axes[1, 0].set_xlabel("Дата")
    axes[1, 0].set_ylabel("Лог-доходность")
    axes[1, 0].grid(True, alpha=0.3)

    # График 4: Распределение лог-доходностей
    axes[1, 1].hist(log_returns, bins=50, alpha=0.7, color="orange")
    axes[1, 1].set_title("Распределение дневных лог-доходностей")
    axes[1, 1].set_xlabel("Лог-доходность")
    axes[1, 1].set_ylabel("Частота")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
