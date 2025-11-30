import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.simulate_model import create_model, simulate_model


def compare_models(models, initial_rate, start_date, end_date):
    """
    Сравнение моделей CIR с постоянной theta и на основе G-кривой

    Parameters
    ----------
    models : dict
        Словарь с параметрами калиброванных моделей
    initial_rate : float
        Начальное значение ставки для симуляции
    start_date : str or pd.Timestamp
        Дата начала симуляции
    end_date : str or pd.Timestamp
        Дата окончания симуляции

    Returns
    -------
    str
        Тип лучшей модели ('constant' или 'g_curve')
    """
    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)

    # Вывод статистики калибровки
    for model_type, params in models.items():
        if model_type == "constant":
            print(f"Постоянная θ: LogL = {params['logl']:.4f}")
            print(f"  θ = {params['theta']:.4f} ({params['theta'] * 100:.2f}%)")
        else:
            # Для G-кривой выводим информацию о данных
            if "start_date" in params:
                start_dt = params["start_date"]
            print(f"G-кривая θ: LogL = {params['logl']:.4f}")
            print(
                f"  Диапазон ставок: {params['rates'].min() * 100:.2f}% - {params['rates'].max() * 100:.2f}%"
            )

    # Подготовка данных для графиков
    print("\nПостроение графиков сравнения...")
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Вычисляем время в годах от начала симуляции
    start_dt = pd.Timestamp(start_date)
    time_points = (dates - start_dt).days / 365.0

    plt.figure(figsize=(12, 10))
    colors = {"constant": "red", "g_curve_spline": "green", "g_curve_piecewise": "blue"}
    labels = {
        "constant": "Постоянная θ",
        "g_curve_spline": "G-кривая θ (сплайн)",
        "g_curve_piecewise": "G-кривая θ (кусочно-линейная)",
    }

    # Верхний график: средние траектории симуляций
    plt.subplot(2, 1, 1)
    for model_type in models:
        if "g_curve" in model_type:
            # Создание модели CIR с theta на основе G-кривой
            from src.models.cir import CIRModel

            model = CIRModel(
                theta_func=models[model_type]["theta_function"],
                alpha=models[model_type]["alpha"],
                sigma=models[model_type]["sigma"],
            )
        else:
            # Создание модели CIR с постоянной theta
            model = create_model(models[model_type], model_type)

        # Симуляция траекторий
        _, trajectories = simulate_model(model, initial_rate, start_date, end_date)
        mean_traj = trajectories.mean(axis=1)

        color = colors.get(model_type, "gray")
        label = labels.get(model_type, model_type)

        plt.plot(
            dates,
            mean_traj * 100,
            color=color,
            linewidth=2,
            label=label,
        )

        # Доверительный интервал
        std_traj = trajectories.std(axis=1)
        plt.fill_between(
            dates,
            (mean_traj - std_traj) * 100,
            (mean_traj + std_traj) * 100,
            alpha=0.2,
            color=color,
        )

    plt.title("Сравнение средних траекторий")
    plt.ylabel("Ставка (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Нижний график: Функции theta(t)
    plt.subplot(2, 1, 2)

    # Постоянная theta
    if "constant" in models:
        theta_const = models["constant"]["theta"] * 100
        plt.axhline(
            y=theta_const,
            color=colors["constant"],
            linestyle="-",
            linewidth=2,
            label=f"Постоянная θ = {theta_const:.2f}%",
        )

    # G-кривые theta
    for model_type in models:
        if "g_curve" in model_type:
            color = colors.get(model_type, "gray")
            label = labels.get(model_type, model_type)

            # Для G-кривой время отсчитывается от start_date симуляции
            theta_values = np.array([
                models[model_type]["theta_function"](t) * 100 for t in time_points
            ])
            plt.plot(
                dates,
                theta_values,
                color=color,
                linewidth=2,
                label=label,
            )

    plt.title("Функции theta(t)")
    plt.xlabel("Дата")
    plt.ylabel("Theta (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
