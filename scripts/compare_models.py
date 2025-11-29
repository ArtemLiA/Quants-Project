import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.simulate_model import create_model, simulate_model
from src.theta.theta_factory import ThetaFactory


def compare_models(models, sofr_df, initial_rate):
    """Сравнение моделей"""
    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")

    # Вывод качества моделей
    for model_type, params in models.items():
        print(f"{model_type.capitalize()} θ: LogL = {params['logl']:.4f}")

    best_type = max(models.keys(), key=lambda x: models[x]["logl"])
    best_logl = models[best_type]["logl"]

    print(f"\nЛучшая модель: {best_type} theta")
    print(f"Log-правдоподобие: {best_logl:.4f}")

    # Расчет улучшений
    const_logl = models["constant"]["logl"]
    if best_type != "constant":
        improvement = ((best_logl - const_logl) / const_logl) * 100
        print(f"Улучшение vs постоянная: +{improvement:.4f}")

    # Построение графика сравнения
    print("\nГрафик сравнения моделей...")
    dates = pd.date_range(start=sofr_df["Date"].max(), periods=253, freq="B")
    time_points = np.linspace(0, 1, len(dates))

    plt.figure(figsize=(12, 8))
    colors = {"constant": "red", "linear": "orange", "periodic": "purple"}

    # Верхний график: средние траектории
    plt.subplot(2, 1, 1)
    for model_type in models:
        model = create_model(models[model_type], model_type)
        _, trajectories = simulate_model(model, initial_rate, sofr_df)
        mean_traj = trajectories.mean(axis=1)

        color = colors[model_type]
        is_best = model_type == best_type
        plt.plot(
            dates,
            mean_traj,
            color=color,
            linestyle="-" if is_best else "--",
            linewidth=2 if is_best else 1.5,
            label=f"{model_type}{' (лучшая)' if is_best else ''}",
        )

    plt.title("Сравнение средних траекторий")
    plt.ylabel("Ставка")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Нижний график: функции theta через ThetaFactory
    plt.subplot(2, 1, 2)
    for model_type, params in models.items():
        factory = ThetaFactory({"type": model_type, **params})
        theta_values = factory.get_theta_values(time_points)
        plt.plot(dates, theta_values, color=colors[model_type], label=f"{model_type} θ(t)")

    plt.title("Функции theta(t)")
    plt.xlabel("Дата")
    plt.ylabel("Theta")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return best_type
