import matplotlib.pyplot as plt


def plot_historical_data(sofr_df):
    """График исторических данных"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(sofr_df["Date"], sofr_df["Rate"], linewidth=1)
    plt.title("Исторические данные")
    plt.ylabel("Ставка")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.hist(sofr_df["Rate"], bins=30, alpha=0.7)
    plt.title("Распределение ставок")
    plt.xlabel("Ставка")
    plt.ylabel("Частота")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_simulation(dates, trajectories, model_type, params):
    """Построение графика симуляции"""
    plt.figure(figsize=(12, 8))

    # Цвета для разных моделей
    colors = {"constant": "red", "linear": "orange", "periodic": "purple"}
    color = colors[model_type]

    # Верхний график: траектории
    plt.subplot(2, 1, 1)
    for i in range(min(20, trajectories.shape[1])):
        plt.plot(dates, trajectories.iloc[:, i], alpha=0.3, linewidth=0.8, color=color)

    # Значение theta
    theta_val = params["theta"] if model_type == "constant" else params["a"]
    label = f"θ={theta_val:.4f}" if model_type == "constant" else f"θ(t₀)={theta_val:.4f}"
    plt.axhline(y=theta_val, color=color, linestyle="--", label=label)

    plt.title(f"Симуляция ({model_type} theta)")
    plt.ylabel("Ставка")
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
    plt.axhline(y=theta_val, color=color, linestyle="--", label=label)

    plt.title(f"Статистики симуляции ({model_type} theta)")
    plt.xlabel("Дата")
    plt.ylabel("Ставка")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
