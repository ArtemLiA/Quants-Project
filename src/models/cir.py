from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd


class CIRModel:
    def __init__(
          self,
          theta_func: Callable[[float], float],
          alpha: float,
          sigma: float
    ):
        """
        CIR-модель для моделирования мгновенной процентной ставки

        Параметры
        ----------
        theta_func: Callable[[float], float] -> функция времени, возвращающая уровень среднего
        alpha: float -> коэффициент стремления к среднему
        sigma: float -> волатильность
        """
        self.theta_func = theta_func
        self.alpha = alpha
        self.sigma = sigma

    def __call__(
        self,
        start_date: str,
        end_date: str,
        freq: str,
        n_trajectories: int,
        r0: float,
        dt: float = 1 / 252,
        dW: Optional[np.ndarray] = None,
        return_df: bool = True,
    ):
        """
        Произвести симуляции траекторий на основе CIR-модели при помощи
        разностной схемы Эйлера-Мураяны с зависящим от времени theta
        """
        np.random.seed(42)  # Для воспроизводимости результатов

        # Создаем временные метки
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_timestamps = timestamps.size

        if dW is not None:
            assert dW.shape == (n_timestamps, n_trajectories)

        # Преобразуем даты в числовые значения (годы от начальной даты)
        start_timestamp = pd.Timestamp(start_date)
        time_years = np.array([(ts - start_timestamp).days * dt for ts in timestamps])

        # Вычисляем theta для каждого момента времени
        theta_values = np.array([self.theta_func(t) for t in time_years])

        # Случайные колебания
        r = np.zeros(shape=(n_timestamps, n_trajectories))
        r[0] = r0


        alpha = self.alpha
        sigma = self.sigma

        for i in range(1, n_timestamps):
            if dW is None:
                dw = np.random.normal(0, np.sqrt(dt), size=n_trajectories)
            else:
                dw = dW[i]

            r[i] = (
                r[i - 1]
                + alpha * (theta_values[i - 1] - r[i - 1]) * dt
                + sigma * np.sqrt(r[i - 1]) * dw
            )
            r[i] = np.maximum(r[i], 0.0)

        if return_df:
            df = pd.DataFrame(data=r, index=timestamps)
            df = df.rename(columns=lambda x: f"Trajectory_{x + 1}")
            df.columns.name = "Trajectory"
            return df

        return timestamps, r
