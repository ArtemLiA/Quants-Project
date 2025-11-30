from typing import Optional

import numpy as np
import pandas as pd


class FXLogModel:
    def __init__(self, sigma: float):
        """
        Логарифмическая модель для моделирования обменного курса

        Параметры
        ----------
        sigma: float -> годовая волатильность обменного курса
        """
        self.sigma = sigma

    def __call__(
        self,
        start_date: str,
        end_date: str,
        freq: str,
        n_trajectories: int,
        rf: np.ndarray,
        rd: np.ndarray,
        fx0: float,
        dt: float = 1 / 252,
        dW: Optional[np.ndarray] = None,
        return_df: bool = True,
    ):
        """
        Произвести симуляции траекторий обменного курса на основе
        геометрического броуновского движения с учетом разницы процентных ставок

        Параметры
        ----------
        start_date: str -> начальная дата
        end_date: str -> конечная дата
        freq: str -> частота (например, 'B' для рабочих дней)
        n_trajectories: int -> количество траекторий
        rf: np.ndarray -> безрисковые ставки для иностранной валюты
        rd: np.ndarray -> безрисковые ставки для domestic валюты
        fx0: float -> начальное значение обменного курса
        dt: float -> размер шага по времени (по умолчанию 1/252 ~ 1 торговый день)
        dW: Optional[np.ndarray] -> внешние случайные величины
        return_df: bool -> возвращать DataFrame или массивы
        """
        np.random.seed(42)  # Для воспроизводимости результатов

        # Создаем временные метки
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_timestamps = timestamps.size

        # Проверка согласованности размеров
        if len(rf) != n_timestamps or len(rd) != n_timestamps:
            raise ValueError(
                f"Размеры массивов ставок должны соответствовать количеству временных меток. "
                f"rf: {len(rf)}, rd: {len(rd)}, n_timestamps: {n_timestamps}"
            )

        sigma = self.sigma
        log_fx = np.zeros(shape=(n_timestamps, n_trajectories))
        log_fx[0] = np.log(fx0)

        for i in range(1, n_timestamps):
            if dW is None:
                # Генерируем случайный шум
                eps = np.random.randn(n_trajectories)
                dw = eps * np.sqrt(dt)
            else:
                if dW.shape[0] != n_timestamps or dW.shape[1] != n_trajectories:
                    raise ValueError(
                        f"Неверная размерность dW: ожидается ({n_timestamps}, {n_trajectories}), "
                        f"получено {dW.shape}"
                    )
                dw = dW[i]

            # Разница процентных ставок как дрейф
            mu = rf[i] - rd[i]

            # Детерминированная часть: средний тренд с поправкой Ито
            drift = (mu - 0.5 * sigma**2) * dt

            # Стохастическая часть: случайное отклонение
            diffusion = sigma * dw

            # Обновляем лог-курс: d(log FX) = (mu - 0.5*sigma^2)*dt + sigma*dW
            log_fx[i] = log_fx[i - 1] + drift + diffusion

        # Преобразуем обратно в обменный курс
        fx_trajectories = np.exp(log_fx)

        if return_df:
            df = pd.DataFrame(data=fx_trajectories, index=timestamps)
            df = df.rename(columns=lambda x: f"Trajectory_{x + 1}")
            df.columns.name = "Trajectory"
            return df

        return timestamps, fx_trajectories
