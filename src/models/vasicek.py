from typing import Optional

import numpy as np
import pandas as pd


class vasicek_model:
  def __init__(self, theta: float, alpha: float, sigma: float):
    """
    Vasicek-модель для моделирования мгновенной процентной ставки

    Параметры
    ----------
    theta: float -> уровень среднего
    alpha: float -> коэффициент стремления к среднему
    sigma: float -> волатильность
    """
    self.theta = theta
    self.alpha = alpha
    self.sigma = sigma

  def __call__(
      self,
      start_date: str,
      end_date: str,
      n_trajectories: int,
      r0: float,
      dt: float = 0.05,
      dW: Optional[np.ndarray] = None,  # shape of [n_timestamps, n_trajectories]
      return_df: bool = True
  ):
    """
    Произвести симуляции траекторий на основе Vasicek-модели при помощи
    разностной схемы Эйлера-Мураяны
    """
    timestamps = pd.date_range(start=start_date, end=end_date, freq="B")
    n_timestamps = timestamps.size
    
    theta = self.theta
    alpha = self.alpha
    sigma = self.sigma

    r = np.full(shape=(n_timestamps, n_trajectories), fill_value=r0)

    for i in range(1, n_timestamps):
      if dW is None:
        dw = np.random.normal(0, np.sqrt(dt), size=n_trajectories)
      else:
        dw = dW[i]

      r[i] = r[i - 1] + alpha * (theta - r[i - 1]) * dt + sigma * dw
      r[i] = np.maximum(r[i], 0.0)

    if return_df:
      df = pd.DataFrame(data=r, index=timestamps)
      df = df.rename(axis=1, mapper=lambda x: x+1)
      df.columns.name = "Trajectory"
      return df
    
    return timestamps, r
