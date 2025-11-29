from typing import Optional

import numpy as np
import pandas as pd


class FXLogModel:
  def __init__(self, sigma: float):
    self.sigma = sigma

  def __call__(
      self,
      start_date: str,
      end_date: str,
      n_trajectories: int,
      rf: np.ndarray, 
      rd: np.ndarray, 
      fx0: float,
      dt: float = 0.05,
      dW: Optional[np.ndarray] = None,  # shape of [n_timestamps, n_trajectories]
      return_df: bool = True
  ):
    timestamps = pd.date_range(start=start_date, end=end_date, freq="B")
    n_timestamps = timestamps.size

    sigma = self.sigma
    log_fx = np.full(shape=(n_timestamps, n_trajectories), fill_value=np.log(fx0))

    for i in range(1, n_timestamps):
      if dW is None:
        dw = np.random.normal(0, np.sqrt(dt), size=n_trajectories)
      else:
        dw = dW[i]

      log_fx[i] = log_fx[i - 1] + (rf[i] - rd[i]) * dt + sigma * dw

    if return_df:
      df = pd.DataFrame(data=np.exp(log_fx), index=timestamps)
      df = df.rename(axis=1, mapper=lambda x: x+1)
      df.columns.name = "trajectory"
      return df
    
    return timestamps, np.exp(log_fx)
  