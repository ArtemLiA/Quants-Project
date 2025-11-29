import numpy as np
import pandas as pd

from .models import CIRModel
from .models import FXLogModel

from .utils import generate_multivariate_normal


class RangeAccrualPricingModel:
    def __init__(
        self,
        domestic_rate_model: CIRModel,
        foreign_rate_model: CIRModel,
        fx_model: FXLogModel,
        cov_matrix: np.ndarray,
    ):
        self.domestic_rate_model = domestic_rate_model
        self.foreign_rate_model = foreign_rate_model
        self.fx_model = fx_model
        self.cov_matrix = cov_matrix

    def price(
        self, 
        start_date: str,
        end_date: str,
        nominal: float,
        lower_bound: float | None,
        upper_bound: float | None,
        r0_domestic: float,
        r0_foreign: float,
        fx0: float,            
        n_trajectories: int = 10000,
        dt: float = 1/252,
        return_trajectories: bool = False
    ):
        if lower_bound is None:
            lower_bound = -np.inf
        if upper_bound is None:
            upper_bound = np.inf

        n_timestamps = pd.date_range(start_date, end_date, freq="B").size

        mean = np.array([0.0, 0.0, 0.0])
        cov = np.sqrt(dt) * self.cov_matrix

        dW = generate_multivariate_normal(
            mean, cov, size=(n_timestamps, n_trajectories)
        )

        df_domestic = self.domestic_rate_model(
            start_date, end_date, n_trajectories=n_trajectories,
            r0=r0_domestic, dt=1 / 252, dW=dW[:, :, 0] 
        )

        df_foreign = self.foreign_rate_model(
            start_date, end_date, n_trajectories=n_trajectories,
            r0=r0_foreign, dt=1 / 252, dW=dW[:, :, 1]
        )

        df_fx = self.fx_model(
            start_date, end_date, n_trajectories=n_trajectories,
            fx0=fx0, rf=df_foreign.values, rd=df_domestic.values,
            dt = 1 / 252, dW=dW[:, :, 2]
        )

        df_in_range = (lower_bound <= df_fx) & (df_fx <= upper_bound)
        price = nominal * df_in_range.mean(axis=0).mean()

        if return_trajectories:
            return {
                "price": price,
                "trajectories": df_fx
            }
        return price
