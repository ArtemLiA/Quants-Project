from typing import Optional

import numpy as np
import pandas as pd

from .models import CIRModel, FXLogModel
from .utils import generate_multivariate_normal


class RangeAccrualPricingModel:
    """
    Модель для оценки справедливой стоимости дериватива Range Accrual.

    Делает оценку на основе симуляций динамики обменного курса RUB/USD
    и процентных ставок с учетом корреляции между факторами.

    Range Accrual - дериватив, выплата по которому зависит от нахождения
    обменного курса в заданном диапазоне в течение срока действия контракта.

    Attributes
    ----------
    domestic_rate_model : CIRModel
        Модель для domestic процентных ставок (рублевые ставки)
    foreign_rate_model : CIRModel
        Модель для foreign процентных ставок (долларовые ставки)
    fx_model : FXLogModel
        Модель для обменного курса RUB/USD
    cov_matrix : np.ndarray
        Ковариационная матрица 3×3 для коррелированных шоков:
        - [0,0]: шоки domestic ставок
        - [1,1]: шоки foreign ставок
        - [2,2]: шоки обменного курса
        - недиагональные элементы: корреляции между факторами
    """

    def __init__(
        self,
        domestic_rate_model: CIRModel,
        foreign_rate_model: CIRModel,
        fx_model: FXLogModel,
        cov_matrix: np.ndarray,
        seed: Optional[int] = None
    ):
        """
        Инициализация модели прайсинга Range Accrual.

        Parameters
        ----------
        domestic_rate_model : CIRModel
            Модель CIR для domestic ставок (рублевые ставки)
        foreign_rate_model : CIRModel
            Модель CIR для foreign ставок (долларовые ставки)
        fx_model : FXLogModel
            Логарифмическая модель для обменного курса RUB/USD
        cov_matrix : np.ndarray
            Ковариационная матрица 3×3 для коррелированных шоков
        """
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
        dt: float = 1 / 252,
        return_trajectories: bool = False,
    ):
        """
        Оценка справедливой стоимости дериватива Range Accrual.

        Производит симуляцию траекторий обменного курса и рассчитывает вероятность
        нахождения курса в заданном диапазоне для определения стоимости дериватива.

        Parameters
        ----------
        start_date : str
            Дата начала действия контракта (формат 'YYYY-MM-DD')
        end_date : str
            Дата окончания действия контракта (формат 'YYYY-MM-DD')
        nominal : float
            Номинал сделки - максимальная выплата при нахождении в диапазоне
            в течение всего срока жизни сделки
        lower_bound : float | None
            Нижняя граница диапазона. Если None - диапазон не ограничен снизу
        upper_bound : float | None
            Верхняя граница диапазона. Если None - диапазон не ограничен сверху
        r0_domestic : float
            Начальное значение domestic процентной ставки (рубли)
        r0_foreign : float
            Начальное значение foreign процентной ставки (доллары)
        fx0 : float
            Начальное значение обменного курса RUB/USD
        n_trajectories : int, default=10000
            Количество траекторий Монте-Карло для симуляции
        dt : float, default=1/252
            Шаг по времени в годах (по умолчанию 1 торговый день)
        return_trajectories : bool, default=False
            Если True, возвращает словарь с ценой и траекториями симуляции

        Returns
        -------
        float | dict
            Если return_trajectories=False: справедливая стоимость дериватива
            Если return_trajectories=True: словарь с ключами:
                - "price": справедливая стоимость
                - "trajectories": DataFrame с симулированными траекториями курса
        """
        if lower_bound is None:
            lower_bound = -np.inf
        if upper_bound is None:
            upper_bound = np.inf

        n_timestamps = pd.date_range(start_date, end_date, freq="B").size

        mean = np.array([0.0, 0.0, 0.0])
        cov = np.sqrt(dt) * self.cov_matrix

        dW = generate_multivariate_normal(mean, cov, size=(n_timestamps, n_trajectories))

        df_domestic = self.domestic_rate_model(
            start_date,
            end_date,
            n_trajectories=n_trajectories,
            r0=r0_domestic,
            dt=1 / 252,
            dW=dW[:, :, 0],
        )

        df_foreign = self.foreign_rate_model(
            start_date,
            end_date,
            n_trajectories=n_trajectories,
            r0=r0_foreign,
            dt=1 / 252,
            dW=dW[:, :, 1],
        )

        df_fx = self.fx_model(
            start_date,
            end_date,
            n_trajectories=n_trajectories,
            fx0=fx0,
            rf=df_foreign.values,
            rd=df_domestic.values,
            dt=1 / 252,
            dW=dW[:, :, 2],
        )

        df_in_range = (lower_bound <= df_fx) & (df_fx <= upper_bound)
        price = nominal * df_in_range.mean(axis=0).mean()

        if return_trajectories:
            return {"price": price, "trajectories": df_fx}
        return price
