from typing import Callable, Union

import numpy as np
from scipy.interpolate import CubicSpline


class ThetaFactory:
    """
    Фабрика для создания функций theta(t) с различными параметризациями

    Поддерживает постоянную theta и theta на основе G-кривой
    """

    def __init__(self, params_source: Union[dict, list], theta_kwargs: dict = None):
        """
        Parameters
        ----------
        params_source : dict | list
            Источник параметров для функции theta
        theta_kwargs : dict, optional
            Дополнительные параметры для функции theta
        """
        self.params_source = params_source
        self.theta_kwargs = theta_kwargs or {}

        # Определение типа функции theta из параметров
        if isinstance(params_source, dict):
            self.func_type = params_source.get("type", "constant")
        else:
            self.func_type = "constant"  # По умолчанию для списка параметров

    def constant_theta(self, t: float) -> float:
        """
        Постоянная функция theta(t) = value

        Parameters
        ----------
        t : float
            Время (игнорируется для постоянной функции)

        Returns
        -------
        float
            Постоянное значение theta
        """
        if isinstance(self.params_source, dict):
            return self.params_source.get("value", 0.0)
        else:
            # Для списка параметров theta находится на позиции 2
            return self.params_source[2] if len(self.params_source) > 2 else 0.0

    def g_curve_theta(self, t: float) -> float:
        """
        Функция theta(t) на основе G-кривой

        Parameters
        ----------
        t : float
            Время в годах

        Returns
        -------
        float
            Значение theta, интерполированное по G-кривой
        """
        if "spline_function" in self.theta_kwargs:
            # Использование сплайн-интерполяции
            spline_func = self.theta_kwargs["spline_function"]
            return max(spline_func(t), 1e-6)  # Защита от отрицательных значений
        elif "times" in self.theta_kwargs and "rates" in self.theta_kwargs:
            # Кусочно-линейная интерполяция
            times = self.theta_kwargs["times"]
            rates = self.theta_kwargs["rates"]
            idx = np.searchsorted(times, t)
            if idx == 0:
                return rates[0]
            elif idx == len(times):
                return rates[-1]
            else:
                # Линейная интерполяция между узлами
                t1, t2 = times[idx - 1], times[idx]
                r1, r2 = rates[idx - 1], rates[idx]
                return r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        else:
            # Fallback на постоянное значение
            return self.params_source.get("value", 0.0)

    def get_theta_func(self) -> Callable[[float], float]:
        """
        Возвращает функцию theta(t) соответствующего типа

        Returns
        -------
        Callable[[float], float]
            Функция, принимающая время t и возвращающая значение theta

        Raises
        ------
        ValueError
            Если указан неизвестный тип функции theta
        """
        # Сопоставление типов функций с методами
        func_mapping = {
            "constant": self.constant_theta,
            "g_curve": self.g_curve_theta,
        }

        theta_func = func_mapping.get(self.func_type)

        if theta_func is None:
            raise ValueError(
                f"Неизвестный тип функции theta: {self.func_type}. "
                f"Доступные типы: {list(func_mapping.keys())}"
            )

        return theta_func

    def get_theta_values(self, time_points: np.ndarray) -> np.ndarray:
        """
        Вычисляет значения theta для массива временных точек

        Parameters
        ----------
        time_points : np.ndarray
            Массив временных точек в годах

        Returns
        -------
        np.ndarray
            Массив значений theta для каждого момента времени
        """
        theta_func = self.get_theta_func()
        return np.array([theta_func(t) for t in time_points])

    @classmethod
    def from_constant(cls, theta_value):
        """
        Создание фабрики с постоянной theta

        Parameters
        ----------
        theta_value : float
            Постоянное значение theta

        Returns
        -------
        ThetaFactory
            Фабрика с постоянной функцией theta
        """
        params = {"type": "constant", "value": theta_value}
        return cls(params)

    @classmethod
    def from_g_curve(cls, g_curve_data, method="spline"):
        """
        Создание фабрики на основе G-кривой

        Parameters
        ----------
        g_curve_data : pd.DataFrame
            DataFrame с колонками ['Date', 'Rate'] - данные G-кривой
        method : str
            Метод интерполяции: 'spline' или 'piecewise'

        Returns
        -------
        ThetaFactory
            Фабрика с функцией theta на основе G-кривой
        """
        # Извлекаем даты и ставки из DataFrame
        dates = g_curve_data["Date"]
        rates = g_curve_data["Rate"].values

        # Вычисляем время в годах от начальной даты
        start_date = dates.min()
        times = (dates - start_date).dt.days / 365.0

        if method == "spline":
            # Кубическая сплайн-интерполяция
            spline_func = CubicSpline(times, rates)
            theta_kwargs = {
                "spline_function": spline_func,
                "times": times.values,
                "rates": rates,
                "start_date": start_date,
            }
        else:
            # Кусочно-линейная интерполяция
            theta_kwargs = {"times": times.values, "rates": rates, "start_date": start_date}

        params = {"type": "g_curve", "value": np.mean(rates)}
        return cls(params, theta_kwargs)
