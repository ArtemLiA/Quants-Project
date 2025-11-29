from typing import Callable, Union

import numpy as np


class ThetaFactory:
    """
    Фабрика для создания функций theta(t) с различными параметризациями.

    Класс предоставляет единый интерфейс для создания функций долгосрочного среднего
    в CIR-модели, которые могут зависеть от времени.

    Параметры
    ----------
    params_source : dict | list
        Источник параметров для функции theta. Может быть словарем с ключами:
        - 'type': тип функции ('constant', 'linear', 'periodic')
        - 'value': значение для постоянной функции
        - 'a': коэффициент сдвига для линейной и периодической функций
        - 'b': коэффициент наклона/амплитуды
        - 'freq': частота для периодической функции
        Или список параметров в порядке: [alpha, sigma, theta_params...]
    theta_kwargs : dict, optional
        Дополнительные параметры для функции theta, например:
        - 'phase': фаза для периодической функции
    """

    def __init__(self, params_source: Union[dict, list], theta_kwargs: dict = None):
        self.params_source = params_source
        self.theta_kwargs = theta_kwargs or {}

        # Определяем тип функции theta
        if isinstance(params_source, dict):
            self.func_type = params_source.get("type", "constant")
        else:
            # Если передан список параметров, предполагаем постоянную theta
            self.func_type = "constant"

    def _extract_base_params(self) -> tuple:
        """
        Извлекает базовые параметры a, b, freq из источника параметров.

        Возвращает
        -------
        tuple
            Кортеж параметров (a, b, freq) в зависимости от типа источника
        """
        if isinstance(self.params_source, dict):
            # Для словаря извлекаем параметры по ключам
            a = self.params_source.get("a", self.params_source.get("value", 0.0))
            b = self.params_source.get("b", 0.0)
            freq = self.params_source.get("freq", 1.0)
        else:
            # Для списка извлекаем параметры по позициям
            # Предполагаем формат: [alpha, sigma, theta_value] или [alpha, sigma, a, b, ...]
            if len(self.params_source) > 2:
                a = self.params_source[2]  # theta_value или параметр a
            else:
                a = 0.0

            if len(self.params_source) > 3:
                b = self.params_source[3]  # параметр b для линейной/периодической
            else:
                b = 0.0

            if len(self.params_source) > 4:
                freq = self.params_source[4]  # частота для периодической
            else:
                freq = 1.0

        return a, b, freq

    def constant_theta(self, t: float) -> float:
        """
        Постоянная функция theta(t) = value.

        Параметры
        ----------
        t : float
            Время (игнорируется для постоянной функции)

        Возвращает
        -------
        float
            Постоянное значение theta
        """
        if isinstance(self.params_source, dict):
            return self.params_source.get("value", 0.0)
        else:
            # Для списка параметров theta_value находится на позиции 2
            return self.params_source[2] if len(self.params_source) > 2 else 0.0

    def linear_theta(self, t: float) -> float:
        """
        Линейная функция theta(t) = a + b * t.

        Параметры
        ----------
        t : float
            Время в годах

        Возвращает
        -------
        float
            Линейно зависящее от времени значение theta
        """
        a, b, _ = self._extract_base_params()
        return a + b * t

    def periodic_theta(self, t: float) -> float:
        """
        Периодическая функция theta(t) = a + b * sin(2π * freq * t + phase).

        Параметры
        ----------
        t : float
            Время в годах

        Возвращает
        -------
        float
            Периодически зависящее от времени значение theta
        """
        a, b, freq = self._extract_base_params()
        phase = self.theta_kwargs.get("phase", 0.0)  # Фаза из дополнительных параметров
        return a + b * np.sin(2 * np.pi * freq * t + phase)

    def get_theta_func(self) -> Callable[[float], float]:
        """
        Возвращает функцию theta(t) соответствующего типа.

        Возвращает
        -------
        Callable[[float], float]
            Функция, принимающая время t и возвращающая значение theta

        Исключения
        ----------
        ValueError
            Если указан неизвестный тип функции theta
        """
        # Словарь соответствия типов функций методам класса
        func_mapping = {
            "constant": self.constant_theta,
            "linear": self.linear_theta,
            "periodic": self.periodic_theta,
        }

        # Получаем функцию по типу
        theta_func = func_mapping.get(self.func_type)

        if theta_func is None:
            raise ValueError(
                f"Неизвестный тип функции theta: {self.func_type}. "
                f"Доступные типы: {list(func_mapping.keys())}"
            )

        return theta_func

    def get_theta_values(self, time_points: np.ndarray) -> np.ndarray:
        """
        Вычисляет значения theta для массива временных точек.

        Параметры
        ----------
        time_points : np.ndarray
            Массив временных точек в годах

        Возвращает
        -------
        np.ndarray
            Массив значений theta для каждого момента времени
        """
        theta_func = self.get_theta_func()
        return np.array([theta_func(t) for t in time_points])
