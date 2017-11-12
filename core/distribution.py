import numpy as np
from typing import List


class BaseDistribution:
    @property
    def mean(self) -> np.array:
        raise NotImplementedError()

    @property
    def covariance(self) -> np.array:
        raise NotImplementedError()


class GaussDistribution(BaseDistribution):
    def __init__(self, mean: np.array, covariance: np.array):
        self._mean = mean
        self._covariance = covariance

    @property
    def mean(self) -> np.array:
        return self._mean

    @property
    def covariance(self) -> np.array:
        return self._covariance

    @staticmethod
    def create_independent(mean: List[float], variances: List[float]):
        return GaussDistribution(np.array(mean, dtype=float), np.diag(variances))

    @staticmethod
    def create_1d(mean: float, variance: float):
        return GaussDistribution(np.array([mean], dtype=float), np.diag([variance]))