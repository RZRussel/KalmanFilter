import numpy as np
from typing import List
from random import randint


class BaseDistribution:
    @property
    def mean(self) -> np.array:
        raise NotImplementedError()

    @property
    def covariance(self) -> np.array:
        raise NotImplementedError()

    def sample(self) -> np.array:
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

    def sample(self) -> np.array:
        return np.random.multivariate_normal(self._mean, self._covariance)

    @staticmethod
    def create_independent(mean: List[float], variances: List[float]):
        return GaussDistribution(np.array(mean, dtype=float), np.diag(variances))

    @staticmethod
    def create_1d(mean: float, variance: float):
        return GaussDistribution(np.array([mean], dtype=float), np.diag([variance]))


class NaiveSampleDistribution(BaseDistribution):
    def __init__(self, samples: np.array):
        if samples.shape[0] < 1:
            raise ValueError("Sample list must contain more than one item")

        self._samples = samples

    @property
    def samples(self) -> np.array:
        return self._samples

    @property
    def mean(self) -> np.array:
        return np.mean(a=self._samples, axis=0)

    @property
    def covariance(self) -> np.array:
        return np.cov(self._samples.transpose())

    def sample(self) -> np.array:
        n = self._samples.shape[0]
        index = randint(0, n-1)
        return self._samples[index]
