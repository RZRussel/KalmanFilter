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
    def __init__(self, samples: np.array, weights: np.array):
        if len(samples) != len(weights):
            raise ValueError("Sample length must match weights length")

        self._samples = samples
        self._weights = weights

    @property
    def samples(self) -> np.array:
        return self._samples

    @property
    def weights(self) -> np.array:
        return self._weights

    @property
    def mean(self) -> np.array:
        return np.average(a=self._samples, weights=self._weights, axis=0)

    @property
    def covariance(self) -> np.array:
        return np.diag(np.average((self._samples - self.mean) ** 2, weights=self._weights, axis=0))

    def sample(self) -> np.array:
        n = self._samples.shape[0]
        index = randint(0, n-1)
        return self._samples[index]
