import numpy as np


class BaseDistribution:
    @property
    def mean(self) -> float:
        raise NotImplementedError()

    @property
    def variance(self) -> float:
        raise NotImplementedError()


class GaussDistribution(BaseDistribution):
    def __init__(self, mean: float, variance: float):
        if variance < 0.0:
            raise ValueError("Variance can't be negative")

        self._mean = mean
        self._variance = variance

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._variance


class MultidimensionalDistribution:
    def __init__(self, distributions=None, mean=None, covariance=None):
        if distributions is not None:
            n = len(distributions)

            if n == 0:
                raise ValueError("At least one dimensional must be provided")

            self._distributions = distributions

            mean = [d.mean for d in distributions]
            self._mean = np.array(mean, dtype=float)

            covariance = [d.variance for d in distributions]
            self._covariance = np.diag(covariance)
        elif mean is not None and covariance is not None:
            self._mean = mean
            self._covariance = covariance
        else:
            raise ValueError("Distributions or means/covariance must be provided")

    @property
    def mean(self) -> np.array:
        return self._mean

    @property
    def covariance(self) -> np.array:
        return self._covariance
