import numpy as np
from unittest import TestCase
from core.distribution import GaussDistribution, MultidimensionalDistribution


class TestGaussDistribution(TestCase):
    def test_initialization(self):
        mean = 10.0
        var = 0.5
        distr = GaussDistribution(mean, var)

        self.assertAlmostEqual(first=mean, second=distr.mean, delta=1e-6)
        self.assertAlmostEqual(first=var, second=distr.variance, delta=1e-6)


class TestMultidimensionalDistribution(TestCase):
    def test_distr_initialization(self):
        means = [10.0, 15.0]
        variances = [0.1, 0.5]

        distrs = [GaussDistribution(means[i], variances[i]) for i in range(0, len(means))]
        multi_distr = MultidimensionalDistribution(distrs)

        self.assertTrue(np.array_equal(np.array(means, dtype=float), multi_distr.mean))
        self.assertTrue(np.array_equal(np.diag(variances), multi_distr.covariance))

    def test_means_covariance_initialization(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = MultidimensionalDistribution(mean=np.array(mean, dtype=float), covariance=np.diag(variances))

        self.assertTrue(np.array_equal(np.array(mean, dtype=float), multi_distr.mean))
        self.assertTrue(np.array_equal(np.diag(variances), multi_distr.covariance))
