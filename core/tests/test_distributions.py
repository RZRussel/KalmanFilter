import numpy as np
import math
from unittest import TestCase
from core.distribution import GaussDistribution, NaiveSampleDistribution


class TestGaussDistribution(TestCase):
    def test_initialization(self):
        mean = 10.0
        var = 0.5
        distr = GaussDistribution.create_1d(mean, var)

        self.assertAlmostEqual(first=mean, second=distr.mean[0], delta=1e-6)
        self.assertAlmostEqual(first=var, second=distr.covariance[0][0], delta=1e-6)

    def test_sample(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = GaussDistribution.create_independent(mean, variances)

        for i in range(0, 10):
            sample = multi_distr.sample()

            self.assertTrue(abs(mean[0] - sample[0]) < 6.0*math.sqrt(variances[0]))
            self.assertTrue(abs(mean[1] - sample[1]) < 6.0 * math.sqrt(variances[1]))


class TestMultidimensionalDistribution(TestCase):
    def test_distr_initialization(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = GaussDistribution.create_independent(mean, variances)

        self.assertTrue(np.array_equal(np.array(mean, dtype=float), multi_distr.mean))
        self.assertTrue(np.array_equal(np.diag(variances), multi_distr.covariance))

    def test_means_covariance_initialization(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = GaussDistribution(mean=np.array(mean, dtype=float), covariance=np.diag(variances))

        self.assertTrue(np.array_equal(np.array(mean, dtype=float), multi_distr.mean))
        self.assertTrue(np.array_equal(np.diag(variances), multi_distr.covariance))


class TestNaiveSampleDistribution(TestCase):
    def test_mean(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = GaussDistribution.create_independent(mean, variances)

        n = 100
        samples = np.zeros((n, 2))
        for i in range(0, n):
            samples[i] = multi_distr.sample()

        sample_distr = NaiveSampleDistribution(samples)
        sample_mean = sample_distr.mean

        for i in range(0, len(mean)):
            self.assertTrue(abs(mean[0] - sample_mean[0]) < math.sqrt(variances[0]))

    def test_covariance(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = GaussDistribution.create_independent(mean, variances)

        n = 100
        samples = np.zeros((n, 2))
        for i in range(0, n):
            samples[i] = multi_distr.sample()

        sample_distr = NaiveSampleDistribution(samples)
        sample_cov = sample_distr.covariance

        for i in range(0, len(mean)):
            self.assertTrue(abs(variances[i] - sample_cov[i][i]) < 1.0)

    def test_sample(self):
        mean = [10.0, 15.0]
        variances = [0.1, 0.5]

        multi_distr = GaussDistribution.create_independent(mean, variances)

        n = 100
        samples = np.zeros((n, 2))
        for i in range(0, n):
            samples[i] = multi_distr.sample()

        sample_distr = NaiveSampleDistribution(samples)

        for i in range(0, n):
            self.assertTrue(sample_distr.sample() in samples)