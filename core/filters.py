import numpy as np
import math
from core.distribution import BaseDistribution, GaussDistribution, NaiveSampleDistribution


class BayesFilter:
    def predicted(self) -> BaseDistribution:
        raise NotImplementedError()

    def updated(self) -> BaseDistribution:
        raise NotImplementedError()

    def predict(self, control: np.array):
        raise NotImplementedError()

    def update(self, measurements: np.array):
        raise NotImplementedError()


class KalmanFilter:
    def __init__(self, initial: GaussDistribution,
                 state_matrix: np.array = None,
                 control_matrix: np.array = None,
                 state_noise: GaussDistribution = None,
                 measurement_matrix: np.array = None,
                 measurement_noise: GaussDistribution = None):

        self._predicted = initial
        self._updated = initial
        self._state_matrix = state_matrix
        self._control_matrix = control_matrix
        self._state_noise = state_noise
        self._measurement_matrix = measurement_matrix
        self._measurement_noise = measurement_noise

    def update_state_matrix(self, state_matrix: np.array):
        self._state_matrix = state_matrix

    def update_control_matrix(self, control_matrix: np.array):
        self._control_matrix = control_matrix

    def update_state_noise(self, state_noise: GaussDistribution):
        self._state_noise = state_noise

    def update_measurement_matrix(self, measurement_matrix: np.array):
        self._measurement_matrix = measurement_matrix

    def update_measurement_noise(self, measurement_noise: GaussDistribution):
        self._measurement_noise = measurement_noise

    def predict(self, control: np.array):
        priori_mean = self._predict_state(control)

        op_matrix = self._state_matrix.dot(self._updated.covariance).dot(self._state_matrix.transpose())
        priori_covariance = op_matrix + self._state_noise.covariance

        self._predicted = GaussDistribution(mean=priori_mean, covariance=priori_covariance)

    def _predict_state(self, control: np.array) -> np.array:
        return self._state_matrix.dot(self._updated.mean) + self._control_matrix.dot(control)

    def update(self, measurements: np.array):
        op_matrix = self._measurement_matrix.dot(self._predicted.covariance).dot(self._measurement_matrix.transpose())
        op_matrix = np.linalg.inv(op_matrix + self._measurement_noise.covariance)
        kalman_gain = self._predicted.covariance.dot(self._measurement_matrix.transpose()).dot(op_matrix)

        op_matrix = measurements - self._calculate_measurement()
        op_matrix = kalman_gain.dot(op_matrix)
        posterior_mean = self._predicted.mean + op_matrix

        op_matrix = kalman_gain.dot(self._measurement_matrix)
        op_matrix = (np.eye(N=op_matrix.shape[0], M=op_matrix.shape[1]) - op_matrix)
        posterior_covariance = op_matrix.dot(self._predicted.covariance)

        self._updated = GaussDistribution(mean=posterior_mean, covariance=posterior_covariance)

    def _calculate_measurement(self) -> np.array:
        return self._measurement_matrix.dot(self._predicted.mean)

    def predicted(self) -> GaussDistribution:
        return self._predicted

    def updated(self) -> GaussDistribution:
        return self._updated

UKF_COV_FIX = 1e-6


class BaseUnscentedKalmanFilter(BayesFilter):
    def __init__(self, initial: GaussDistribution,
                 state_noise: GaussDistribution = None,
                 measurement_noise: GaussDistribution = None,
                 alpha: float = 1.0,
                 beta: float = 0.0,
                 kappa: float = 0.0):
        self._predicted = initial
        self._updated = initial
        self._state_noise = state_noise
        self._measurement_noise = measurement_noise
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._n = 2*len(initial.mean)
        self._mean_weights = self._sample_mean_weights()
        self._cov_weights = self._sample_cov_weights()

    def update_state_noise(self, noise: GaussDistribution):
        self._state_noise = noise

    def update_measurement_noise(self, noise: GaussDistribution):
        self._measurement_noise = noise

    def predict(self, control: np.array):
        samples = self._sample_points(self._updated, self._state_noise)

        eval_samples = np.zeros((samples.shape[0], len(self._updated.mean)))
        for i in range(0, samples.shape[0]):
            eval_samples[i] = self._eval_state_func(control, samples[i])

        mean = np.zeros(self._updated.mean.shape)
        for i in range(0, 2 * self._n + 1):
            mean += self._mean_weights[i]*eval_samples[i]

        cov = np.zeros(self._updated.covariance.shape)
        for i in range(0, 2 * self._n + 1):
            op_vector = eval_samples[i] - mean
            op_vector = op_vector.reshape((1, len(op_vector)))
            cov += self._cov_weights[i] * op_vector.transpose().dot(op_vector)

        cov = self.fix_covariance(cov)

        self._predicted = GaussDistribution(mean=mean, covariance=cov)

    def update(self, measurements: np.array):
        samples = self._sample_points(self._predicted, self._measurement_noise)

        x_len = len(self._updated.mean)
        z_len = len(measurements)

        eval_samples = np.zeros((samples.shape[0], z_len))
        for i in range(0, samples.shape[0]):
            eval_samples[i] = self._eval_measurement_func(samples[i])

        mean = np.zeros((z_len,))
        for i in range(0, 2 * self._n + 1):
            mean += self._mean_weights[i] * eval_samples[i]

        cov = np.zeros((z_len, z_len))
        for i in range(0, 2 * self._n + 1):
            op_vector = eval_samples[i] - mean
            op_vector = op_vector.reshape((1, len(op_vector)))
            cov += self._cov_weights[i] * op_vector.transpose().dot(op_vector)

        cov = self.fix_covariance(cov)

        state_samples = np.zeros((samples.shape[0], x_len))
        for i in range(0, 2 * self._n + 1):
            state_samples[i] = samples[i][:len(self._updated.mean)]

        cross_cov = np.zeros((x_len, z_len))
        for i in range(0, 2 * self._n + 1):
            op_sample = state_samples[i] - self._predicted.mean
            op_sample = op_sample.reshape((1, len(op_sample)))
            op_eval = eval_samples[i] - mean
            op_eval = op_eval.reshape((1, len(op_eval)))
            cross_cov += self._cov_weights[i] * op_sample.transpose().dot(op_eval)

        kalman_gain = cross_cov.dot(np.linalg.inv(cov))

        mean_update = self._predicted.mean + kalman_gain.dot(measurements - mean)
        cov_update = self._predicted.covariance - kalman_gain.dot(cov).dot(kalman_gain.transpose())

        cov_update = self.fix_covariance(cov_update)

        self._updated = GaussDistribution(mean=mean_update, covariance=cov_update)

    def predicted(self) -> GaussDistribution:
        return self._predicted

    def updated(self) -> GaussDistribution:
        return self._updated

    def _sample_points(self, distr: GaussDistribution, noise: GaussDistribution) -> np.array:
        mean_aug = np.hstack((distr.mean, noise.mean))

        cov_top = np.hstack((distr.covariance, np.zeros((distr.covariance.shape[0], noise.covariance.shape[1]))))
        cov_bottom = np.hstack((np.zeros((noise.covariance.shape[0], distr.covariance.shape[1])), noise.covariance))
        cov_aug = np.vstack((cov_top, cov_bottom))

        samples = np.zeros((2*self._n + 1, len(mean_aug)))
        samples[0] = mean_aug

        op_cholesky = np.linalg.cholesky(cov_aug)
        op_matrix = math.sqrt(self._n + self._calculate_lambda())*op_cholesky.transpose()
        for i in range(1, self._n + 1):
            samples[i] = mean_aug + op_matrix[i - 1]
            samples[self._n + i] = mean_aug - op_matrix[i - 1]

        return samples

    def _sample_mean_weights(self) -> np.array:
        weights = np.zeros((2 * self._n + 1,))

        lmda = self._calculate_lambda()
        weights[0] = lmda / (self._n + lmda)

        for i in range(1, 2 * self._n + 1):
            weights[i] = 1 / (2 * (self._n + lmda))

        return weights

    def _sample_cov_weights(self) -> np.array:
        weights = np.zeros((2 * self._n + 1,))

        lmda = self._calculate_lambda()
        weights[0] = lmda / (self._n + lmda) + 1 - self._alpha*self._alpha + self._beta

        for i in range(1, 2 * self._n + 1):
            weights[i] = 1 / (2 * (self._n + lmda))

        return weights

    def _calculate_lambda(self) -> float:
        return self._alpha*self._alpha*(self._n + self._kappa) - self._n

    def _eval_state_func(self, control: np.array, point: np.array) -> np.array:
        raise NotImplementedError()

    def _eval_measurement_func(self, point: np.array) -> np.array:
        raise NotImplementedError()

    @staticmethod
    def fix_covariance(cov: np.array) -> np.array:
        new_cov = 0.5*cov + 0.5*cov.transpose()
        new_cov = new_cov + UKF_COV_FIX*np.eye(cov.shape[0], cov.shape[1])
        return new_cov


class BaseParticleFilter(BayesFilter):
    def __init__(self, sample_size: int, initial: BaseDistribution):
        self._sample_size = sample_size

        samples = np.array((sample_size, len(initial.mean)))
        for i in range(0, sample_size):
            samples[i] = initial.sample()

        sample_distr = NaiveSampleDistribution(samples)

        self._predicted = sample_distr
        self._updated = sample_distr

    def predict(self, control: np.array):
        samples = self._sample_state(control)
        self._predicted = NaiveSampleDistribution(samples)

    def update(self, measurements: np.array):
        weights = self._calculate_weights(measurements)
        samples = self._resample(weights)

        self._updated = NaiveSampleDistribution(samples)

    def predicted(self) -> BaseDistribution:
        return self._predicted

    def updated(self) -> BaseDistribution:
        return self._updated

    def _sample_state(self, control: np.array) -> np.array:
        raise NotImplementedError()

    def _calculate_weights(self, measurements: np.array) -> np.array:
        raise NotImplementedError()

    def _resample(self, weights: np.array) -> np.array:
        norm = sum(weights)

        old_samples = self._predicted.samples
        new_samples = np.zeros(old_samples.shape)
        for i in range(0, len(weights)):
            new_samples[i] = weights[i]*old_samples[i]/norm

        return new_samples