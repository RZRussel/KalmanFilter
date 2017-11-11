import numpy as np
import math
from core.distribution import MultidimensionalDistribution


class BayesFilter:
    def predicted(self) -> MultidimensionalDistribution:
        raise NotImplementedError()

    def updated(self) -> MultidimensionalDistribution:
        raise NotImplementedError()

    def predict(self, control: np.array):
        raise NotImplementedError()

    def update(self, measurements: np.array):
        raise NotImplementedError()


class KalmanFilter:
    def __init__(self, initial: MultidimensionalDistribution,
                 state_matrix: np.array = None,
                 control_matrix: np.array = None,
                 state_noise: MultidimensionalDistribution = None,
                 measurement_matrix: np.array = None,
                 measurement_noise: MultidimensionalDistribution = None):

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

    def update_state_noise(self, state_noise: MultidimensionalDistribution):
        self._state_noise = state_noise

    def update_measurement_matrix(self, measurement_matrix: np.array):
        self._measurement_matrix = measurement_matrix

    def update_measurement_noise(self, measurement_noise: MultidimensionalDistribution):
        self._measurement_noise = measurement_noise

    def predict(self, control: np.array):
        priori_mean = self._predict_state(control)

        op_matrix = self._state_matrix.dot(self._updated.covariance).dot(self._state_matrix.transpose())
        priori_covariance = op_matrix + self._state_noise.covariance

        self._predicted = MultidimensionalDistribution(mean=priori_mean, covariance=priori_covariance)

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

        self._updated = MultidimensionalDistribution(mean=posterior_mean, covariance=posterior_covariance)

    def _calculate_measurement(self) -> np.array:
        return self._measurement_matrix.dot(self._predicted.mean)

    def predicted(self) -> MultidimensionalDistribution:
        return self._predicted

    def updated(self) -> MultidimensionalDistribution:
        return self._updated


class UnscentedKalmanFilter(BayesFilter):
    def __init__(self, initial: MultidimensionalDistribution,
                 state_noise: MultidimensionalDistribution = None,
                 measurement_noise: MultidimensionalDistribution = None,
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

    def predict(self, control: np.array):
        samples = self._sample_points(self._updated, self._state_noise)

        for i in range(0, samples.shape[0]):
            samples[i] = self._eval_state_func(control, samples[i])

        mean = np.zeros(self._updated.mean.shape)
        for i in range(0, 2*self._n + 1):
            mean += self._mean_weights[i]*samples[i]

        cov = np.zeros(self._updated.covariance.shape)
        for i in range(0, 2*self._n + 1):
            op_vector = samples[i] - mean
            op_vector = op_vector.reshape((1, len(op_vector)))
            cov += self._cov_weights*op_vector.transpose().dot(op_vector)

        self._predicted = MultidimensionalDistribution(mean=mean, covariance=cov)

    def update(self, measurements: np.array):
        samples = self._sample_points(self._predicted, self._measurement_noise)

        eval_samples = np.zeros(samples.shape)
        for i in range(0, samples.shape[0]):
            eval_samples[i] = self._eval_measurement_func(samples[i])

        mean = np.zeros(self._updated.mean.shape)
        for i in range(0, 2 * self._n + 1):
            mean += self._mean_weights[i] * samples[i]

        cov = np.zeros(self._updated.covariance.shape)
        for i in range(0, 2 * self._n + 1):
            op_vector = eval_samples[i] - mean
            op_vector = op_vector.reshape((1, len(op_vector)))
            cov += self._cov_weights * op_vector.transpose().dot(op_vector)

        cross_cov = np.zeros(self._updated.covariance.shape)
        for i in range(0, 2 * self._n + 1):
            op_sample = samples[i] - self._predicted.mean
            op_sample = op_sample.reshape((1, len(op_sample)))
            op_eval = eval_samples[i] - mean
            op_eval = op_eval.reshape((1, len(op_eval)))
            cross_cov += self._cov_weights * op_sample.transpose().dot(op_eval)

        kalman_gain = cross_cov.dot(np.linalg.inv(cov))

        mean_update = self._predicted.mean + kalman_gain.dot(measurements - mean)
        cov_update = self._predicted.covariance - kalman_gain.dot(cov).dot(kalman_gain.transpose())

        self._updated = MultidimensionalDistribution(mean=mean_update, covariance=cov_update)

    def predicted(self) -> MultidimensionalDistribution:
        return self._predicted

    def updated(self) -> MultidimensionalDistribution:
        return self._updated

    def _sample_points(self, distr: MultidimensionalDistribution, noise: MultidimensionalDistribution) -> np.array:
        mean_aug = np.hstack((distr.mean, noise.mean))

        cov_top = np.hstack((distr.covariance, np.zeros((distr.covariance.shape[0], noise.covariance.shape[1]))))
        cov_bottom = np.hstack((np.zeros((noise.covariance.shape[0], distr.covariance.shape[1])), noise.covariance))
        cov_aug = np.vstack((cov_top, cov_bottom))

        samples = np.zeros((2*self._n + 1, len(distr.mean)))
        samples[0] = mean_aug

        op_matrix = math.sqrt(self._n + self._calculate_lambda())*np.linalg.cholesky(cov_aug).transpose()
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