import numpy as np
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
                 state_matrix: np.array,
                 control_matrix: np.array,
                 state_noise: MultidimensionalDistribution,
                 measurement_matrix: np.array,
                 measurement_noise: MultidimensionalDistribution):

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
        priori_mean = self._state_matrix.dot(self._updated.mean) + self._control_matrix.dot(control)

        op_matrix = self._state_matrix.dot(self._updated.covariance).dot(self._state_matrix.transpose())
        priori_covariance = op_matrix + self._state_noise.covariance

        self._predicted = MultidimensionalDistribution(mean=priori_mean, covariance=priori_covariance)

    def update(self, measurements: np.array):
        op_matrix = self._measurement_matrix.dot(self._predicted.covariance).dot(self._measurement_matrix.transpose())
        op_matrix = np.linalg.inv(op_matrix + self._measurement_noise.covariance)
        kalman_gain = self._predicted.covariance.dot(self._measurement_matrix.transpose()).dot(op_matrix)

        op_matrix = self._measurement_matrix.dot(self._predicted.mean)
        op_matrix = measurements - op_matrix
        op_matrix = kalman_gain.dot(op_matrix)
        posterior_mean = self._predicted.mean + op_matrix

        op_matrix = kalman_gain.dot(self._measurement_matrix)
        op_matrix = (np.eye(N=op_matrix.shape[0], M=op_matrix.shape[1]) - op_matrix)
        posterior_covariance = op_matrix.dot(self._predicted.covariance)

        self._updated = MultidimensionalDistribution(mean=posterior_mean, covariance=posterior_covariance)

    def predicted(self) -> MultidimensionalDistribution:
        return self._predicted

    def updated(self) -> MultidimensionalDistribution:
        return self._updated
