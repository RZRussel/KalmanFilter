import numpy as np
import scipy
from scipy import stats
import math
from core.filters import KalmanFilter, BaseUnscentedKalmanFilter, BaseParticleFilter
from core.distribution import GaussDistribution, BaseDistribution

K_SONAR_BIG = 1e+4


def calculate_state_func(control: np.array, state: np.array) -> np.array:
    x = state[0] + control[0] * math.cos(state[2] + control[1])
    y = state[1] + control[0] * math.sin(state[2] + control[1])
    angle = state[2] + control[1]

    return np.array([x, y, angle], dtype=float)


def calculate_measurement_func(state: np.array) -> np.array:
    x_cam = state[0]
    y_cam = state[1]

    if -math.pi / 2.0 < state[2] < math.pi / 2.0:
        sonar = state[1] / math.cos(state[2])
    else:
        sonar = K_SONAR_BIG

    gyro = state[2]

    return np.array([x_cam, y_cam, sonar, gyro], dtype=float)


class BaseRobot:
    def predict(self, v: float, w: float, dt: float, noise: GaussDistribution):
        raise NotImplementedError()

    def update(self, x_cam: float, y_cam: float, sonar: float, gyro: float, noise: GaussDistribution):
        raise NotImplementedError()

    @property
    def state(self) -> np.array:
        raise NotImplementedError()


class EKFRobot(BaseRobot):
    class ExtendedKalmanFilter(KalmanFilter):
        def _predict_state(self, control: np.array):
            return calculate_state_func(control, self._updated.mean)

        def _calculate_measurement(self):
            return calculate_measurement_func(self._updated.mean)

    def __init__(self, initial: GaussDistribution):
        self._kf = self.ExtendedKalmanFilter(initial)

    def predict(self, v: float, w: float, dt: float, noise: GaussDistribution):
        control = np.array([v*dt, w*dt], dtype=float)
        state_matrix = self._calculate_state_jacobian(control)
        self._kf.update_state_matrix(state_matrix)
        self._kf.update_state_noise(noise)
        self._kf.predict(control)

    def update(self, x_cam: float, y_cam: float, sonar: float, gyro: float, noise: GaussDistribution):
        measurements = np.array([x_cam, y_cam, sonar, gyro], dtype=float)
        measurement_matrix = self._calculate_measurement_jacobian()
        self._kf.update_measurement_matrix(measurement_matrix)
        self._kf.update_measurement_noise(noise)
        self._kf.update(measurements)

    @property
    def state(self) -> np.array:
        return self._kf.updated().mean

    def _calculate_state_jacobian(self, control: np.array) -> np.array:
        p_angle = self._kf.updated().mean[2]

        j = np.eye(3, 3)
        j[0][2] = -control[0]*math.sin(p_angle + control[1])
        j[1][2] = control[0]*math.cos(p_angle + control[1])

        return j

    def _calculate_measurement_jacobian(self) -> np.array:
        y = self._kf.predicted().mean[1]
        angle = self._kf.predicted().mean[2]

        j = np.eye(4, 3)

        if -math.pi/2.0 < angle < math.pi/2.0:
            j[2][1] = 1 / math.cos(angle)
            j[2][2] = y * math.sin(angle) / (math.cos(angle) * math.cos(angle))
        else:
            j[2][1] = K_SONAR_BIG
            j[2][2] = K_SONAR_BIG

        j[3][2] = 1.0

        return j


class UKFRobot(BaseRobot):
    class UnscentedKalmanFilter(BaseUnscentedKalmanFilter):
        def _eval_state_func(self, control: np.array, point: np.array):
            return calculate_state_func(control, point)

        def _eval_measurement_func(self, point: np.array):
            return calculate_measurement_func(point)

    def __init__(self, initial: GaussDistribution, alpha=1e-3, kappa=0, beta=2):
        self._ukf = self.UnscentedKalmanFilter(initial=initial, alpha=alpha, kappa=kappa, beta=beta)

    def predict(self, v: float, w: float, dt: float, noise: GaussDistribution):
        control = np.array([v * dt, w * dt], dtype=float)
        self._ukf.update_state_noise(noise)
        self._ukf.predict(control)

    def update(self, x_cam: float, y_cam: float, sonar: float, gyro: float, noise: GaussDistribution):
        measurements = np.array([x_cam, y_cam, sonar, gyro], dtype=float)
        self._ukf.update_measurement_noise(noise)
        self._ukf.update(measurements)

    @property
    def state(self) -> np.array:
        return self._ukf.updated().mean


class PFRobot(BaseRobot):
    class ParticleFilter(BaseParticleFilter):

        def __init__(self, initial: BaseDistribution, sample_size: int):
            super().__init__(initial, sample_size)

            self._state_noise = None
            self._measurement_noise = None

        def update_state_noise(self, noise: BaseDistribution):
            self._state_noise = noise

        def update_measurement_noise(self, noise: BaseDistribution):
            self._measurement_noise = noise

        def _sample_state(self, control: np.array) -> np.array:
            prev_sample = self._updated.samples
            sample = np.zeros(prev_sample.shape)

            for i in range(0, prev_sample.shape[0]):
                sample_control = control + self._state_noise.sample()
                sample_control[1] %= 2*math.pi
                sample[i] = calculate_state_func(control, prev_sample[i])

            return sample

        def _calculate_weights(self, measurements: np.array) -> np.array:
            weights = np.full((self._sample_size,), 1.0)

            cov = self._measurement_noise.covariance
            for i in range(0, len(weights)):
                pred_measurement = calculate_measurement_func(self._predicted.samples[i])
                distr = scipy.stats.multivariate_normal(pred_measurement, cov)
                weights[i] = distr.pdf(measurements)

            weights += 1e-300

            return weights

    def __init__(self, initial: GaussDistribution, sample_size: int = 1000):
        self._pf = self.ParticleFilter(initial, sample_size)

    def predict(self, v: float, w: float, dt: float, noise: GaussDistribution):
        control = np.array([v*dt, w*dt], dtype=float)
        self._pf.update_state_noise(noise)
        self._pf.predict(control)

    def update(self, x_cam: float, y_cam: float, sonar: float, gyro: float, noise: GaussDistribution):
        measurements = np.array([x_cam, y_cam, sonar, gyro], dtype=float)
        self._pf.update_measurement_noise(noise)
        self._pf.update(measurements)

    @property
    def state(self) -> np.array:
        return self._pf.updated().mean
