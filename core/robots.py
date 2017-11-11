import numpy as np
import math
import sys
from core.filters import KalmanFilter
from core.distribution import MultidimensionalDistribution


class EKFRobot:
    class ExtendedKalmanFilter(KalmanFilter):
        def _predict_state(self, control: np.array):
            p_x = self._updated.mean[0]
            p_y = self._updated.mean[1]
            p_angle = self._updated.mean[2]

            x = p_x + control[0]*math.cos(p_angle + control[1])
            y = p_y + control[0]*math.sin(p_angle + control[1])
            angle = p_angle + control[1]

            return np.array([x, y, angle], dtype=float)

        def _calculate_measurement(self):
            x = self._predicted.mean[0]
            y = self._predicted.mean[1]
            angle = self._predicted.mean[2]

            x_cam = x
            y_cam = y

            if -math.pi/2.0 < angle < math.pi/2.0:
                sonar = y / math.cos(angle)
            else:
                sonar = 1e+4

            gyro = angle

            return np.array([x_cam, y_cam, sonar, gyro], dtype=float)

    def __init__(self, initial: MultidimensionalDistribution):
        self._kf = self.ExtendedKalmanFilter(initial)

    def predict(self, v: float, w: float, dt: float, noise: MultidimensionalDistribution):
        control = np.array([v*dt, w*dt], dtype=float)
        state_matrix = self._calculate_state_jacobian(control)
        self._kf.update_state_matrix(state_matrix)
        self._kf.update_state_noise(noise)
        self._kf.predict(control)

    def update(self, x_cam: float, y_cam: float, sonar: float, gyro: float, noise: MultidimensionalDistribution):
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

        if -math.pi/2.0 < angle < math.pi:
            j[2][1] = 1 / math.cos(angle)
            j[2][2] = y * math.sin(angle) / (math.cos(angle) * math.cos(angle))
        else:
            j[2][1] = 1e+4
            j[2][2] = 1e+4

        j[3][2] = 1.0

        return j
