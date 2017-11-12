from unittest import TestCase
import numpy as np
from core.filters import KalmanFilter
from core.distribution import GaussDistribution


class TestKalmanFilter(TestCase):
    def test_initialization(self):
        init_distr = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([0.3, 0.2]))
        state_matrix = np.array([[1, 0], [0, 1]], dtype=float)
        control_matrix = np.array([[2, 0], [0, 2]], dtype=float)
        state_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([0.3, 0.2]))
        measurement_matrix = np.array([[1, 0], [0, 1]], dtype=float)
        measurement_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([1.0, 1.0]))

        kalman = KalmanFilter(init_distr, state_matrix,
                              control_matrix,
                              state_noise,
                              measurement_matrix,
                              measurement_noise)

        self.assertTrue(np.array_equal(kalman.predicted().mean, init_distr.mean))
        self.assertTrue(np.array_equal(kalman.predicted().covariance, init_distr.covariance))
        self.assertTrue(np.array_equal(kalman.updated().mean, init_distr.mean))
        self.assertTrue(np.array_equal(kalman.updated().covariance, init_distr.covariance))
        self.assertTrue(np.array_equal(kalman._state_matrix, state_matrix))
        self.assertTrue(np.array_equal(kalman._control_matrix, control_matrix))
        self.assertTrue(np.array_equal(kalman._state_noise.mean, state_noise.mean))
        self.assertTrue(np.array_equal(kalman._state_noise.covariance, state_noise.covariance))
        self.assertTrue(np.array_equal(kalman._measurement_matrix, measurement_matrix))
        self.assertTrue(np.array_equal(kalman._measurement_noise.mean, measurement_noise.mean))
        self.assertTrue(np.array_equal(kalman._measurement_noise.covariance, measurement_noise.covariance))

    def test_update(self):
        init_distr = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([0.3, 0.2]))
        state_matrix = np.array([[1, 0], [0, 1]], dtype=float)
        control_matrix = np.array([[2, 0], [0, 2]], dtype=float)
        state_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([0.3, 0.2]))
        measurement_matrix = np.array([[1, 0], [0, 1]], dtype=float)
        measurement_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([1.0, 1.0]))

        kalman = KalmanFilter(init_distr, state_matrix,
                              control_matrix,
                              state_noise,
                              measurement_matrix,
                              measurement_noise)

        updated_state_matrix = np.array([[2, 0], [0, 1]], dtype=float)
        kalman.update_state_matrix(updated_state_matrix)

        updated_control_matrix = np.array([[2, 3], [3, 3]], dtype=float)
        kalman.update_control_matrix(updated_control_matrix)

        updated_state_noise = GaussDistribution(mean=np.array([1, 0]), covariance=np.diag([0.2, 0.2]))
        kalman.update_state_noise(updated_state_noise)

        updated_measurement_matrix = np.array([[2, 2], [2, 2]], dtype=float)
        kalman.update_measurement_matrix(updated_measurement_matrix)

        updated_measurement_noise = GaussDistribution(mean=np.array([1, 1]), covariance=np.diag([2.0, 2.0]))
        kalman.update_measurement_noise(updated_measurement_noise)

        self.assertTrue(np.array_equal(kalman._state_matrix, updated_state_matrix))
        self.assertTrue(np.array_equal(kalman._control_matrix, updated_control_matrix))
        self.assertTrue(np.array_equal(kalman._state_noise.mean, updated_state_noise.mean))
        self.assertTrue(np.array_equal(kalman._state_noise.covariance, updated_state_noise.covariance))
        self.assertTrue(np.array_equal(kalman._measurement_matrix, updated_measurement_matrix))
        self.assertTrue(np.array_equal(kalman._measurement_noise.mean, updated_measurement_noise.mean))
        self.assertTrue(np.array_equal(kalman._measurement_noise.covariance, updated_measurement_noise.covariance))

    def test_priori(self):
        init_distr = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([0.1, 0.1]))
        state_matrix = np.array([[1, 0], [0, 1]], dtype=float)
        control_matrix = np.array([[2, 0], [0, 2]], dtype=float)
        state_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([0.3, 0.2]))
        measurement_matrix = np.array([[1, 0], [0, 1]], dtype=float)
        measurement_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([1.0, 1.0]))

        kalman = KalmanFilter(init_distr, state_matrix,
                              control_matrix,
                              state_noise,
                              measurement_matrix,
                              measurement_noise)

        kalman.predict(control=np.array([1, 1], dtype=float))
        kalman.update(measurements=np.array([2, 2]))

        kalman.predict(control=np.array([1, 1], dtype=float))
        kalman.update(measurements=np.array([4.1, 4.02]))

        self.assertTrue(np.array_equal(kalman.predicted().mean, np.array([4.0, 4.0], dtype=float)))

    def test_posteriori(self):
        init_distr = GaussDistribution(mean=np.array([1, -1]), covariance=np.diag([1, 1]))
        state_matrix = np.array([[1, -0.5], [0.5, 1]], dtype=float)
        control_matrix = np.array([[0, 0], [0, 0]], dtype=float)
        state_noise = GaussDistribution(mean=np.array([0, 0]), covariance=np.diag([1, 1]))
        measurement_matrix = np.array([[1, 2]], dtype=float)
        measurement_noise = GaussDistribution(mean=np.array([[0]]), covariance=np.diag([1.0]))

        kalman = KalmanFilter(init_distr, state_matrix,
                              control_matrix,
                              state_noise,
                              measurement_matrix,
                              measurement_noise)

        measurements = [-2.0, 4.5, 1.75, 7.625]

        final = [[1.04081633, -1.41836735], [3.53106567, 0.25088478], [0.94444962, 0.66461669],
                 [2.68852441, 2.25424821]]
        for i in range(0, len(measurements)):
            kalman.predict(control=np.array([0, 0]))
            kalman.update(measurements=np.array([measurements[i]]))

            compare = final[i]
            self.assertAlmostEqual(first=compare[0], second=kalman.updated().mean[0], delta=1e-2)
            self.assertAlmostEqual(first=compare[1], second=kalman.updated().mean[1], delta=1e-2)
