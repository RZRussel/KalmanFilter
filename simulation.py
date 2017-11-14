from core.transformers import CSVMultiChannelReader, SonarAngleToPointsTransformer, WheelToRobotTransformer, \
    MultiChannelSyncTransformer, DifferentialDriveTransformer
from core.robots import BaseRobot
from core.distribution import GaussDistribution
from core.parsers import *
import matplotlib.pyplot as plt


class BaseSimulation:
    camera_log_path = 'resources/log_camera_2.csv'
    robot_log_path = 'resources/log_robot_2.csv'

    wheel_radius = 2.7
    wheel_base_half = 7.5
    sonar_zero_distance = 13.8

    init_x = 0.0
    init_y = 0.0
    init_angle = 0.0

    x_cam_noise = (0.0, 49.0)
    y_cam_noise = (0.0, 49.0)
    gyro_noise = (0.0, math.radians(16.0))

    sonar_normal_noise = (0.0, 4.0)
    sonar_invalid_noise = (0.0, 1e+6)

    def __init__(self, robot: BaseRobot, state_noise: GaussDistribution, label: str):
        self._robot = robot
        self._label = label
        self._state_noise = state_noise

    def show(self):
        raise NotImplementedError()


class RobotResultSimulation(BaseSimulation):
    def show(self):

        camera = CSVMultiChannelReader(path=self.camera_log_path, delimiter=';',
                                       parser=(lambda row, column, value: float(value)))
        robot = CSVMultiChannelReader(path=self.robot_log_path, delimiter=';', parser=parse_robot_log_column)

        camera_time_list = camera.channel_at_index(0)
        min_time = camera_time_list[0]
        max_time = camera_time_list[-1]

        merged = MultiChannelSyncTransformer([robot, camera], min_time, max_time)

        sonar_channel = merged.channel_at_index(1)
        gyro_channel = merged.channel_at_index(2)
        y_sensor_transformer = SonarAngleToPointsTransformer(channels=[sonar_channel, gyro_channel],
                                                             zero_distance=self.sonar_zero_distance)

        x_camera = merged.channel_at_index(5)
        y_camera = merged.channel_at_index(6)
        plt.plot(x_camera, y_camera, 'r-', label='camera')
        plt.plot(x_camera, y_sensor_transformer.channel_at_index(0), 'y-', label='y-sonar')

        left_channel = merged.channel_at_index(3)
        right_channel = merged.channel_at_index(4)

        velocities_transformer = WheelToRobotTransformer(channels=[left_channel, right_channel],
                                                         radius=self.wheel_radius, base_half=self.wheel_base_half)

        time_channel = merged.channel_at_index(0)
        v_channel = velocities_transformer.channel_at_index(0)
        w_channel = velocities_transformer.channel_at_index(1)

        kinematics_transformer = DifferentialDriveTransformer([time_channel, v_channel, w_channel],
                                                              init_x=self.init_x,
                                                              init_y=self.init_y,
                                                              init_angle=self.init_angle)
        x_kinemat = kinematics_transformer.channel_at_index(1)
        y_kinemat = kinematics_transformer.channel_at_index(2)

        plt.plot(x_kinemat, y_kinemat, 'b-', label='kinematic model')

        x_kalman = []
        y_kalman = []

        for i in range(0, len(merged.channel_at_index(0)) - 1):
            dt = merged.channel_at_index(0)[i + 1] - merged.channel_at_index(0)[i]

            v = v_channel[i]
            w = w_channel[i]

            self._robot.predict(v, w, dt, self._state_noise)

            x_cam = merged.channel_at_index(5)[i]
            y_cam = merged.channel_at_index(6)[i]

            sonar = merged.channel_at_index(1)[i]
            gyro = merged.channel_at_index(2)[i]

            if -math.radians(-30) < gyro < math.radians(30):
                sonar_noise = self.sonar_normal_noise
            else:
                sonar_noise = self.sonar_invalid_noise

            measurement_noise_mean = [self.x_cam_noise[0], self.y_cam_noise[0], sonar_noise[0], self.gyro_noise[0]]
            measurement_noise_vars = [self.x_cam_noise[1], self.y_cam_noise[1], sonar_noise[1], self.gyro_noise[1]]
            measurement_noise = GaussDistribution.create_independent(mean=measurement_noise_mean,
                                                                     variances=measurement_noise_vars)

            self._robot.update(x_cam, y_cam, sonar, gyro, measurement_noise)

            x_kalman.append(self._robot.state[0])
            y_kalman.append(self._robot.state[1])

        plt.plot(x_kalman, y_kalman, 'g-', label=self._label)

        plt.legend(loc='upper right')
        plt.show()
