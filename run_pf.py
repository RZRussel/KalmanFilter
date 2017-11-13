from core.transformers import CSVMultiChannelReader, SonarAngleToPointsTransformer, WheelToRobotTransformer, \
    MultiChannelSyncTransformer, DifferentialDriveTransformer
from core.robots import PFRobot
from core.distribution import GaussDistribution
from core.parsers import *
import matplotlib.pyplot as plt

camera_log_path = 'resources/log_camera_2.csv'
robot_log_path = 'resources/log_robot_2.csv'
compass_log_path = 'resources/data_phone_good_2.csv'

wheel_radius = 2.7
wheel_base_half = 7.5
sonar_zero_distance = 13.8

init_x = 0.0
init_y = 0.0
init_angle = 0.0

camera = CSVMultiChannelReader(path=camera_log_path, delimiter=';', parser=(lambda row, column, value: float(value)))
robot = CSVMultiChannelReader(path=robot_log_path, delimiter=';', parser=parse_robot_log_column)
compass = CSVMultiChannelReader(path=compass_log_path, delimiter=';', parser=parse_phone_log_column)

camera_time_list = camera.channel_at_index(0)
min_time = camera_time_list[0]
max_time = camera_time_list[-1]

merged = MultiChannelSyncTransformer([robot, camera, compass], min_time, max_time)

robot_time_list = robot.channel_at_index(0)

sonar_channel = merged.channel_at_index(1)
gyro_channel = merged.channel_at_index(2)
y_sensor_transformer = SonarAngleToPointsTransformer(channels=[sonar_channel, gyro_channel],
                                                     zero_distance=sonar_zero_distance)

axes = plt.gca()
axes.set_xlim([0, 150])
axes.set_ylim([-20, 50])
x_camera = merged.channel_at_index(5)
y_camera = merged.channel_at_index(6)
plt.plot(x_camera, y_camera, 'r.')
plt.plot(x_camera, y_sensor_transformer.channel_at_index(0), 'g.')

left_channel = merged.channel_at_index(3)
right_channel = merged.channel_at_index(4)

velocities_transformer = WheelToRobotTransformer(channels=[left_channel, right_channel],
                                                 radius=wheel_radius, base_half=wheel_base_half)

time_channel = merged.channel_at_index(0)
v_channel = velocities_transformer.channel_at_index(0)
w_channel = velocities_transformer.channel_at_index(1)

kinematics_transformer = DifferentialDriveTransformer([time_channel, v_channel, w_channel],
                                                      init_x=init_x, init_y=init_y, init_angle=init_angle)
x_kinemat = kinematics_transformer.channel_at_index(1)
y_kinemat = kinematics_transformer.channel_at_index(2)

plt.plot(x_kinemat, y_kinemat, 'b.')

x_kalman = []
y_kalman = []

x_noise = (0.0, 100.0)
y_noise = (0.0, 100.0)
angle_noise = (0.0, math.radians(25.0))
initial = GaussDistribution.create_independent(mean=[x_noise[0], y_noise[0], angle_noise[0]],
                                                variances=[x_noise[1], y_noise[1], angle_noise[1]])

x_cam_noise = (0.0, 25.0)
y_cam_noise = (0.0, 25.0)
gyro_noise = (0.0, math.radians(16.0))

pf_robot = PFRobot(initial=initial, sample_size=1000)

state_noise = GaussDistribution.create_independent(mean=[0, 0], variances=[25.0, math.radians(25.0)])

for i in range(0, len(merged.channel_at_index(0)) - 1):
    dt = merged.channel_at_index(0)[i+1] - merged.channel_at_index(0)[i]

    v = v_channel[i]
    w = w_channel[i]

    pf_robot.predict(v, w, dt, state_noise)

    x_cam = merged.channel_at_index(5)[i]
    y_cam = merged.channel_at_index(6)[i]

    sonar = merged.channel_at_index(1)[i]
    gyro = merged.channel_at_index(2)[i]

    if -math.radians(-30) < gyro < math.radians(30):
        sonar_noise = (0.0, 4.0)
    else:
        sonar_noise = (0.0, 1e+6)

    measurement_noise_mean = [x_cam_noise[0], y_cam_noise[0], sonar_noise[0], gyro_noise[0]]
    measurement_noise_vars = [x_cam_noise[1], y_cam_noise[1], sonar_noise[1], gyro_noise[1]]
    measurement_noise = GaussDistribution.create_independent(mean=measurement_noise_mean,
                                                             variances=measurement_noise_vars)

    pf_robot.update(x_cam, y_cam, sonar, gyro, measurement_noise)

    x_kalman.append(pf_robot.state[0])
    y_kalman.append(pf_robot.state[1])

plt.plot(x_kalman, y_kalman, 'c.')

plt.show()
