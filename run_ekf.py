from core.readers import CSVMultiChannelReader, SonarAngleToPointsTransformer, WheelToRobotTransformer, \
    MultiChannelSynchronizer, DifferentialDriveKinematics
from core.parsers import *
import matplotlib.pyplot as plt

camera_log_path = 'resources/log_camera.csv'
robot_log_path = 'resources/log_robot.csv'
compass_log_path = 'resources/data_phone_good_1.csv'

wheel_radius = 2.7
wheel_base_half = 7.5
sonar_zero_distance = 13.8

init_x = 0.0
init_y = 0.0
init_angle = 0.0

camera = CSVMultiChannelReader(path=camera_log_path, delimiter=';', parser=(lambda row, column, value: float(value)))
robot = CSVMultiChannelReader(path=robot_log_path, delimiter=';', parser=parse_robot_log_column)
compass = CSVMultiChannelReader(path=compass_log_path, delimiter=';', parser=parse_phone_log_column)

robot_time_list = robot.channel_at_index(0)
min_time = robot_time_list[0]
max_time = robot_time_list[-1]

merged = MultiChannelSynchronizer([robot, camera, compass], min_time, max_time)

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

kinematics_transformer = DifferentialDriveKinematics([time_channel, v_channel, w_channel],
                                                     init_x=init_x, init_y=init_y, init_angle=init_angle)
x_kinemat = kinematics_transformer.channel_at_index(1)
y_kinemat = kinematics_transformer.channel_at_index(2)

plt.plot(x_kinemat, y_kinemat, 'b.')

plt.show()
