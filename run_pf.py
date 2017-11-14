import math
from core.distribution import GaussDistribution
from core.robots import PFRobot
from simulation import RobotResultSimulation

x_noise = (0.0, 100.0)
y_noise = (0.0, 100.0)
angle_noise = (0.0, math.radians(25.0))

initial_mean = [x_noise[0], y_noise[0], angle_noise[0]]
initial_var = [x_noise[1], y_noise[1], angle_noise[1]]
initial = GaussDistribution.create_independent(mean=initial_mean, variances=initial_var)
robot = PFRobot(initial, sample_size=1000, resampling_threshold=500)

state_noise = GaussDistribution.create_independent(mean=[0.0, 0.0], variances=[25.0, math.radians(25.0)])

simulation = RobotResultSimulation(robot, state_noise, 'pf')
simulation.show()
