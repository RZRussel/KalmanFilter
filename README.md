# Description
Project is designed to provide implementations of different filters based on probabilistic Bayes algorithm. 
Currently, there are implementations for:
..* Kalman filter;
..* Extended Kalman filter;
..* Unscented Kalman filter;
..* Particle filter;

Moreover the project includes example of the application of the algorithms to the differential drive robot case.

# Installation

Be sure that you have installed Python 3.x on your computer.

Than install project requirements by switching to the project directory and executing:

```pip install -r requirements.pip```

Feel free to test any of the filters on the data from our robot by executing one of the ```run_*.py``` files.
For example, to see Extended Kalman filter in work run:

```python run_ekf.py```

# Case Study

As was mentioned above for our case study we gathered data from observation of the differential drive robot
following non-linear trajectory. The robot was equiped with sonar and gyro sensors. The sonar was placed perpendicular
to the robot axes and the fact that trajectory was lying along the wall allowed to capture distance to the wall. Thus, 
combining gyro with sonar one could calculate y-position of the robot. However, sometimes angle of the trajectory's
slope becomes high enough to allow sonar's ray to intersect another objects instead wall. It was done intentionally 
to be more closer to real situations. Another source of the data we used is odometry (rotation speed of the left 
and right wheels). Finally, the camera was placed above the robot to capture the process and estimate robot's position.
To build robot's motion model standard differential drive kinematics 
[equations](https://chess.eecs.berkeley.edu/eecs149/documentation/differentialDrive.pdf) were used.

The data can be found in ```resources/``` directory of the project. For one trajectory
there are 2 files: ```resources/log_robot*.csv``` and ```resources/log_camera*.csv```. First file consists of
5 columns and contains timestamp, sonar distance, gyro angle, left and right wheel rotation speed per
measurement. Second file contains timestamp, x and y coordinates of the robot per measurement.
