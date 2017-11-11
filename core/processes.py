import math


class DifferentialDriveProcess:
    def __init__(self, init_x: float, init_y: float, init_angle: float):
        self._x = init_x
        self._y = init_y
        self._angle = init_angle

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def angle(self) -> float:
        return self._angle

    def update(self, v: float, w: float, dt: float):
        self._x = self._x + v * dt * math.cos(self._angle + w * dt)
        self._y = self._y + v * dt * math.sin(self._angle + w * dt)
        self._angle = self._angle + w * dt
