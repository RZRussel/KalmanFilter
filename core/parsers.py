import math
from typing import Any


def parse_robot_log_column(row_index: int, column_index: int, value: Any):
    if row_index == 0:
        return None

    if column_index == 2:
        return math.radians(float(value))

    if column_index > 2:
        return float(value)*math.pi/180.0

    return float(value)


def parse_phone_log_column(row_index: int, column_index: int, value: Any):
    if row_index == 0:
        return None

    return float(value)