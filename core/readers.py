import csv
import math
from typing import List, Any, Iterator


class BaseMultiChannel:
    @property
    def number_of_channels(self) -> int:
        raise NotImplementedError()

    def channel_at_index(self, index: int) -> List[Any]:
        raise NotImplementedError()


class DefaultMultiChannel(BaseMultiChannel):
    def __init__(self, channels: List):
        self._channels = channels

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    def channel_at_index(self, index: int) -> List[Any]:
        return self._channels[index]


class CSVMultiChannelReader(BaseMultiChannel):
    def __init__(self, path, delimiter=' ', parser=None):
        with open(path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=delimiter)
            self._parse_channels(csv_reader, parser)

    def _parse_channels(self, csv_reader: Iterator, parser: Any):
        row_index = 0
        channels = None
        for row in csv_reader:
            if channels is None:
                channels = [list() for _ in range(0, len(row))]

            for column_index in range(0, len(row)):
                channel = channels[column_index]

                if parser is not None:
                    result = parser(row_index, column_index, row[column_index])
                    if result is not None:
                        channel.append(result)
                else:
                    channel.append(row[column_index])

            row_index += 1

        self._channels = channels

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    def channel_at_index(self, index: int) -> List[Any]:
        return self._channels[index]


# noinspection PyAbstractClass
class BaseMultiChannelTransformer(BaseMultiChannel):
    def __init__(self, channels: List, row_replacement=None, **kwargs):
        self._kwargs = kwargs
        self._process(channels, row_replacement)

    def _process(self, channels: List, row_replacement=None):
        raise NotImplementedError()


class SonarAngleToPointsTransformer(BaseMultiChannelTransformer):
    ZERO_DISTANCE = "zero_distance"

    def _process(self, channels: List, row_replacement=None):
        if self.ZERO_DISTANCE not in self._kwargs:
            raise ValueError("Sonar initial distance was not provided")

        zero_distance = self._kwargs[SonarAngleToPointsTransformer.ZERO_DISTANCE]

        channel = []

        sonar_list = channels[0]
        angle_list = channels[1]
        for row_index in range(0, len(sonar_list)):
            if row_replacement is not None:
                y = row_replacement(sonar_list[row_index], angle_list[row_index])
            else:
                y = sonar_list[row_index]*math.cos(angle_list[row_index])

            channel.append(y - zero_distance)

        self._channel = channel

    @property
    def number_of_channels(self) -> int:
        return 1

    def channel_at_index(self, index: int) -> List[Any]:
        if index != 0:
            raise ValueError("Only one channel exists")

        return self._channel


class WheelToRobotTransformer(BaseMultiChannelTransformer):
    WHEEL_RADIUS = "radius"
    WHEEL_BASE_HALF = "base_half"

    def _process(self, channels: List, row_replacement=None):
        if self.WHEEL_RADIUS not in self._kwargs:
            raise ValueError("Wheel radius was not provided")

        if self.WHEEL_BASE_HALF not in self._kwargs:
            raise ValueError("Wheel base half was not provided")

        radius = self._kwargs[WheelToRobotTransformer.WHEEL_RADIUS]
        base_half = self._kwargs[WheelToRobotTransformer.WHEEL_BASE_HALF]

        v_list = []
        w_list = []

        left_list = channels[1]
        right_list = channels[0]

        for row_index in range(0, len(left_list)):
            if row_replacement is not None:
                v, w = row_replacement(left_list[row_index], right_list[row_index])
            else:
                v = (right_list[row_index] + left_list[row_index]) * radius / 2.0
                w = (right_list[row_index] - left_list[row_index]) * radius / (2.0 * base_half)

            v_list.append(v)
            w_list.append(w)

        self._channels = [v_list, w_list]

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    def channel_at_index(self, index: int) -> List[Any]:
        return self._channels[index]


class DifferentialDriveKinematics(BaseMultiChannelTransformer):
    INIT_X = "init_x"
    INIT_Y = "init_y"
    INIT_ANGLE = "init_angle"

    def _process(self, channels: List, row_replacement=None):
        if self.INIT_X not in self._kwargs:
            raise ValueError("Initial robot x position was not provided")

        if self.INIT_Y not in self._kwargs:
            raise ValueError("Initial robot y position was not provided")

        if self.INIT_ANGLE not in self._kwargs:
            raise ValueError("Initial robot angle was not provided")

        x = self._kwargs[self.INIT_X]
        y = self._kwargs[self.INIT_Y]
        angle = self._kwargs[self.INIT_ANGLE]

        t_list = []
        x_list = []
        y_list = []
        angle_list = []

        for i in range(0, len(channels[0])):
            t = channels[0][i]
            t_list.append(t)
            x_list.append(x)
            y_list.append(y)
            angle_list.append(angle)

            if i < len(channels[0]) - 1:
                v = channels[1][i]
                w = channels[2][i]
                t_next = channels[0][i + 1]

                dt = t_next - t
                x = x + v * dt * math.cos(angle + w * dt)
                y = y + v * dt * math.sin(angle + w * dt)
                angle = angle + w * dt

        self._channels = [t_list, x_list, y_list, angle_list]

    @property
    def number_of_channels(self) -> int:
        return len(self._channels)

    def channel_at_index(self, index: int) -> List[Any]:
        return self._channels[index]


class MultiChannelSynchronizer(BaseMultiChannel):
    def __init__(self, multi_channels: List[BaseMultiChannel], min_time: float, max_time: float):
        self._synchronize(multi_channels, min_time, max_time)

    def _synchronize(self, multi_channels: List[BaseMultiChannel], min_time: float, max_time: float):
        chan_count = sum(multi_channel.number_of_channels - 1 for multi_channel in multi_channels)
        channels = [list() for _ in range(0, chan_count + 1)]

        bounded_multi_channels = [self._filter_by_time(multi_channel, min_time, max_time)
                                  for multi_channel in multi_channels]

        min_len = None

        for i in range(0, len(bounded_multi_channels)):
            multi_channel = bounded_multi_channels[i]
            if min_len is None:
                min_len = len(multi_channel.channel_at_index(0))
            elif min_len > len(multi_channel.channel_at_index(0)):
                min_len = len(multi_channel.channel_at_index(0))

        time_step = (max_time - min_time)/(min_len - 1)

        reduced_multi_channels = [self._reduce_len(multi_channel, min_len) for multi_channel in bounded_multi_channels]

        for i in range(0, min_len):
            channels[0].append(i*time_step)

            offset = 1
            for multi_channel in reduced_multi_channels:
                for j in range(0, multi_channel.number_of_channels):
                    chan = multi_channel.channel_at_index(j)
                    channels[offset + j].append(chan[i])

                offset += multi_channel.number_of_channels

        self._channels = channels

    @staticmethod
    def _filter_by_time(multi_channel: BaseMultiChannel, min_time: float, max_time: float) -> BaseMultiChannel:
        time_channel = multi_channel.channel_at_index(0)

        channels = [list() for _ in range(0, multi_channel.number_of_channels)]

        for i in range(0, len(time_channel)):
            if min_time <= time_channel[i] <= max_time:
                for j in range(0, len(channels)):
                    chan = multi_channel.channel_at_index(j)
                    channels[j].append(chan[i])

        return DefaultMultiChannel(channels)

    @staticmethod
    def _reduce_len(multi_channel: BaseMultiChannel, final_len: int) -> BaseMultiChannel:
        curr_len = len(multi_channel.channel_at_index(0))
        step = curr_len / final_len

        channels = [list() for _ in range(0, multi_channel.number_of_channels - 1)]

        i = 0
        while len(channels[0]) < final_len:
            position = int(math.floor(i * step))

            for j in range(1, multi_channel.number_of_channels):
                chan = multi_channel.channel_at_index(j)
                channels[j - 1].append(chan[position])

            i += 1
        return DefaultMultiChannel(channels)

    @property
    def number_of_channels(self):
        return len(self._channels)

    def channel_at_index(self, index: int):
        return self._channels[index]
