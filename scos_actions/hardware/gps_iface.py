from abc import ABC, abstractmethod

from scos_actions.hardware.sensor import Sensor


class GPSInterface(ABC):
    @abstractmethod
    def get_location(self, sensor : Sensor, timeout_s : float=1):
        pass

    @abstractmethod
    def get_gps_time(self, sensor : Sensor):
        pass
