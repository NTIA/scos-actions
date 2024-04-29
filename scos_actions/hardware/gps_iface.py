from abc import ABC, abstractmethod


class GPSInterface(ABC):
    @abstractmethod
    def get_location(
        self, sensor: "scos_actions.hardware.sensor.Sensor", timeout_s: float = 1
    ):
        pass

    @abstractmethod
    def get_gps_time(self, sensor: "scos_actions.hardware.sensor.Sensor"):
        pass
