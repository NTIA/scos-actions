from abc import ABC, abstractmethod


class GPSInterface(ABC):
    @abstractmethod
    def get_location(self, timeout_s=1):
        pass

    @abstractmethod
    def get_gps_time(self):
        pass
