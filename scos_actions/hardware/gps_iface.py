from abc import ABC, abstractmethod


class GPSInterface(ABC):

    @abstractmethod
    def get_lat_long(timeout_s=1):
        pass