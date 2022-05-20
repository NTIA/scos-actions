from abc import ABC, abstractmethod
from scos_actions.actions.sigmf_builder import SigMFBuilder

class MetadataDecorator(ABC):

    def __init__(self, sigmf_builder:SigMFBuilder, start=None, length=None):
        self.sigmf_builder = sigmf_builder
        self.start = start
        self.length = length

    @abstractmethod
    def decorate(self, sigan_cal: dict, sensor_cal: dict, measurement_result: dict):
        pass