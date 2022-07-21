from abc import ABC, abstractmethod
from scos_actions.actions.metadata.sigmf_builder import SigMFBuilder


class Metadata(ABC):

    def __init__(self, start=None, count=None, recording=None):
        self.start = start
        self.count = count
        self.recording = recording

    @abstractmethod
    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result: dict):
        pass
