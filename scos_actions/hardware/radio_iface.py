from abc import ABC, abstractmethod


class RadioInterface(ABC):
    @property
    @abstractmethod
    def is_available(self):
        pass

    @abstractmethod
    def acquire_time_domain_samples(self, num_samples, num_samples_skip=0, retries=5):
        raise NotImplementedError("Implement acquire_time_domain_samples")

    @abstractmethod
    def create_calibration_annotation(self):
        pass

    @property
    @abstractmethod
    def overload(self):
        pass

    @property
    @abstractmethod
    def capture_time(self):
        pass

    @property
    @abstractmethod
    def last_calibration_time(self):
        pass
