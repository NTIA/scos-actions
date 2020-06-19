from abc import ABC, abstractmethod


class RadioInterface(ABC):
    @property
    @abstractmethod
    def is_available(self):
        pass

    @abstractmethod
    def acquire_time_domain_samples(self, num_samples, num_samples_skip=0, retries=5) -> dict:
        """ Acquires time domain IQ samples
        :type num_samples: integer
        :param num_samples: Number of samples to acquire
    
        :type num_samples_skip: integer
        :param num_samples_skip: Number of samples to skip
    
        :type retries: integer
        :param retries: Maximum number of retries on failure
    
        :rtype: dictionary containing data, sample_rate, frequency, capture_time, etc
        """
        raise NotImplementedError("Implement acquire_time_domain_samples")

    @abstractmethod
    def create_calibration_annotation(self):
        pass

    @property
    @abstractmethod
    def last_calibration_time(self):
        pass
