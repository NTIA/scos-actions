from abc import ABC, abstractmethod

import numpy as np


class ProductInterface(ABC):
    """
    Common interface for product processors initialized from params.
    """

    def __init__(self, params: dict):
        self._configure_from_params(params)

    @abstractmethod
    def _configure_from_params(self, params: dict) -> None:
        """
        Configure product state from a params dictionary.
        """
        pass

    @abstractmethod
    def compute(self, iq: np.ndarray) -> np.ndarray:
        """
        Compute a product result from IQ samples.
        """
        pass
