from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np

from scos_actions.metadata.structs import ntia_algorithm


class DataProductGraphInfo(NamedTuple):
    """Lengths and SigMF graph metadata describing a product's data."""

    data_length: int
    total_length: int
    graph: ntia_algorithm.Graph
    dft: Optional[ntia_algorithm.DFT]


class DataProductResult(NamedTuple):
    """Computed product data, optional summary, and graph metadata."""

    data_result: np.ndarray
    summary: Optional[np.ndarray]
    graph_info: DataProductGraphInfo


class DataProductInterface(ABC):
    """
    Common interface for data product processors initialized from action constants.
    """

    def __init__(self, consts: dict):
        self._configure_from_constants(consts)

    @abstractmethod
    def _configure_from_constants(self, consts: dict) -> None:
        """
        Configure product state from the action-owned constants' dictionary.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """
        Set product parameters from a parameters' dictionary.
        """
        pass

    @abstractmethod
    def compute(
        self, iq: np.ndarray, params: dict
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute product data, an optional summary, and graph metadata.

        ``summary`` is ``None`` when the product has no summary values.
        """
        pass

    @abstractmethod
    def get_graph(self) -> DataProductGraphInfo:
        pass
