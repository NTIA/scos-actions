from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np

from scos_actions.metadata.structs import ntia_algorithm


class DataProductGraphInfo(NamedTuple):
    """Lengths and SigMF graph metadata describing a data_product's data."""

    data_length: int
    total_length: int
    graph: ntia_algorithm.Graph
    dft: Optional[ntia_algorithm.DFT]


class DataProductResult(NamedTuple):
    """Computed data_product data, optional summary, and graph metadata."""

    data_result: np.ndarray
    summary: Optional[np.ndarray]
    graph_info: DataProductGraphInfo


class DataProductInterface(ABC):
    """Lifecycle shared by products configured from action-owned constants.

    Set per-capture parameters before computing data or building graph metadata.
    """

    def __init__(self, consts: dict):
        self._configure_from_constants(consts)

    @abstractmethod
    def _configure_from_constants(self, consts: dict) -> None:
        """
        Configure data_product state from the action-owned constants' dictionary.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """
        Set per-capture parameters before calling ``compute`` or ``get_graph``.
        """
        pass

    @abstractmethod
    def compute(
        self, iq: np.ndarray, recording_id: int
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute data_product data, an optional summary, and graph metadata.

        Parameters must already have been applied with ``set_parameters``.
        ``summary`` is ``None`` when the data_product has no summary values.
        """
        pass

    @abstractmethod
    def get_graph(self) -> DataProductGraphInfo:
        pass
