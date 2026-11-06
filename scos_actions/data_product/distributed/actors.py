"""Ray actor wrapper that holds one local data_product object for its lifetime."""
from typing import Optional

import numpy as np
import ray

from scos_actions.data_product.amplitude_probability_distribution import AmplitudeProbabilityDistribution
from scos_actions.data_product.interfaces import DataProductResult
from scos_actions.data_product.periodic_frame_power import PeriodicFramePower
from scos_actions.data_product.power_spectral_density import PowerSpectralDensity
from scos_actions.data_product.power_vs_time import PowerVsTime

_product_builders = {
    "psd": PowerSpectralDensity,
    "pvt": PowerVsTime,
    "pfp": PeriodicFramePower,
    "apd": AmplitudeProbabilityDistribution,
}


@ray.remote
class DataProductActor:
    """Builds one data_product object once and reuses it for every ``run`` call."""

    def __init__(self, name: str, consts: dict):
        self.data_product = _product_builders[name](consts)

    def run(
        self, iq: np.ndarray, params: dict
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        self.data_product.set_parameters(params)
        return self.data_product.compute(iq, params["recording_id"])

    def run_with_graph(
        self, iq: np.ndarray, params: dict
    ) -> DataProductResult:
        self.data_product.set_parameters(params)
        data_result, summary = self.data_product.compute(iq, params["recording_id"])
        return DataProductResult(
            data_result=data_result,
            summary=summary,
            graph_info=self.data_product.get_graph(),
        )
