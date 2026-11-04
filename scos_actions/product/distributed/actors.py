"""Ray actor wrapper that holds one local product object for its lifetime."""

import numpy as np
import ray

from scos_actions.product.amplitude_probability_distribution import AmplitudeProbabilityDistribution
from scos_actions.product.interfaces import DataProductResult
from scos_actions.product.periodic_frame_power import PeriodicFramePower
from scos_actions.product.power_spectral_density import PowerSpectralDensity
from scos_actions.product.power_vs_time import PowerVsTime

_product_builders = {
    "psd": PowerSpectralDensity,
    "pvt": PowerVsTime,
    "pfp": PeriodicFramePower,
    "apd": AmplitudeProbabilityDistribution,
}


@ray.remote
class ProductActor:
    """Builds one product object once and reuses it for every ``run`` call."""

    def __init__(self, name: str, consts: dict):
        self.product = _product_builders[name](consts)

    def run(
        self, iq: np.ndarray, params: dict
    ) -> DataProductResult:
        return self.product.compute(iq, params)
