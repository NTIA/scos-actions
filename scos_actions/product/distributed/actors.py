"""Ray actor wrapper that holds one local product object for its lifetime."""

import numpy as np
import ray

from scos_actions.product.distributed.tasks import _product_builders


@ray.remote
class ProductActor:
    """Builds one product object once and reuses it for every ``run`` call."""

    def __init__(self, name: str, params: dict):
        self.product = _product_builders[name](params)

    def run(self, iq: np.ndarray):
        return self.product.compute(iq)
