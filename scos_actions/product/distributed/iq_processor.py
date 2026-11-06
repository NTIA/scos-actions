"""
Supervisor actors that filter IQ data and launch the data product computations.

The variants differ in how products are held and executed:

- ``IQProcessorActors``: one long-lived product actor per product, spawned by
  each supervisor (original design).
- ``IQProcessorProductActors``: like ``IQProcessorActors`` but the workers are
  generic ``ProductActor`` objects built from a parameter dict (option B).
- ``IQProcessorTasks``: product objects are shared by object reference and
  each product runs as a Ray task (option A).
- ``IQProcessorLocal``: the supervisor builds the products and computes them
  sequentially in its own process (option D).

``params`` are assumed identical for every channel. ``IQProcessorProductActors``
returns results in the configured product order; the other variants use the
fixed ``PRODUCT_NAMES`` order.
"""
import logging

import numpy as np
import ray
from environs import Env
from scipy.signal import sosfilt

from scos_actions.product.distributed.actors import DataProductActor

env = Env()
logger = logging.getLogger(__name__)


@ray.remote
class IQProcessorProductActors:
    """
    Supervisor that spawns one long-lived ``ProductActor`` per product.

    Each worker builds its product object once, in its own process, and reuses
    it for every ``run`` call, so no product object is ever copied.
    """

    def __init__(self, actor_id: int, compute_only: bool=False) -> None:
        self.actor_id = actor_id
        self.compute_only = compute_only
        logger.debug("IQProcessorProductActors initialized: %s", self.actor_id)
        self.workers = {}

    @ray.method(num_returns=1)
    def run(self, iqdata: np.ndarray, params: dict, iir_sos: np.ndarray, consts: dict) -> tuple:
        recording_id = params["recording_id"]
        logger.debug("IQProcessorProductActors run: %s, %s", self.actor_id, recording_id)

        product_names = params["data_products"]
        for name in product_names:
            if name not in self.workers:
                self.workers[name] = DataProductActor.remote(name, consts)

        """Return product references in the configured product order."""
        iq_ref = ray.put(sosfilt(sos=iir_sos, x=iqdata))
        if self.compute_only:
            return tuple(
                self.workers[name].run.remote(iq_ref, params) for name in product_names
            )
        else:
            return tuple(
                self.workers[name].run_with_graph.remote(iq_ref, params) for name in product_names
            )