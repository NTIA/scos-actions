"""Filter captured IQ and dispatch it to reusable product actors.

Product workers receive a reference to the shared filtered buffer, and their
results are returned in the order specified by ``params["data_products"]``.
"""
import logging

import numpy as np
import ray
from environs import Env
from scipy.signal import sosfilt

from scos_actions.data_product.distributed.actors import DataProductActor

env = Env()
logger = logging.getLogger(__name__)


@ray.remote
class IQProcessorProductActors:
    """
    Supervisor that spawns one long-lived ``ProductActor`` per data_product.

    Each worker builds its data_product object once, in its own process, and reuses
    it for every ``run`` call, so no data_product object is ever copied.
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

        # Ray's object store lets all requested products reuse this filtered buffer.
        iq_ref = ray.put(sosfilt(sos=iir_sos, x=iqdata))
        if self.compute_only:
            return tuple(
                self.workers[name].run.remote(iq_ref, params) for name in product_names
            )
        else:
            return tuple(
                self.workers[name].run_with_graph.remote(iq_ref, params) for name in product_names
            )