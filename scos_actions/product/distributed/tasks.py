"""Ray task wrappers for the SEA data products.

``build_products`` builds the local product objects once per action run
(for example, ``PowerSpectralDensity``, from the shared ``params`` dict) and
puts them in the object store. The object references are passed to the
compute tasks, which call ``compute`` on them. Product objects are treated
as read-only by ``compute``.
"""

import numpy as np
import ray
from environs import Env

from scos_actions.product.amplitude_probability_distribution import (
    AmplitudeProbabilityDistribution,
)
from scos_actions.product.periodic_frame_power import PeriodicFramePower
from scos_actions.product.power_spectral_density import PowerSpectralDensity
from scos_actions.product.power_vs_time import PowerVsTime

_product_builders = {
    "psd": PowerSpectralDensity,
    "pvt": PowerVsTime,
    "pfp": PeriodicFramePower,
    "apd": AmplitudeProbabilityDistribution,
}

env = Env()

# Retries of a task whose worker process died (env RAY_TASK_MAX_RETRIES)
TASK_MAX_RETRIES = env.int("RAY_TASK_MAX_RETRIES", default=3)
# numexpr threads per compute task (env RAY_TASK_THREADS)
TASK_THREADS = env.int("RAY_TASK_THREADS", default=4)

PRODUCT_NAMES = ("psd", "pvt", "pfp", "apd")

# Parameters that determine each product object's state
PRODUCT_PARAM_KEYS = {
    "psd": ("sample_rate", "nffts", "fft_size"),
    "pvt": ("sample_rate", "td_bin_size_ms"),
    "pfp": ("sample_rate", "pfp_frame_period_ms"),
    "apd": ("apd_bin_size_dB", "apd_min_bin_dBm", "apd_max_bin_dBm"),
}


def product_key(name: str, params: dict) -> tuple:
    """Key identifying the product object needed for ``name`` and ``params``."""
    return (name,) + tuple(params[k] for k in PRODUCT_PARAM_KEYS[name])


def build_product_ref(name: str, params: dict) -> ray.ObjectRef:
    """Build one local product object and put it in the Ray object store.

    Building locally raises invalid-parameter errors immediately in the caller.
    """
    return ray.put(_product_builders[name](params))


def build_local_products(params: dict) -> dict:
    """Build the local product objects (no Ray object store involved)."""
    return {name: _product_builders[name](params) for name in PRODUCT_NAMES}


def build_products(params: dict) -> dict:
    """Build the local product objects and put them in the Ray object store.

    Building locally raises invalid-parameter errors immediately in the caller.

    :return: Dict mapping product name to an object reference.
    """
    return {name: ray.put(_product_builders[name](params)) for name in PRODUCT_NAMES}


@ray.remote(num_cpus=1, max_retries=TASK_MAX_RETRIES)
def compute_product(product, iq: np.ndarray):
    """Run ``product.compute`` on IQ data in its own worker process."""
    return product.compute(iq)


_threads_pinned = False


def _pin_threads() -> None:
    """Limit numexpr in this worker process to ``TASK_THREADS`` threads.

    A task reserves 1 CPU, but numexpr would otherwise start ``cpu_count``
    threads per task and oversubscribe the machine. (The PSD FFT already uses
    ``workers=1``.) Runs once per worker process.
    """
    global _threads_pinned
    if _threads_pinned:
        return
    import numexpr as ne

    ne.set_num_threads(TASK_THREADS)
    _threads_pinned = True


@ray.remote(num_cpus=1, max_retries=TASK_MAX_RETRIES)
def filter_iq(iq: np.ndarray, iir_sos: np.ndarray) -> np.ndarray:
    """Low-pass filter IQ data in its own worker process (1 CPU reserved)."""
    from scipy.signal import sosfilt

    return sosfilt(sos=iir_sos, x=iq)


@ray.remote(num_cpus=1, max_retries=TASK_MAX_RETRIES)
def compute_product_pinned(product, iq: np.ndarray):
    """Like ``compute_product`` but limited to ``TASK_THREADS`` numexpr threads."""
    _pin_threads()
    return product.compute(iq)
