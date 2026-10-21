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

All variants expose ``run(iqdata)`` returning four values in the order
[FFT, PVT, PFP, APD]. ``params`` are assumed identical for every channel.
"""

import numpy as np
import ray
from environs import Env
from scipy.signal import sosfilt

from scos_actions.product import product_classes
from scos_actions.product.distributed.actors import ProductActor
from scos_actions.product.distributed.tasks import (
    PRODUCT_NAMES,
    build_local_products,
    build_products,
    compute_product,
    compute_product_pinned,
    filter_iq,
)
from scos_actions.signal_processing.power_analysis import create_statistical_detector

env = Env()

# Supervisor restarts and re-run of interrupted calls (env
# RAY_SUPERVISOR_MAX_RESTARTS, RAY_SUPERVISOR_MAX_TASK_RETRIES); -1 means unlimited.
SUPERVISOR_MAX_RESTARTS = env.int("RAY_SUPERVISOR_MAX_RESTARTS", default=2)
SUPERVISOR_MAX_TASK_RETRIES = env.int("RAY_SUPERVISOR_MAX_TASK_RETRIES", default=2)

# Parameter keys read by IQProcessorActors
NUM_FFTS = "nffts"
FFT_SIZE = "fft_size"
APD_BIN_SIZE_DB = "apd_bin_size_dB"
APD_MIN_BIN_DBM = "apd_min_bin_dBm"
APD_MAX_BIN_DBM = "apd_max_bin_dBm"
TD_BIN_SIZE_MS = "td_bin_size_ms"
SAMPLE_RATE = "sample_rate"
PFP_FRAME_PERIOD_MS = "pfp_frame_period_ms"

# Constants used by IQProcessorActors
PFP_FRAME_RESOLUTION_S = (1e-3 * (1 + 1 / (14)) / 15) / 4
FFT_PERCENTILES = np.array([25, 75, 90, 95, 99, 99.9, 99.99])
FFT_WINDOW_TYPE = "flattop"
FFT_WINDOW_CORRECTION_TYPE = "energy"
IMPEDANCE_OHMS = 50.0

TD_DETECTOR = create_statistical_detector("TdMeanMaxDetector", ["max", "mean"])
FFT_M3_DETECTOR = create_statistical_detector(
    "FftM3Detector", ["max", "mean", "median"]
)
PFP_M3_DETECTOR = create_statistical_detector("PfpM3Detector", ["min", "max", "mean"])


@ray.remote
class IQProcessorActors:
    """
    Supervisor actor that owns one product worker actor per product.

    Upon initialization, workers are spawned for each component of the data
    product. The workers are stateful and reuse values that do not depend on
    the IQ data for each consecutive run. ``run`` filters the IQ data, then
    calls the workers concurrently and returns Ray object references
    immediately, which can be used later to retrieve the results.
    """

    def __init__(self, params: dict, iir_sos: np.ndarray):
        self.iir_sos = iir_sos

        self.fft_worker = product_classes["psd"].remote(
            params[SAMPLE_RATE],
            params[NUM_FFTS],
            params[FFT_SIZE],
            fft_window_type=FFT_WINDOW_TYPE,
            fft_window_correction_type=FFT_WINDOW_CORRECTION_TYPE,
            detector=FFT_M3_DETECTOR,
            percentiles=FFT_PERCENTILES,
            impedance_ohms=IMPEDANCE_OHMS,
        )
        self.pvt_worker = product_classes["pvt"].remote(
            params[SAMPLE_RATE],
            params[TD_BIN_SIZE_MS],
            impedance_ohms=IMPEDANCE_OHMS,
            detector=TD_DETECTOR,
        )
        self.pfp_worker = product_classes["pfp"].remote(
            params[SAMPLE_RATE],
            params[PFP_FRAME_PERIOD_MS],
            detector_period_s=PFP_FRAME_RESOLUTION_S,
            impedance_ohms=IMPEDANCE_OHMS,
            detector=PFP_M3_DETECTOR,
        )
        self.apd_worker = product_classes["apd"].remote(
            params[APD_BIN_SIZE_DB],
            params[APD_MIN_BIN_DBM],
            params[APD_MAX_BIN_DBM],
            impedance_ohms=IMPEDANCE_OHMS,
        )

    @ray.method(num_returns=4)
    def run(self, iqdata: np.ndarray) -> list:
        """
        :return: Object references in the order [FFT, PVT, PFP, APD].
        """
        iqdata = ray.put(sosfilt(sos=self.iir_sos, x=iqdata))
        fft_reference = self.fft_worker.run.remote(iqdata)
        pvt_reference = self.pvt_worker.run.remote(iqdata)
        pfp_reference = self.pfp_worker.run.remote(iqdata)
        apd_reference = self.apd_worker.run.remote(iqdata)
        return fft_reference, pvt_reference, pfp_reference, apd_reference


@ray.remote
class IQProcessorProductActors:
    """
    Supervisor that spawns one long-lived ``ProductActor`` per product.

    Each worker builds its product object once, in its own process, and reuses
    it for every ``run`` call, so no product object is ever copied. This keeps
    ``RAY_WORKERS * 4`` worker processes alive for the whole run.
    """

    def __init__(self, params: dict, iir_sos: np.ndarray):
        self.iir_sos = iir_sos
        self.workers = {
            name: ProductActor.remote(name, params) for name in PRODUCT_NAMES
        }

    @ray.method(num_returns=4)
    def run(self, iqdata: np.ndarray) -> tuple:
        """
        :return: Object references in the order [FFT, PVT, PFP, APD].
        """
        iq_ref = ray.put(sosfilt(sos=self.iir_sos, x=iqdata))
        return tuple(self.workers[name].run.remote(iq_ref) for name in PRODUCT_NAMES)


@ray.remote(
    num_cpus=0,
    max_restarts=SUPERVISOR_MAX_RESTARTS,
    max_task_retries=SUPERVISOR_MAX_TASK_RETRIES,
)
class IQProcessorTasks:
    """
    Supervisor that launches one ``compute_product`` task per product.

    Builds the product objects once from ``params`` and puts them in the
    object store (``build_products``). For each ``run`` call it
    filters the IQ data, places it in the object store once, and launches the
    four tasks, each in its own worker process with 1 CPU reserved. Object
    references are returned immediately without waiting.

    If the supervisor process dies it is restarted (rebuilding its products)
    and the interrupted ``run`` calls are retried.
    """

    def __init__(self, params: dict, iir_sos: np.ndarray):
        self.products = build_products(params)
        self.iir_sos = iir_sos

    @ray.method(num_returns=4)
    def run(self, iqdata: np.ndarray) -> tuple:
        """
        :return: Object references in the order [FFT, PVT, PFP, APD].
        """
        iq_ref = ray.put(sosfilt(sos=self.iir_sos, x=iqdata))
        return tuple(
            compute_product.remote(self.products[name], iq_ref)
            for name in PRODUCT_NAMES
        )


@ray.remote(
    num_cpus=0,
    max_restarts=SUPERVISOR_MAX_RESTARTS,
    max_task_retries=SUPERVISOR_MAX_TASK_RETRIES,
)
class IQProcessorTasks2:
    """
    Option A2: like ``IQProcessorTasks`` but the supervisor does no compute.

    Filtering runs in a ``filter_iq`` task (1 CPU) and its output reference is
    passed straight to the four single-thread ``compute_product_pinned`` tasks.
    The supervisor only holds the product references and launches tasks.
    """

    def __init__(self, params: dict, iir_sos: np.ndarray):
        self.products = build_products(params)
        self.iir_sos = iir_sos

    @ray.method(num_returns=4)
    def run(self, iqdata: np.ndarray) -> tuple:
        """
        :return: Object references in the order [FFT, PVT, PFP, APD].
        """
        iq_ref = filter_iq.remote(iqdata, self.iir_sos)
        return tuple(
            compute_product_pinned.remote(self.products[name], iq_ref)
            for name in PRODUCT_NAMES
        )


@ray.remote(num_cpus=1)
class IQProcessorLocal:
    """
    Supervisor that builds the four products once and computes them itself.

    There is no parallelism inside a channel: parallelism comes only from
    running ``RAY_WORKERS`` supervisors on different channels. Each supervisor
    reserves 1 CPU.
    """

    def __init__(self, params: dict, iir_sos: np.ndarray):
        self.iir_sos = iir_sos
        self.products = build_local_products(params)

    @ray.method(num_returns=4)
    def run(self, iqdata: np.ndarray) -> tuple:
        """
        :return: Results in the order [FFT, PVT, PFP, APD].
        """
        iq = sosfilt(sos=self.iir_sos, x=iqdata)
        return tuple(self.products[name].compute(iq) for name in PRODUCT_NAMES)
