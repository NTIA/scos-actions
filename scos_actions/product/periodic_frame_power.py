
"""Periodic Frame Power"""

import logging
from enum import EnumMeta

import numpy as np
import ray

from scos_actions.product.interfaces import ProductInterface
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_pseudo_power, create_statistical_detector,
)

logger = logging.getLogger(__name__)


class PeriodicFramePower(ProductInterface):
    """
    Periodic Frame Power
    """

    # PFP parameter keys
    PFP_FRAME_PERIOD_MS = "pfp_frame_period_ms"
    SAMPLE_RATE = "sample_rate"

    # PFP constants
    PFP_FRAME_RESOLUTION_S = (1e-3 * (1 + 1 / (14)) / 15) / 4
    IMPEDANCE_OHMS = 50.0

    def __init__(self, params: dict):
        super().__init__(params)


    """
    self,
        sample_rate_Hz: float,
        frame_period_ms: float,
        detector_period_s: float,
        impedance_ohms: float,
        detector: EnumMeta,
    ):
    
    params[SAMPLE_RATE],
            params[PFP_FRAME_PERIOD_MS],
            detector_period_s = PFP_FRAME_RESOLUTION_S,
    """

    def _configure_from_params(self, params: dict) -> None:
        #self.impedance_ohms = impedance_ohms
        self.detector = create_statistical_detector("PfpM3Detector", ["min", "max", "mean"])
        sampling_period_s = 1.0 / params[self.SAMPLE_RATE]
        frame_period_s = 1e-3 * params[self.PFP_FRAME_PERIOD_MS]
        if not np.isclose(frame_period_s % sampling_period_s, 0, 1e-6):
            raise ValueError(
                "Frame period must be a positive integer multiple of sampling period"
            )
        if not np.isclose(self.PFP_FRAME_RESOLUTION_S % sampling_period_s, 0, 1e-6):
            raise ValueError(
                "Detector period must be a positive integer multiple of sampling period"
            )
        self.n_frames = int(round(frame_period_s / sampling_period_s))
        self.n_points = int(round(frame_period_s / self.PFP_FRAME_RESOLUTION_S))
        self.n_detectors = len(self.detector)
        # PFP result is in pseudo-power log units, and needs to be scaled by
        # adding the following factor
        self.pfp_scale_factor = (
            -10.0 * np.log10(self.IMPEDANCE_OHMS)       # Conversion from pseudo-power
            + 27.0      # dBW to dBm (+30), baseband to RF (-3)
        )

    def compute(self, iq: np.ndarray) -> np.ndarray:
        """
        Compute periodic frame power results from IQ samples.

        The result is a 2D NumPy array, with the shape depending on the
        parameters which initialize this worker. In current form, the result
        is a (6, 560) NumPy array. Power samples are chunked into periodic frames,
        and statistical detectors are applied twice.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: A (6, 560) NumPy array containing the PFP results.
            The order is (min-of-mean, max-of-mean, mean-of-mean,
            min-of-max, max-of-max, mean-of-max).
        """
        # Set up dimensions to make the statistics fast
        chunked_shape = (
            iq.shape[0] // self.n_frames,
            self.n_points,
            self.n_frames // self.n_points,
        ) + tuple([iq.shape[1]] if iq.ndim == 2 else [])
        power_bins = calculate_pseudo_power(iq.reshape(chunked_shape))
        # compute statistics first by cycle
        mean_power = power_bins.mean(axis=0)
        max_power = power_bins.max(axis=0)
        del power_bins

        # then do the detector
        pfp = np.array(
            [
                apply_statistical_detector(p, self.detector, axis=1)
                for p in [mean_power, max_power]
            ]
        ).reshape(self.n_detectors * 2, self.n_points)

        # Finish conversion to power and scale result
        pfp = 10.0 * np.log10(pfp) + self.pfp_scale_factor
        return pfp


@ray.remote
class PeriodicFramePowerRayActor:
    """
    Compute periodic frame power results from IQ samples.

    The result is a 2D NumPy array, with the shape depending on the
    parameters which initialize this worker. In current form, the result
    is a (6, 560) NumPy array. Power samples are chunked into periodic frames,
    and statistical detectors are applied twice.
    """

    def __init__(
        self,
        sample_rate_Hz: float,
        frame_period_ms: float,
        detector_period_s: float,
        impedance_ohms: float,
        detector: EnumMeta,
    ):
        self.impedance_ohms = impedance_ohms
        sampling_period_s = 1.0 / sample_rate_Hz
        frame_period_s = 1e-3 * frame_period_ms
        if not np.isclose(frame_period_s % sampling_period_s, 0, 1e-6):
            raise ValueError(
                "Frame period must be a positive integer multiple of sampling period"
            )
        if not np.isclose(detector_period_s % sampling_period_s, 0, 1e-6):
            raise ValueError(
                "Detector period must be a positive integer multiple of sampling period"
            )
        self.n_frames = int(round(frame_period_s / sampling_period_s))
        self.n_points = int(round(frame_period_s / detector_period_s))
        self.n_detectors = len(detector)
        self.detector = detector
        # PFP result is in pseudo-power log units, and needs to be scaled by
        # adding the following factor
        self.pfp_scale_factor = (
            -10.0 * np.log10(impedance_ohms)  # Conversion from pseudo-power
            + 27.0  # dBW to dBm (+30), baseband to RF (-3)
        )

    def run(self, iq: ray.ObjectRef) -> np.ndarray:
        """
        Compute periodic frame power results from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: A (6, 560) NumPy array containing the PFP results.
            The order is (min-of-mean, max-of-mean, mean-of-mean,
            min-of-max, max-of-max, mean-of-max).
        """
        # Set up dimensions to make the statistics fast
        chunked_shape = (
            iq.shape[0] // self.n_frames,
            self.n_points,
            self.n_frames // self.n_points,
        ) + tuple([iq.shape[1]] if iq.ndim == 2 else [])
        power_bins = calculate_pseudo_power(iq.reshape(chunked_shape))
        # compute statistics first by cycle
        mean_power = power_bins.mean(axis=0)
        max_power = power_bins.max(axis=0)
        del power_bins

        # then do the detector
        pfp = np.array(
            [
                apply_statistical_detector(p, self.detector, axis=1)
                for p in [mean_power, max_power]
            ]
        ).reshape(self.n_detectors * 2, self.n_points)

        # Finish conversion to power and scale result
        pfp = 10.0 * np.log10(pfp) + self.pfp_scale_factor
        return pfp