
"""Periodic Frame Power"""

import logging

import numpy as np

from scos_actions.metadata.structs import ntia_algorithm
from scos_actions.product.interfaces import (
    DataProductInterface,
    DataProductResult,
    ProductGraphInfo,
)
from scos_actions.signal_processing.power_analysis import (
    apply_statistical_detector,
    calculate_pseudo_power,
)

logger = logging.getLogger(__name__)


# PFP parameter keys
PFP_FRAME_PERIOD_MS = "pfp_frame_period_ms"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
TD_BIN_SIZE_MS = "td_bin_size_ms"


class PeriodicFramePower(DataProductInterface):
    """
    Periodic Frame Power
    """
    def __init__(self, consts: dict):
        super().__init__(consts)

    def _configure_from_constants(self, consts: dict) -> None:
        self.detector = consts["PFP_M3_DETECTOR"]
        self.impedance_ohms = consts["IMPEDANCE_OHMS"]
        self.detector_period_s = consts["PFP_FRAME_RESOLUTION_S"]

    def set_parameters(self, params: dict) -> None:
        self.sample_rate = params[SAMPLE_RATE]
        self.duration_ms = params[DURATION_MS]
        self.td_bin_size_ms = params[TD_BIN_SIZE_MS]
        sampling_period_s = 1.0 / self.sample_rate
        self.pfp_frame_period_ms = params[PFP_FRAME_PERIOD_MS]
        frame_period_s = 1e-3 * self.pfp_frame_period_ms
        if not np.isclose(frame_period_s % sampling_period_s, 0, 1e-6):
            raise ValueError(
                "Frame period must be a positive integer multiple of sampling period"
            )
        if not np.isclose(self.detector_period_s % sampling_period_s, 0, 1e-6):
            raise ValueError(
                "Detector period must be a positive integer multiple of sampling period"
            )
        self.n_frames = int(round(frame_period_s / sampling_period_s))
        self.n_points = int(round(frame_period_s / self.detector_period_s))
        self.n_detectors = len(self.detector)

        # PFP result is in pseudo-power log units, and needs to be scaled by
        # adding the following factor
        self.pfp_scale_factor = (
                -10.0 * np.log10(self.impedance_ohms)  # Conversion from pseudo-power
                + 27.0  # dBW to dBm (+30), baseband to RF (-3)
        )

    def compute(
        self, iq: np.ndarray, params: dict
    ) -> DataProductResult:
        """
        Compute periodic frame power results from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: PFP data, no summary, and graph metadata.
        """
        # Set up dimensions to make the statistics fast
        recording_id = params["recording_id"]
        logger.warning("PeriodicFramePower started: %s", recording_id)

        self.set_parameters(params)

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

        logger.warning("PeriodicFramePower completed: %s", recording_id)
        return DataProductResult(pfp, None, self.get_graph())

    def get_graph(self) -> ProductGraphInfo:
        pvt_length = int(self.duration_ms / self.td_bin_size_ms)
        pfp_length = int(round((self.pfp_frame_period_ms / 1e3) / self.detector_period_s))
        pfp_x_axis__s = np.arange(pfp_length) * (
                self.duration_ms / 1e3 / pfp_length / pvt_length
        )
        pfp_graph = ntia_algorithm.Graph(
            name="Periodic Frame Power",
            series=[
                f"{det}_{stat.value}"
                for det in ["mean", "max"]
                for stat in self.detector
            ],
            length=pfp_length,
            x_units="s",
            x_start=[pfp_x_axis__s[0]],
            x_stop=[pfp_x_axis__s[-1]],
            x_step=[pfp_x_axis__s[1] - pfp_x_axis__s[0]],
            y_units="dBm",
            #reference=data_products_reference,
            description=(
                    "Channelized periodic frame power statistics reported over"
                    + f" a {self.pfp_frame_period_ms} ms frame period, with frame resolution"
                    + f" of {self.detector_period_s} s. Mean and max detectors are first "
                    + f"applied over the frame resolution, then {[d.value for d in self.detector]} statistics"
                    + " are computed on samples sharing the same index within the frame period."
            ),
        )

        pfp_total_length = pfp_length * len(self.detector) * 2

        return ProductGraphInfo(pfp_length, pfp_total_length, pfp_graph, None)