# What follows is a parameterizable description of the algorithm used by this
# action. The first line is the summary and should be written in plain text.
# Everything following that is the extended description, which can be written
# in Markdown and MathJax. Each name in curly brackets '{}' will be replaced
# with the value specified in the `description` method which can be found at
# the very bottom of this file. Since this parameterization step affects
# everything in curly brackets, math notation such as {m \over n} must be
# escaped to {{m \over n}}.
#
# To print out this docstring after parameterization, see
# scos-sensor/scripts/print_action_docstring.py. You can then paste that into the
# SCOS Markdown Editor (link below) to see the final rendering.
#
# Resources:
# - MathJax reference: https://math.meta.stackexchange.com/q/5020
# - Markdown reference: https://commonmark.org/help/
# - SCOS Markdown Editor: https://ntia.github.io/scos-md-editor/
#
r"""Acquire a NASCTN SEA data product.

Currently in development.
"""
import gc
import logging
import lzma
from time import perf_counter
from typing import Tuple

import numexpr as ne
import numpy as np
import ray
from scipy.signal import sosfilt

from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import measurement_action_completed
from scos_actions.hardware import gps as mock_gps
from scos_actions.metadata.annotation_segment import AnnotationSegment
from scos_actions.metadata.annotations import (
    CalibrationAnnotation,
    FrequencyDomainDetection,
    SensorAnnotation,
    TimeDomainDetection,
)
from scos_actions.metadata.sigmf_builder import SigMFBuilder
from scos_actions.signal_processing.apd import get_apd
from scos_actions.signal_processing.fft import (
    get_fft,
    get_fft_window,
    get_fft_window_correction,
)
from scos_actions.signal_processing.filtering import (
    generate_elliptic_iir_low_pass_filter,
)
from scos_actions.signal_processing.power_analysis import (
    apply_power_detector,
    calculate_power_watts,
    calculate_pseudo_power,
    create_power_detector,
    filter_quantiles,
)
from scos_actions.signal_processing.unit_conversion import (
    convert_linear_to_dB,
    convert_watts_to_dBm,
)

logger = logging.getLogger(__name__)

# Define parameter keys
RF_PATH = "rf_path"
IIR_APPLY = "iir_apply"
IIR_GPASS = "iir_gpass_dB"
IIR_GSTOP = "iir_gstop_dB"
IIR_PB_EDGE = "iir_pb_edge_Hz"
IIR_SB_EDGE = "iir_sb_edge_Hz"
IIR_RESP_FREQS = "iir_num_response_frequencies"
QFILT_APPLY = "qfilt_apply"
Q_LO = "qfilt_qlo"
Q_HI = "qfilt_qhi"
# FFT_SIZE = "fft_size"
NUM_FFTS = "nffts"
# FFT_WINDOW_TYPE = "fft_window_type"
APD_BIN_SIZE_DB = "apd_bin_size_dB"
TD_BIN_SIZE_MS = "td_bin_size_ms"
ROUND_TO = "round_to_places"
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"
PFP_FRAME_PERIOD_MS = "pfp_frame_period_ms"

# Constants
DATA_TYPE = np.half
PFP_FRAME_RESOLUTION_S = (1e-3 * (1 + 1 / (14)) / 15) / 4


def get_fft_results(iqdata: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute data product mean/max FFT results from IQ samples."""
    # IQ data already scaled for calibrated gain
    fft_result = get_fft(
        time_data=iqdata,
        fft_size=FFT_SIZE,
        norm="backward",
        fft_window=FFT_WINDOW,
        num_ffts=params[NUM_FFTS],
        shift=False,
        workers=4,
    )
    fft_result = calculate_pseudo_power(fft_result)
    fft_result = apply_power_detector(fft_result, FFT_DETECTOR)  # (max, mean)
    ne.evaluate("fft_result/50", out=fft_result)  # Finish conversion to Watts
    # Shift frequencies of reduced result
    fft_result = np.fft.fftshift(fft_result, axes=(1,))
    fft_result = convert_watts_to_dBm(fft_result)
    fft_result -= 3  # Baseband/RF power conversion
    fft_result -= 10.0 * np.log10(params[SAMPLE_RATE] * FFT_SIZE)  # PSD scaling
    fft_result += 2.0 * convert_linear_to_dB(FFT_WINDOW_ECF)  # Window energy correction

    # Truncate FFT result to middle 625 samples (middle 10 MHz from 14 MHz)
    # bw_trim = (params[SAMPLE_RATE] / 1.4) / 5  # Bandwidth to trim from each side: 2 MHz
    # delta_f = params[SAMPLE_RATE] / FFT_SIZE  # 875 bins -> 16 kHz
    # bin_start = int(bw_trim / delta_f)  # Bin 125
    bin_start = int(FFT_SIZE / 7)  # bin_start = 125 with FFT_SIZE 875
    bin_end = FFT_SIZE - bin_start  # bin_end = 750 with FFT_SIZE 875
    fft_result = fft_result[:, bin_start:bin_end]  # See comments above

    return fft_result[0], fft_result[1]


def get_apd_results(iqdata: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Generate downsampled APD result from IQ samples."""
    p, a = get_apd(iqdata, params[APD_BIN_SIZE_DB])
    # Convert dBV to dBm:
    # a = a * 2 : dBV --> dB(V^2)
    # a = a - impedance_dB : dB(V^2) --> dBW
    # a = a + 27 : dBW --> dBm (+30) and RF/baseband conversion (-3)
    scale_factor = 27 - convert_linear_to_dB(50.0)  # Hard-coded for 50 Ohms.
    ne.evaluate("(a*2)+scale_factor", out=a)
    return p, a


def get_td_power_results(
    iqdata: np.ndarray, params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/max time domain power statistics from IQ samples, with optional quantile filtering."""
    # Reshape IQ data into blocks
    block_size = int(params[TD_BIN_SIZE_MS] * params[SAMPLE_RATE] * 1e-3)
    n_blocks = len(iqdata) // block_size
    iqdata = iqdata.reshape(block_size, n_blocks)
    print(
        f"Calculating time-domain power statistics on {n_blocks} blocks of {block_size} samples each"
    )

    iq_pwr = calculate_power_watts(iqdata, impedance_ohms=50.0)

    if params[QFILT_APPLY]:
        # Apply quantile filtering before computing power statistics
        print("Quantile-filtering time domain power data...")
        iq_pwr = filter_quantiles(iq_pwr, params[Q_LO], params[Q_HI])
        # Diagnostics
        num_nans = np.count_nonzero(np.isnan(iq_pwr))
        nan_pct = num_nans * 100 / len(iq_pwr.flatten())
        print(f"Rejected {num_nans} samples ({nan_pct:.2f}% of total capture)")
    else:
        print("Quantile-filtering disabled. Skipping...")

    # Apply mean/max detectors
    td_result = apply_power_detector(iq_pwr, TD_DETECTOR, ignore_nan=True)

    # Convert to dBm
    td_result = convert_watts_to_dBm(td_result)

    # Account for RF/baseband power difference
    td_result -= 3

    return td_result[0], td_result[1]  # (max, mean)


def get_periodic_frame_power(
    iqdata: np.ndarray,
    params: dict,
    detector_period_s: float = PFP_FRAME_RESOLUTION_S,
) -> dict:
    """
    Compute a time series of periodic frame power statistics.

    The time axis on the frame time elapsed spans [0, frame_period) binned with step size
    `detector_period`, for a total of `int(frame_period/detector_period)` samples.
    RMS and peak power detector data are returned. For each type of detector, a time
    series is returned for (min, mean, max) statistics, which are computed across the
    number of frames (`frame_period/Ts`).

    :param iqdata: Complex-valued input waveform samples.
    :param params:
    :param detector_period_s: Sampling period (s) within the frame.
    :return: A dict with keys "rms" and "peak", each with values (min: np.ndarray, mean: np.ndarray,
        max: np.ndarray)
    :raises ValueError: if detector_period%Ts != 0 or frame_period%detector_period != 0
    """
    sampling_period_s = 1.0 / params[SAMPLE_RATE]
    frame_period_s = 1e-3 * params[PFP_FRAME_PERIOD_MS]
    if not np.isclose(frame_period_s % sampling_period_s, 0, 1e-6):
        raise ValueError(
            "frame period must be positive integer multiple of the sampling period"
        )

    if not np.isclose(detector_period_s % sampling_period_s, 0, 1e-6):
        raise ValueError(
            "detector_period period must be positive integer multiple of the sampling period"
        )

    Nframes = int(np.round(frame_period_s / sampling_period_s))
    Npts = int(np.round(frame_period_s / detector_period_s))
    print(f"PFP Nframes: {Nframes}, Npts: {Npts}")

    # set up dimensions to make the statistics fast
    chunked_shape = (iqdata.shape[0] // Nframes, Npts, Nframes // Npts) + tuple(
        [iqdata.shape[1]] if iqdata.ndim == 2 else []
    )
    iq_bins = iqdata.reshape(chunked_shape)
    power_bins = calculate_pseudo_power(iq_bins)

    # compute statistics first by cycle
    rms_power = power_bins.mean(axis=0)
    peak_power = power_bins.max(axis=0)

    # then do the detector
    pfp = np.array(
        [
            apply_power_detector(p, PFP_M3_DETECTOR, axis=1)
            for p in [rms_power, peak_power]
        ]
    ).reshape(6, Npts)

    # Finish conversion to power
    ne.evaluate("pfp/50", out=pfp)

    # Convert to dBm
    pfp = convert_watts_to_dBm(pfp)
    pfp -= 3  # RF/baseband
    return tuple(pfp)


@ray.remote
def generate_data_product(
    iqdata: np.ndarray, params: dict, iir_sos: np.ndarray
) -> np.ndarray:
    """Process IQ data and generate the SEA data product."""
    # Use print instead of logger.debug inside ray.remote function
    print(f"Generating data product @ {params[FREQUENCY]}...")
    tic1 = perf_counter()
    data_product = []

    data_product.extend(get_fft_results(iqdata, params))
    toc = perf_counter()
    print(f"Got FFT result @ {params[FREQUENCY]} in {toc-tic1:.2f} s")

    tic = perf_counter()
    iqdata = sosfilt(iir_sos, iqdata)
    toc = perf_counter()
    print(f"Applied IIR filter to IQ data @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    tic = perf_counter()
    data_product.extend(get_td_power_results(iqdata, params))
    toc = perf_counter()
    print(f"Got TD result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    tic = perf_counter()
    data_product.extend(get_periodic_frame_power(iqdata, params))
    toc = perf_counter()
    print(f"Got PFP result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    tic = perf_counter()
    data_product.extend(get_apd_results(iqdata, params))
    toc = perf_counter()
    print(f"Got APD result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    print(f"Got all data product @ {params[FREQUENCY]} results in {toc-tic1:.2f} s")

    # TODO: Further optimize memory usage
    print(f"GC Count: {gc.get_count()}")
    tic = perf_counter()
    del iqdata
    gc.collect()
    toc = perf_counter()
    print(f"GC Count after collection: {gc.get_count()}")
    print(f"Deleted IQ @ {params[FREQUENCY]} and collected garbage in {toc-tic:.2f} s")

    # Skip rounding for now (may return to this if it benefits other compression methods)
    # Quantize power results
    # tic = perf_counter()
    # for i, data in enumerate(data_product):
    #     if i == 4:
    #         # Do not round APD probability axis
    #         continue
    #     data.round(decimals=params[ROUND_TO], out=data)
    # toc = perf_counter()
    # logger.debug(
    #     f"Data product rounded to {params[ROUND_TO]} decimal places in {toc-tic:.2f} s"
    # )

    # Flatten data product but retain component indices
    tic = perf_counter()
    data_product, dp_idx = NasctnSeaDataProduct.transform_data(data_product)
    toc = perf_counter()
    print(f"Data @ {params[FREQUENCY]} transformed in {toc-tic:.2f} s")

    return data_product, dp_idx


# Hard-coded algorithm parameters
FFT_SIZE = 875
FFT_WINDOW_TYPE = "flattop"

# Generate FFT window and correction factor
FFT_WINDOW = get_fft_window(FFT_WINDOW_TYPE, FFT_SIZE)
FFT_WINDOW_ECF = get_fft_window_correction(FFT_WINDOW, "energy")

# Create power detectors
TD_DETECTOR = create_power_detector("TdMeanMaxDetector", ["mean", "max"])
FFT_DETECTOR = create_power_detector("FftMeanMaxDetector", ["mean", "max"])
PFP_M3_DETECTOR = create_power_detector("PfpM3Detector", ["min", "max", "mean"])


class NasctnSeaDataProduct(Action):
    """Acquire a stepped-frequency NASCTN SEA data product.

    :param parameters: The dictionary of parameters needed for
        the action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)
        # Assume preselector is present
        rf_path_name = utils.get_parameter(RF_PATH, self.parameters)
        self.rf_path = {self.PRESELECTOR_PATH_KEY: rf_path_name}

        # Setup/pull config parameters
        # TODO: Some parameters in this section should end up hard-coded
        # For now they are all parameterized in the action config for testing
        self.iir_gpass_dB = utils.get_parameter(IIR_GPASS, self.parameters)
        self.iir_gstop_dB = utils.get_parameter(IIR_GSTOP, self.parameters)
        self.iir_pb_edge_Hz = utils.get_parameter(IIR_PB_EDGE, self.parameters)
        self.iir_sb_edge_Hz = utils.get_parameter(IIR_SB_EDGE, self.parameters)
        self.sample_rate_Hz = utils.get_parameter(SAMPLE_RATE, self.parameters)

        # Construct IIR filter
        self.iir_sos = generate_elliptic_iir_low_pass_filter(
            self.iir_gpass_dB,
            self.iir_gstop_dB,
            self.iir_pb_edge_Hz,
            self.iir_sb_edge_Hz,
            self.sample_rate_Hz,
        )

        # TODO: remove config parameters which will be hard-coded
        for key in [
            IIR_GPASS,
            IIR_GSTOP,
            IIR_PB_EDGE,
            IIR_SB_EDGE,
            IIR_RESP_FREQS,
            # Q_LO,
            # Q_HI,
            # FFT_WINDOW_TYPE,
        ]:
            self.parameters.pop(key)

    def __call__(self, schedule_entry, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()

        iteration_params = utils.get_iterable_parameters(self.parameters)

        start_action = perf_counter()
        logger.debug(f"Setting RF path to {self.rf_path}")
        self.configure_preselector(self.rf_path)

        # Collect all IQ data and spawn data product computation processes
        # all_data, all_idx, dp_procs, start_times, end_times, sensor_cals = ([] for _ in range(5))
        all_data, all_idx, dp_procs, cap_meta = ([] for _ in range(4))
        for parameters in iteration_params:
            # Capture IQ data
            measurement_result = self.capture_iq(parameters)

            cap_meta.append(
                {
                    "start_time": measurement_result["start_time"],
                    "end_time": measurement_result["end_time"],
                    "sensor_cal": measurement_result["sensor_cal"],
                    "overload": measurement_result["overload"],
                }
            )

            # Start data product processing but do not stall before next IQ capture
            dp_procs.append(
                generate_data_product.remote(
                    measurement_result["data"], parameters, self.iir_sos
                )
            )

        # Initialize metadata object
        # Assumes all sample rates are the same
        # And uses a single "last calibration time"
        self.get_sigmf_builder(
            iteration_params[0][SAMPLE_RATE],
            task_id,
            schedule_entry,
        )

        # Collect processed data product results
        last_data_len = 0
        results = ray.get(dp_procs)  # Ordering is retained
        for rec_id, (dp, dp_idx) in enumerate(results):
            all_data.extend(dp)
            all_idx.extend((dp_idx + last_data_len).tolist())

            cap_meta[rec_id]["sample_start"] = last_data_len
            cap_meta[rec_id]["sample_count"] = len(dp)

            # Generate metadata for the capture
            self.create_channel_metadata(
                rec_id, iteration_params[rec_id], cap_meta[rec_id], dp_idx
            )

            # Increment start sample
            last_data_len = len(all_data)

        # Build metadata
        self.sigmf_builder.build()

        all_data = self.compress_bytes_data(np.array(all_data).tobytes())
        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=all_data,
            metadata=self.sigmf_builder.metadata,
        )
        action_done = perf_counter()
        logger.debug(
            f"IQ Capture and all data processing completed in {action_done-start_action:.2f}"
        )

    def capture_iq(self, params: dict) -> dict:
        """Acquire a single gap-free stream of IQ samples."""
        logger.debug(f"GC Count (IQ Cap): {gc.get_count()}")
        tic = perf_counter()
        gc.collect()  # Computationally expensive!
        toc = perf_counter()
        logger.debug(f"GC Count (IQ Cap) after collection: {gc.get_count()}")
        print(f"Collected garbage in {toc-tic:.2f} s")

        start_time = utils.get_datetime_str_now()
        tic = perf_counter()
        # Configure signal analyzer + preselector
        self.configure(params)
        # Ensure sample rate is accurately applied
        assert (
            self.sigan.sample_rate == params[SAMPLE_RATE]
        ), "Sample rate setting not applied."
        # Get IQ capture parameters
        duration_ms = utils.get_parameter(DURATION_MS, params)
        nskip = utils.get_parameter(NUM_SKIP, params)
        num_samples = int(params[SAMPLE_RATE] * duration_ms * 1e-3)
        # Collect IQ data
        measurement_result = self.sigan.acquire_time_domain_samples(num_samples, nskip)
        end_time = utils.get_datetime_str_now()
        # Store some metadata with the IQ
        measurement_result.update(params)
        measurement_result["name"] = self.name
        measurement_result["start_time"] = start_time
        measurement_result["end_time"] = end_time
        measurement_result["sigan_cal"] = self.sigan.sigan_calibration_data
        measurement_result["sensor_cal"] = self.sigan.sensor_calibration_data
        toc = perf_counter()
        logger.debug(
            f"IQ Capture ({duration_ms} ms @ {(params[FREQUENCY]/1e6):.1f} MHz) completed in {toc-tic:.2f} s."
        )
        return measurement_result

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            raise RuntimeError(msg)
        # TODO: Add additional health checks
        return None

    def create_channel_metadata(
        self,
        rec_id: int,
        params: dict,
        cap_meta: dict,
        dp_idx=None,
    ) -> SigMFBuilder:
        """Add metadata corresponding to a single-frequency capture in the measurement"""
        # Add Capture
        self.sigmf_builder.set_capture(
            params[FREQUENCY], cap_meta["start_time"], cap_meta["sample_start"]
        )

        # Remove some metadata from calibration annotation
        sensor_cal = {
            k: v
            for k, v in cap_meta["sensor_cal"].items()
            if k in {"gain_sensor", "noise_figure_sensor", "enbw_sensor", "temperature"}
        }

        # Calibration Annotation
        calibration_annotation = CalibrationAnnotation(
            sample_start=cap_meta["sample_start"],
            sample_count=cap_meta["sample_count"],
            sigan_cal=None,  # Do not include sigan cal data
            sensor_cal=sensor_cal,
        )
        self.sigmf_builder.add_metadata_generator(
            type(calibration_annotation).__name__ + f"_{rec_id}", calibration_annotation
        )

        # Sensor Annotation (contains overload indicator)
        sensor_annotation = SensorAnnotation(
            sample_start=cap_meta["sample_start"],
            sample_count=cap_meta["sample_count"],
            overload=cap_meta["overload"],
            attenuation_setting_sigan=params["attenuation"],
        )
        self.sigmf_builder.add_metadata_generator(
            type(sensor_annotation).__name__ + f"_{rec_id}", sensor_annotation
        )

        # FFT Annotation
        for i, detector in enumerate(FFT_DETECTOR):
            fft_annotation = FrequencyDomainDetection(
                sample_start=dp_idx[i] + cap_meta["sample_start"],
                sample_count=dp_idx[i + 1] - dp_idx[i],
                detector=detector.value,
                number_of_ffts=int(params[NUM_FFTS]),
                number_of_samples_in_fft=FFT_SIZE,  # TODO: This is hard-coded
                window=FFT_WINDOW_TYPE,  # TODO: This is hard-coded
                units="dBm/Hz",
                reference="preselector input",
            )
            self.sigmf_builder.add_metadata_generator(
                type(fft_annotation).__name__ + "_" + detector.value + f"_{rec_id}",
                fft_annotation,
            )

        # Time Domain Annotation
        for i, detector in enumerate(TD_DETECTOR):
            td_annotation = TimeDomainDetection(
                sample_start=dp_idx[i + 2] + cap_meta["sample_start"],
                sample_count=dp_idx[i + 3] - dp_idx[i + 2],
                detector=detector.value,
                number_of_samples=int(params[SAMPLE_RATE] * params[DURATION_MS] * 1e-3),
                units="dBm",
                reference="preselector input",
            )
            self.sigmf_builder.add_metadata_generator(
                type(td_annotation).__name__ + "_" + detector.value + f"_{rec_id}",
                td_annotation,
            )

        # dp_idx = [fft_mean, fft_max, td_mean, td_max, pfprms_min, pfp_rms]

        # PFP Annotation (custom, not in spec)
        for i, detector in enumerate(PFP_M3_DETECTOR):
            # RMS result M3 detected
            pfp_annotation = AnnotationSegment(
                sample_start=dp_idx[i + 4] + cap_meta["sample_start"],
                sample_count=dp_idx[i + 5] - dp_idx[i + 4],
                label="pfp_rms_" + detector.value,
            )
            self.sigmf_builder.add_metadata_generator(
                type(pfp_annotation).__name__ + "_rms_" + detector.value + f"_{rec_id}",
                pfp_annotation,
            )

        for i, detector in enumerate(PFP_M3_DETECTOR):
            # Peak result M3 detected
            pfp_annotation = AnnotationSegment(
                sample_start=dp_idx[i + 7] + cap_meta["sample_start"],
                sample_count=dp_idx[i + 8] - dp_idx[i + 7],
                label="pfp_peak_" + detector.value,
            )
            self.sigmf_builder.add_metadata_generator(
                type(pfp_annotation).__name__
                + "_peak_"
                + detector.value
                + f"_{rec_id}",
                pfp_annotation,
            )

        # APD Annotation
        apd_p_annotation = AnnotationSegment(
            sample_start=dp_idx[10] + cap_meta["sample_start"],
            sample_count=dp_idx[11] - dp_idx[10],
            label="apd_p_pct",
        )
        apd_a_annotation = AnnotationSegment(
            sample_start=dp_idx[11] + cap_meta["sample_start"],
            sample_count=cap_meta["sample_count"] - dp_idx[11],
            label="apd_a_dBm",
        )
        self.sigmf_builder.add_metadata_generator(
            type(apd_p_annotation).__name__ + f"_apd_p_{rec_id}",
            apd_p_annotation,
        )
        self.sigmf_builder.add_metadata_generator(
            type(apd_a_annotation).__name__ + f"_apd_a_{rec_id}",
            apd_a_annotation,
        )

    def get_sigmf_builder(
        self,
        sample_rate_Hz: float,
        task_id: int,
        schedule_entry: dict,
    ) -> SigMFBuilder:
        """Build SigMF that applies to the entire capture (all channels)"""
        sigmf_builder = SigMFBuilder()
        sigmf_builder.set_data_type(self.is_complex(), bit_width=16, endianness="")
        sigmf_builder.set_sample_rate(sample_rate_Hz)
        sigmf_builder.set_task(task_id)
        sigmf_builder.set_schedule(schedule_entry)
        sigmf_builder.set_last_calibration_time(
            self.sigan.sensor_calibration_data["calibration_datetime"]
        )  # TODO: this is approximate since each channel is individually calibrated
        self.sigmf_builder = sigmf_builder

    @staticmethod
    def transform_data(data_product: list):
        """Flatten data product list of arrays (single channel), convert to bytes object, then compress"""
        # Get indices for start of each component in flattened result
        data_lengths = [len(d) for d in data_product]
        logger.debug(f"Data product component lengths: {data_lengths}")
        idx = [0] + np.cumsum(data_lengths[:-1]).tolist()
        logger.debug(f"Data product start indices: {idx}")
        # Flatten data product, reduce dtype, and convert to byte array
        data_product = np.hstack(data_product).astype(DATA_TYPE)
        return data_product, np.array(idx)

    @staticmethod
    def compress_bytes_data(data: bytes) -> bytes:
        """Compress some <bytes> data and return the compressed version"""
        # TODO: Explore alternate compression methods.
        return lzma.compress(data)

    def is_complex(self) -> bool:
        return False

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        # TODO (low-priority)
        return __doc__
