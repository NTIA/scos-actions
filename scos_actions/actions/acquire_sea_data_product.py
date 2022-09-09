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
import json
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
from scos_actions.metadata.annotations import (
    CalibrationAnnotation,
    FrequencyDomainDetection,
    SensorAnnotation,
    TimeDomainDetection,
)
from scos_actions.metadata.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.signal_processing.apd import get_apd
from scos_actions.signal_processing.fft import (
    get_fft,
    get_fft_enbw,
    get_fft_frequencies,
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
# Q_LO = "qfilt_qlo"
# Q_HI = "qfilt_qhi"
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


# DSP tasks to parallelize
# ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/dev/null/"}}
        )
    },
)


@ray.remote
def iir_filter(iir_sos: np.ndarray, iqdata: np.ndarray) -> np.ndarray:
    """Apply sosfilt"""
    return sosfilt(iir_sos, iqdata)


@ray.remote
def get_fft_results(
    iqdata: np.ndarray, params: dict, logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute data product mean/max FFT results from IQ samples."""
    # IQ data already scaled for calibrated gain
    fft_result = get_fft(
        time_data=iqdata,
        fft_size=FFT_SIZE,
        norm="backward",
        fft_window=FFT_WINDOW,
        num_ffts=params[NUM_FFTS],
        shift=False,
        workers=1,  # TODO: Configure for parallelization
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

    # Truncate FFT result
    # TODO These parameters can be hardcoded
    logger.debug(f"Pre-truncated FFT result shape: {fft_result.shape}")
    bw_trim = (params[SAMPLE_RATE] / 1.4) / 5
    delta_f = params[SAMPLE_RATE] / FFT_SIZE
    bin_start = int(bw_trim / delta_f)
    bin_end = FFT_SIZE - bin_start
    fft_result = fft_result[:, bin_start:bin_end]
    logger.debug(f"Truncated FFT result length: {fft_result.shape}")

    # Reduce data type
    # fft_result = NasctnSeaDataProduct.reduce_dtype(fft_result)

    # Get FFT metadata for annotation: ENBW, frequency axis
    fft_freqs_Hz = get_fft_frequencies(FFT_SIZE, params[SAMPLE_RATE], params[FREQUENCY])
    fft_freq_start = fft_freqs_Hz[bin_start]
    fft_freq_stop = fft_freqs_Hz[bin_end - 1]
    fft_freq_step = fft_freqs_Hz[1] - fft_freqs_Hz[0]
    fft_enbw = get_fft_enbw(FFT_WINDOW, params[SAMPLE_RATE])
    fft_meta = [fft_freq_start, fft_freq_stop, fft_freq_step, fft_enbw]

    del fft_freqs_Hz

    return (fft_result[0], fft_result[1]), fft_meta


@ray.remote
def get_apd_results(
    iqdata: np.ndarray, params: dict, logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate downsampled APD result from IQ samples."""
    p, a = get_apd(iqdata, params[APD_BIN_SIZE_DB])
    # Convert dBV to dBm:
    # a = a * 2 : dBV --> dB(V^2)
    # a = a - impedance_dB : dB(V^2) --> dBW
    # a = a + 27 : dBW --> dBm (+30) and RF/baseband conversion (-3)
    scale_factor = 27 - convert_linear_to_dB(50.0)  # Hard-coded for 50 Ohms.
    ne.evaluate("(a*2)+scale_factor", out=a)
    # p, a = (NasctnSeaDataProduct.reduce_dtype(x) for x in (p, a))
    return p, a


@ray.remote
def get_td_power_results(
    iqdata: np.ndarray, params: dict, logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/max time domain power statistics from IQ samples, with optional quantile filtering."""
    # Reshape IQ data into blocks
    block_size = int(params[TD_BIN_SIZE_MS] * params[SAMPLE_RATE] * 1e-3)
    n_blocks = len(iqdata) // block_size
    iqdata = iqdata.reshape(block_size, n_blocks)
    logger.debug(
        f"Calculating time-domain power statistics on {n_blocks} blocks of {block_size} samples each"
    )

    iq_pwr = calculate_power_watts(iqdata, impedance_ohms=50.0)

    if params[QFILT_APPLY]:
        # Apply quantile filtering before computing power statistics
        logger.info("Quantile-filtering time domain power data...")
        iq_pwr = filter_quantiles(iq_pwr, QFILT_QLO, QFILT_QHI)
        # Diagnostics
        num_nans = np.count_nonzero(np.isnan(iq_pwr))
        nan_pct = num_nans * 100 / len(iq_pwr.flatten())
        logger.debug(f"Rejected {num_nans} samples ({nan_pct:.2f}% of total capture)")
    else:
        logger.info("Quantile-filtering disabled. Skipping...")

    # Apply mean/max detectors
    td_result = apply_power_detector(iq_pwr, TD_DETECTOR, ignore_nan=True)

    # Convert to dBm
    td_result = convert_watts_to_dBm(td_result)

    # Account for RF/baseband power difference
    td_result -= 3

    # Reduce data type
    # td_result = NasctnSeaDataProduct.reduce_dtype(td_result)

    return td_result[0], td_result[1]  # (max, mean)


@ray.remote
def get_periodic_frame_power(
    iqdata: np.ndarray,
    params: dict,
    logger: logging.Logger,
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
    logger.debug(f"PFP Nframes: {Nframes}, Npts: {Npts}")

    # set up dimensions to make the statistics fast
    chunked_shape = (iqdata.shape[0] // Nframes, Npts, Nframes // Npts) + tuple(
        [iqdata.shape[1]] if iqdata.ndim == 2 else []
    )
    iq_bins = iqdata.reshape(chunked_shape)
    power_bins = calculate_pseudo_power(iq_bins)

    # compute statistics first by cycle
    rms_power = power_bins.mean(axis=0)
    peak_power = power_bins.max(axis=0)

    # Finish conversion to power
    ne.evaluate("rms_power/50", out=rms_power)
    ne.evaluate("peak_power/50", out=peak_power)

    # then do the detector
    pfp = np.array(
        [
            # RMS
            rms_power.min(axis=1),
            rms_power.mean(axis=1),
            rms_power.max(axis=1),
            # Peak
            peak_power.min(axis=1),
            peak_power.mean(axis=1),
            peak_power.max(axis=1),
        ]
    )

    # Convert to dBm
    pfp = convert_watts_to_dBm(pfp)
    pfp -= 3  # RF/baseband
    # pfp = NasctnSeaDataProduct.reduce_dtype(pfp)
    logger.debug(f"PFP result shape: {pfp.shape}")
    return tuple(pfp)


# Hard-coded algorithm parameters
QFILT_QLO = 0.00015
QFILT_QHI = 0.99999
FFT_SIZE = 875
FFT_WINDOW_TYPE = "flattop"

# Generate FFT window and correction factor
FFT_WINDOW = get_fft_window(FFT_WINDOW_TYPE, FFT_SIZE)
FFT_WINDOW_ECF = get_fft_window_correction(FFT_WINDOW, "energy")

# Create power detectors
TD_DETECTOR = create_power_detector("TdMeanMaxDetector", ["mean", "max"])
FFT_DETECTOR = create_power_detector("FftMeanMaxDetector", ["mean", "max"])


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

        # Temporary: remove config parameters which will be hard-coded eventually
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

        # TODO:
        # For now, this iterates (capture IQ -> process data product) for each
        # configured frequency. It is probably better to do all IQ captures first,
        # then generate all data products, or to parallelize captures/processing.

        start_action = perf_counter()
        logger.debug(f"Setting RF path to {self.rf_path}")
        self.configure_preselector(self.rf_path)
        for recording_id, parameters in enumerate(iteration_params, start=1):
            # Capture IQ data
            measurement_result = self.capture_iq(task_id, parameters)

            # Generate data product (overwrites IQ data in measurement_result)
            measurement_result, dp_idx = self.generate_data_product(
                measurement_result, parameters
            )

            # Generate metadata
            sigmf_builder = self.get_sigmf_builder(measurement_result, dp_idx)
            self.create_metadata(
                sigmf_builder, schedule_entry, measurement_result, recording_id
            )

            # Send signal
            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=measurement_result["data"],
                metadata=sigmf_builder.metadata,
            )
        action_done = perf_counter()
        logger.debug(
            f"IQ Capture and data processing completed in {action_done-start_action:.2f}"
        )

    def capture_iq(self, task_id, params) -> dict:
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
        measurement_result["domain"] = Domain.TIME.value
        measurement_result["measurement_type"] = MeasurementType.SINGLE_FREQUENCY.value
        measurement_result["task_id"] = task_id
        measurement_result["calibration_datetime"] = self.sigan.sensor_calibration_data[
            "calibration_datetime"
        ]
        measurement_result["description"] = self.description
        measurement_result["sigan_cal"] = self.sigan.sigan_calibration_data
        measurement_result["sensor_cal"] = self.sigan.sensor_calibration_data
        toc = perf_counter()
        logger.debug(f"IQ Capture ({duration_ms} ms) completed in {toc-tic:.2f} s.")
        return measurement_result

    def generate_data_product(
        self, measurement_result: dict, params: dict
    ) -> np.ndarray:
        """Process IQ data and generate the SEA data product."""
        logger.debug(f'Generating data product for {measurement_result["task_id"]}')

        logger.debug(f"Starting FFT process")  # Other tasks may proceed
        fft_ray = get_fft_results.remote(measurement_result["data"], params, logger)

        logger.debug(f"Applying IIR filter")
        iq = iir_filter.remote(self.iir_sos, measurement_result["data"])

        # Processes won't start until IIR filtering finishes
        logger.debug("Starting APD, PFP, TDPWR processes")
        tic = perf_counter()
        apd_ray = get_apd_results.remote(iq, params, logger)
        pfp_ray = get_periodic_frame_power.remote(iq, params, logger)
        td_ray = get_td_power_results.remote(iq, params, logger)
        toc = perf_counter()
        logger.debug(f"Processes started in {toc-tic:.2f} s")

        # Get process results and construct data product
        # fft_data, fft_meta = ray.get(fft_ray)
        # td_data = ray.get(td_ray)
        # pfp_data = ray.get(pfp_ray)
        # apd_data = ray.get(apd_ray)
        fft_data, td_data, pfp_data, apd_data = (
            ray.get(p) for p in [fft_ray, td_ray, pfp_ray, apd_ray]
        )
        tic = perf_counter()
        logger.debug(f"Got all results {tic-toc:.2f} s after all processes started")
        tic = perf_counter()
        del iq
        toc = perf_counter()
        logger.debug(f"Deleted IQ data in {toc-tic:.2f} s")

        # Construct single data product result
        tic = perf_counter()
        data_product = [
            fft_data[0][0],  # Mean FFT amplitudes
            fft_data[0][1],  # Max FFT amplitudes
            td_data[0],  # Mean TD power
            td_data[1],  # Max TD power
            pfp_data[0],  # Min RMS PFP
            pfp_data[1],  # Mean RMS PFP
            pfp_data[2],  # Max RMS PFP
            pfp_data[3],  # Min Peak PFP
            pfp_data[4],  # Mean Peak PFP
            pfp_data[5],  # Max Peak PFP
            apd_data[0],  # APD probabilities
            apd_data[1],  # APD amplitudes
        ]
        toc = perf_counter()
        logger.debug(f"Combined all results in {toc-tic:.2f} s")

        # Save FFT metadata to measurement_result
        tic = perf_counter()
        measurement_result["fft_frequency_start"] = fft_data[1][0]
        measurement_result["fft_frequency_stop"] = fft_data[1][1]
        measurement_result["fft_frequency_step"] = fft_data[1][2]
        measurement_result["fft_enbw"] = fft_data[1][3]
        toc = perf_counter()
        logger.debug(f"Saved FFT metadata in {toc-tic:.2f} s")

        # Get FFT amplitudes using unfiltered data
        # logger.debug("Getting FFT results...")
        # tic = perf_counter()
        # fft_results, measurement_result = self.get_fft_results(
        #     measurement_result, params
        # )
        # data_product.extend(fft_results)  # (max, mean)
        # toc = perf_counter()
        # logger.debug(f"FFT computation complete in {toc-tic:.2f} s")

        # Filter IQ data
        # if params[IIR_APPLY]:
        #     logger.debug(f"Applying IIR low-pass filter to IQ data...")
        #     tic = perf_counter()
        #     iq = sosfilt(self.iir_sos, measurement_result["data"])
        #     toc = perf_counter()
        #     logger.debug(f"IIR filter applied to IQ samples in {toc-tic:.2f} s")
        # else:
        #     logger.debug(f"Skipping IIR filtering of IQ data...")

        # logger.debug("Calculating time-domain power statistics...")
        # tic = perf_counter()
        # data_product.extend(self.get_td_power_results(iq, params))  # (max, mean)
        # toc = perf_counter()
        # logger.debug(f"Time domain power statistics calculated in {toc-tic:.2f} s")

        # logger.debug("Computing periodic frame power...")
        # tic = perf_counter()
        # data_product.extend(self.get_periodic_frame_power(iq, params))
        # toc = perf_counter()
        # logger.debug(f"Periodic frame power computed in {toc-tic:.2f} s")

        # logger.debug("Generating APD...")
        # tic = perf_counter()
        # data_product.extend(self.get_apd_results(iq, params))
        # toc = perf_counter()
        # logger.debug(f"APD result generated in {toc-tic:.2f} s")

        # del iq

        # TODO: Optimize memory usage
        logger.debug(f"GC Count: {gc.get_count()}")
        tic = perf_counter()
        gc.collect()  # Computationally expensive!
        toc = perf_counter()
        logger.debug(f"GC Count after collection: {gc.get_count()}")
        print(f"Collected garbage in {toc-tic:.2f} s")

        # Skip rounding for now
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

        # Flatten and compress data product
        measurement_result["data"] = data_product
        tic = perf_counter()
        measurement_result, dp_idx = self.transform_data(measurement_result)
        toc = perf_counter()
        print(f"Data transformed in {toc-tic:.2f} s")

        return measurement_result, dp_idx

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            raise RuntimeError(msg)
        # TODO: Add additional health checks
        return None

    def create_metadata(
        self,
        sigmf_builder: SigMFBuilder,
        schedule_entry: dict,
        measurement_result: dict,
        recording=None,
    ):
        sigmf_builder.set_last_calibration_time(
            measurement_result["calibration_datetime"]
        )
        sigmf_builder.set_data_type(self.is_complex(), bit_width=16, endianness="")
        sigmf_builder.set_sample_rate(measurement_result["sample_rate"])
        sigmf_builder.set_schedule(schedule_entry)
        sigmf_builder.set_task(measurement_result["task_id"])
        sigmf_builder.set_recording(recording)
        sigmf_builder.add_sigmf_capture(sigmf_builder, measurement_result)
        sigmf_builder.build()

    def get_sigmf_builder(self, measurement_result: dict, dp_idx: list) -> SigMFBuilder:
        # TODO: Finalize metadata
        # Create metadata annotations for the data
        sigmf_builder = SigMFBuilder()

        # Remove unnecessary metadata from calibration annotation
        sensor_cal = {
            k: v
            for k, v in measurement_result["sensor_cal"].items()
            if k in {"gain_sensor", "noise_figure_sensor", "enbw_sensor", "temperature"}
        }

        # Annotate calibration
        calibration_annotation = CalibrationAnnotation(
            sample_start=0,
            sample_count=self.total_samples,  # Set when transform_data is called
            sigan_cal=None,
            sensor_cal=sensor_cal,
        )
        sigmf_builder.add_metadata_generator(
            type(calibration_annotation).__name__, calibration_annotation
        )

        # Annotate sensor settings
        sensor_annotation = SensorAnnotation(
            sample_start=0,
            sample_count=self.total_samples,
            overload=measurement_result["overload"],
            attenuation_setting_sigan=self.parameters["attenuation"],
        )
        sigmf_builder.add_metadata_generator(
            type(sensor_annotation).__name__, sensor_annotation
        )

        # Annotate FFT
        for i, detector in enumerate(FFT_DETECTOR):
            fft_annotation = FrequencyDomainDetection(
                sample_start=dp_idx[i],
                sample_count=dp_idx[i + 1] - dp_idx[i],
                detector=detector.value,
                number_of_ffts=int(measurement_result[NUM_FFTS]),
                number_of_samples_in_fft=FFT_SIZE,
                window=FFT_WINDOW_TYPE,
                equivalent_noise_bandwidth=measurement_result["fft_enbw"],
                units="dBm/Hz",
                reference="preselector input",
                frequency_start=measurement_result["fft_frequency_start"],
                frequency_stop=measurement_result["fft_frequency_stop"],
                frequency_step=measurement_result["fft_frequency_step"],
            )
            sigmf_builder.add_metadata_generator(
                type(fft_annotation).__name__ + "_" + detector.value, fft_annotation
            )

        # Annotate time domain power statistics
        for i, detector in enumerate(TD_DETECTOR):
            td_annotation = TimeDomainDetection(
                sample_start=dp_idx[i + 2],
                sample_count=dp_idx[i + 3] - dp_idx[i + 2],
                detector=detector.value,
                number_of_samples=int(
                    measurement_result[SAMPLE_RATE]
                    * measurement_result[DURATION_MS]
                    * 1e-3
                ),
                units="dBm",
                reference="preselector input",
            )
            sigmf_builder.add_metadata_generator(
                type(td_annotation).__name__ + "_" + detector.value, td_annotation
            )

        # TODO: Annotate APD + PFP

        return sigmf_builder

    @staticmethod
    def reduce_dtype(data_array: np.ndarray, data_type=DATA_TYPE) -> np.ndarray:
        return data_array.astype(data_type)

    def transform_data(self, measurement_result: dict):
        """Flatten data product list of arrays, convert to bytes object, then compress"""
        # Get indices for start of each component in flattened result
        data_lengths = [len(d) for d in measurement_result["data"]]
        logger.debug(f"Data product component lengths: {data_lengths}")
        idx = [0] + np.cumsum(data_lengths[:-1]).tolist()
        logger.debug(f"Data product start indices: {idx}")
        # Flatten data product, reduce dtype, and convert to byte array
        measurement_result["data"] = np.hstack(measurement_result["data"]).astype(
            DATA_TYPE
        )
        self.total_samples = len(measurement_result["data"])
        measurement_result["data"] = measurement_result["data"].tobytes()
        measurement_result["data"] = self.compress_bytes_data(
            measurement_result["data"]
        )
        return measurement_result, idx

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
