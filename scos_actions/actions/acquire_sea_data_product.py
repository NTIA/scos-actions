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
from scos_actions.hardware import preselector, switches
from scos_actions.hardware.mocks.mock_gps import MockGPS
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
)
from scos_actions.signal_processing.unit_conversion import (
    convert_linear_to_dB,
    convert_watts_to_dBm,
)
from scos_actions.signals import measurement_action_completed

logger = logging.getLogger(__name__)

# Define parameter keys
RF_PATH = "rf_path"
IIR_APPLY = "iir_apply"
IIR_GPASS = "iir_gpass_dB"
IIR_GSTOP = "iir_gstop_dB"
IIR_PB_EDGE = "iir_pb_edge_Hz"
IIR_SB_EDGE = "iir_sb_edge_Hz"
IIR_RESP_FREQS = "iir_num_response_frequencies"
# FFT_SIZE = "fft_size"
NUM_FFTS = "nffts"
# FFT_WINDOW_TYPE = "fft_window_type"
APD_BIN_SIZE_DB = "apd_bin_size_dB"
TD_BIN_SIZE_MS = "td_bin_size_ms"
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
ATTENUATION = "attenuation"
PREAMP_ENABLE = "preamp_enable"
REFERENCE_LEVEL = "reference_level"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"
PFP_FRAME_PERIOD_MS = "pfp_frame_period_ms"

# Constants
DATA_TYPE = np.half
PFP_FRAME_RESOLUTION_S = (1e-3 * (1 + 1 / (14)) / 15) / 4


@ray.remote
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


@ray.remote
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


@ray.remote
def get_td_power_results(
    iqdata: np.ndarray, params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/max time domain power statistics from IQ samples."""
    # Reshape IQ data into blocks
    block_size = int(params[TD_BIN_SIZE_MS] * params[SAMPLE_RATE] * 1e-3)
    n_blocks = len(iqdata) // block_size
    iqdata = iqdata.reshape(block_size, n_blocks)
    print(
        f"Calculating time-domain power statistics on {n_blocks} blocks of {block_size} samples each"
    )

    iq_pwr = calculate_power_watts(iqdata, impedance_ohms=50.0)

    # Apply mean/max detectors
    td_result = apply_power_detector(iq_pwr, TD_DETECTOR)

    # Get single value mean/max statistics
    td_channel_result = np.array([td_result[0].max(), td_result[1].mean()])

    # Convert to dBm and account for RF/baseband power difference
    td_result, td_channel_result = (
        convert_watts_to_dBm(x) - 3.0 for x in [td_result, td_channel_result]
    )

    channel_max, channel_mean = (np.array(a) for a in td_channel_result)

    # packed order is (max, mean)
    return td_result[0], td_result[1], channel_max, channel_mean


@ray.remote
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

    iqdata = sosfilt(iir_sos, iqdata)
    toc = perf_counter()
    print(f"Applied IIR filter to IQ data @ {params[FREQUENCY]} in {toc-tic1:.2f} s")

    remote_procs = []
    remote_procs.append(get_fft_results.remote(iqdata, params))
    remote_procs.append(get_td_power_results.remote(iqdata, params))
    remote_procs.append(get_periodic_frame_power.remote(iqdata, params))
    remote_procs.append(get_apd_results.remote(iqdata, params))
    all_results = ray.get(remote_procs)

    for dp in all_results:
        data_product.extend(dp)

    # tic = perf_counter()
    # data_product.extend(get_fft_results(iqdata, params))
    # toc = perf_counter()
    # print(f"Got FFT result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    # tic = perf_counter()
    # # TODO: Single value results currently don't go anywhere
    # td_result, td_channel_powers = get_td_power_results(iqdata, params)
    # data_product.extend(td_result)
    # toc = perf_counter()
    # print(f"Got TD result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    # tic = perf_counter()
    # data_product.extend(get_periodic_frame_power(iqdata, params))
    # toc = perf_counter()
    # print(f"Got PFP result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    # tic = perf_counter()
    # data_product.extend(get_apd_results(iqdata, params))
    # toc = perf_counter()
    # print(f"Got APD result @ {params[FREQUENCY]} in {toc-tic:.2f} s")

    print(f"Got all data product @ {params[FREQUENCY]} results in {toc-tic1:.2f} s")

    del iqdata
    gc.collect()

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

    def __init__(self, parameters, sigan, gps=None):
        if gps is None:
            gps = MockGPS()
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
        ]:
            self.parameters.pop(key)

    def __call__(self, schedule_entry, task_id):
        """This is the entrypoint function called by the scheduler."""
        self.test_required_components()

        iteration_params = utils.get_iterable_parameters(self.parameters)

        start_action = perf_counter()
        self.configure_preselector(self.rf_path)

        # Collect all IQ data and spawn data product computation processes
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
        logger.debug("**********\nLOOPING CAPTURES\n*************")
        for i, (dp, dp_idx) in enumerate(results):
            all_data.extend(dp)
            all_idx.extend((dp_idx + last_data_len).tolist())

            logger.debug(f"Loop {i}: idx: {dp_idx}, length: {len(dp)}")

            cap_meta[i]["sample_start"] = last_data_len
            cap_meta[i]["sample_count"] = len(dp)

            # Generate metadata for the capture
            self.create_channel_metadata(iteration_params[i], cap_meta[i], dp_idx)

            # Increment start sample
            last_data_len = len(all_data)

        # Add sensor readouts to metadata
        self.capture_sensors()

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
        # Downselect params to suppress logger warnings
        hw_params = {
            k: params[k]
            for k in [
                RF_PATH,
                ATTENUATION,
                PREAMP_ENABLE,
                REFERENCE_LEVEL,
                SAMPLE_RATE,
                FREQUENCY,
            ]
        }
        start_time = utils.get_datetime_str_now()
        tic = perf_counter()
        # Configure signal analyzer + preselector
        self.configure(hw_params)
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

    def capture_sensors(self) -> dict:
        """
        Read values from web relay sensors.

        This method pulls the following values from the following
        web relays:

        PRESELECTOR X410
        LNA Temperature (float)
        Noise Diode Temperature (float)
        Internal PS Temperature (float)
        Internal PS Humidity (float)
        PS Door Sensor (bool)

        SPU X410
        SPU RF Tray Power (bool)
        PS Power (bool)
        Aux 28 VDC (bool)
        RF box temp (float)
        Power/control box temp (float)
        power/control box humidity (float)
        """

        """
        All temperature sensors must be set to degrees F.
        SPU X410 config file must have "SPU X410" in the name field.


        Preselector X410 Setup requires:
        internal temperature : oneWireSensor 1
        noise diode temp : oneWireSensor 2
        LNA temp : oneWireSensor 3
        internal humidity : oneWireSensor 4
        door sensor: digitalInput 1

        SPU X410 requires:
        Power/Control Box Temperature: oneWireSensor 1
        RF Box Temperature: oneWireSensor 2
        Power/Control Box Humidity: oneWireSensor 3
        """
        # Get SPU x410 sensor values and status:
        logger.debug("*********************************\n\n")
        logger.debug(f"SWITCHES: {switches}")
        logger.debug("*********************************\n\n")
        for base_url, switch in switches.items():
            logger.debug(f"Iterating on switch: {switch.name}")
            if switch.name == "SPU X410":
                spu_x410_sensor_values = switch.get_status()
                del spu_x410_sensor_values["name"]
                del spu_x410_sensor_values["healthy"]
                spu_x410_sensor_values["pwr_box_temp_degF"] = switch.get_sensor_value(1)
                spu_x410_sensor_values["rf_box_temp_degF"] = switch.get_sensor_value(2)
                spu_x410_sensor_values[
                    "pwr_box_humidity_pct"
                ] = switch.get_sensor_value(3)

        # Read preselector
        preselector_sensor_values = {
            "internal_temp_degF": preselector.get_sensor_value(1),
            "noise_diode_temp_degF": preselector.get_sensor_value(2),
            "lna_temp_degF": preselector.get_sensor_value(3),
            "internal_humidity_pct": preselector.get_sensor_value(4),
            "door_closed": preselector.get_digital_input_value(1),
        }

        all_sensor_values = {
            "preselector": preselector_sensor_values,
            "spu_x410": spu_x410_sensor_values,
        }

        logger.debug(f"Sensor readout dict: {all_sensor_values}")

        # Make AnnotationSegment from sensor data
        self.sigmf_builder.add_annotation(0, 2, all_sensor_values)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            raise RuntimeError(msg)
        # TODO: Add additional health checks
        return None

    def create_channel_metadata(
        self,
        params: dict,
        cap_meta: dict,
        dp_idx=None,
    ) -> SigMFBuilder:
        """Add metadata corresponding to a single-frequency capture in the measurement"""
        # Construct dict of extra info to attach to capture
        capture_dict = {
            "overload": cap_meta["overload"],
            "sigan_attenuation_dB": params[ATTENUATION],
            "sigan_preamp_on": params[PREAMP_ENABLE],
            "sigan_reference_level_dBm": params[REFERENCE_LEVEL],
            "cal_noise_figure_dB": cap_meta["sensor_cal"]["noise_figure_sensor"],
            "cal_gain_dB": cap_meta["sensor_cal"]["gain_sensor"],
            "cal_temperature_degC": cap_meta["sensor_cal"]["temperature"],
            "fft_sample_count": dp_idx[1] - dp_idx[0],  # Should be 625
            "td_pwr_sample_count": dp_idx[4] - dp_idx[3],  # Should be 400
            "pfp_sample_count": dp_idx[5] - dp_idx[4],  # Should be 560
            "apd_sample_count": dp_idx[11] - dp_idx[10],  # Variable!
        }

        ordered_data_components = [
            "max_fft",
            "mean_fft",
            "max_td_pwr_series",
            "mean_td_pwr_series",
            "max_td_pwr",  # These are single value channel power results
            "mean_td_pwr",  # which should probably be stored in metadata instead
            "min_rms_pfp",
            "max_rms_pfp",
            "mean_rms_pfp",
            "min_peak_pfp",
            "max_peak_pfp",
            "mean_peak_pfp",
            "apd_p",
            "apd_a",
        ]
        for i, dc in zip(dp_idx, ordered_data_components):
            capture_dict.update({dc + "_sample_start": i + cap_meta["sample_start"]})

        # Add Capture
        self.sigmf_builder.set_capture(
            params[FREQUENCY],
            cap_meta["start_time"],
            cap_meta["sample_start"],
            extra_entries=capture_dict,
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
        data_lengths = [d.size for d in data_product]
        logger.debug(f"Data product component lengths: {data_lengths}")
        idx = [0] + np.cumsum(data_lengths[:-1]).tolist()
        logger.debug(f"Data product start indices: {idx}")
        # Flatten data product, reduce dtype, and convert to byte array
        data_product = np.hstack(data_product).astype(DATA_TYPE)
        return data_product, np.array(idx)

    @staticmethod
    def compress_bytes_data(data: bytes) -> bytes:
        """Compress some <bytes> data and return the compressed version"""
        return lzma.compress(data)

    def is_complex(self) -> bool:
        return False

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        # TODO (low-priority)
        return __doc__
