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

import numpy as np
import psutil
import ray
from scipy.signal import sos2tf, sosfilt

from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
from scos_actions.capabilities import SENSOR_DEFINITION_HASH
from scos_actions.hardware import preselector, switches
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.utils import (
    get_cpu_uptime_seconds,
    get_current_cpu_clock_speed,
    get_current_cpu_temperature,
    get_disk_smart_data,
    get_max_cpu_temperature,
)
from scos_actions.metadata.sigmf_builder import SigMFBuilder
from scos_actions.signal_processing.apd import get_apd
from scos_actions.signal_processing.fft import (
    get_fft,
    get_fft_enbw,
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
from scos_actions.signals import measurement_action_completed, trigger_api_restart
from scos_actions.status import start_time
from scos_actions.utils import convert_datetime_to_millisecond_iso_format, get_days_up

logger = logging.getLogger(__name__)

if not ray.is_initialized():
    ray.init(
        include_dashboard=False,
    )

# Define parameter keys
RF_PATH = "rf_path"
IIR_GPASS = "iir_gpass_dB"
IIR_GSTOP = "iir_gstop_dB"
IIR_PB_EDGE = "iir_pb_edge_Hz"
IIR_SB_EDGE = "iir_sb_edge_Hz"
NUM_FFTS = "nffts"
APD_BIN_SIZE_DB = "apd_bin_size_dB"
APD_MIN_BIN_DBM = "apd_min_bin_dBm"
APD_MAX_BIN_DBM = "apd_max_bin_dBm"
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
FFT_SIZE = 875
FFT_WINDOW_TYPE = "flattop"
FFT_WINDOW = get_fft_window(FFT_WINDOW_TYPE, FFT_SIZE)
FFT_WINDOW_ECF = get_fft_window_correction(FFT_WINDOW, "energy")
IMPEDANCE_OHMS = 50.0

# Create power detectors
TD_DETECTOR = create_power_detector("TdMeanMaxDetector", ["mean", "max"])
FFT_DETECTOR = create_power_detector("FftMeanMaxDetector", ["mean", "max"])
PFP_M3_DETECTOR = create_power_detector("PfpM3Detector", ["min", "max", "mean"])

# Expected webswitch configuration:
PRESELECTOR_SENSORS = {
    "temp": 1,  # Internal temperature, deg C
    "noise_diode_temp": 2,  # Noise diode temperature, deg C
    "lna_temp": 3,  # LNA temperature, deg C
    "humidity": 4,  # Internal humidity, percentage
}
PRESELECTOR_DIGITAL_INPUTS = {"door_closed": 1}
SPU_SENSORS = {
    "pwr_box_temp": 1,  # Power tray temperature, deg C
    "rf_box_temp": 2,  # RF tray temperature, deg C
    "pwr_box_humidity": 3,  # Power tray humidity, percentage
}


@ray.remote
def get_fft_results(iqdata: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute data product mean/max FFT results from IQ samples.

    The result is a PSD, with amplitudes in dBm/Hz referenced to the
    calibration terminal. The result is truncated in frequency to the
    middle 10 MHz (middle 625 out of 875 total DFT samples). Some parts
    of this function are hard-coded or depend on constants.

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include
        `NUM_FFTS` and `SAMPLE_RATE` keys.
    :return: A tuple, which contains 2 NumPy arrays of power-detected
        PSD amplitudes, ordered (max, mean).
    """
    # IQ data already scaled for calibrated gain
    fft_result = get_fft(
        time_data=iqdata,
        fft_size=FFT_SIZE,
        norm="backward",
        fft_window=FFT_WINDOW,
        num_ffts=params[NUM_FFTS],
        shift=False,
        workers=1,
    )
    fft_result = calculate_pseudo_power(fft_result)
    fft_result = apply_power_detector(fft_result, FFT_DETECTOR)  # (max, mean)
    fft_result /= IMPEDANCE_OHMS  # Finish conversion to Watts
    fft_result = np.fft.fftshift(fft_result, axes=(1,))  # Shift frequencies
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
def get_apd_results(iqdata: np.ndarray, params: dict) -> np.ndarray:
    """
    Generate downsampled APD result from IQ samples.

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include
        ``APD_BIN_SIZE_DB``, ``APD_MIN_BIN_DBM``, and ``APD_MAX_BIN_DBM`` keys.
    :return: A NumPy array containing the APD probability axis as percentages.
    """
    # get_apd requires amplitude bin edge values in dBW here
    # Scale input to get_apd to account for:
    #     dBm -> dBW (-30)
    #     baseband -> RF power reference (+3)
    p, a = get_apd(
        iqdata,
        params[APD_BIN_SIZE_DB],
        params[APD_MIN_BIN_DBM] - 27.0,
        params[APD_MAX_BIN_DBM] - 27.0,
        IMPEDANCE_OHMS,
    )
    return p


@ray.remote
def get_td_power_results(
    iqdata: np.ndarray, params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean/max time domain power statistics from IQ samples.

    Mean and max detectors are applied over a configurable time window.
    Median and max power statistics are also computed on the detector
    results, and reported as single values. The median is computed on
    the mean detector result, and the max is computed on the max result.

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include
        `TD_BIN_SIZE_MS` and `SAMPLE_RATE` keys.
    :return: A tuple of NumPy arrays containing power detector results,
        in dBm referenced to the calibration terminal. The order of the
        results is (max, mean, max_max_single_value, median_mean_single_value).
        The single_value results will always contain only a single number,
        while the length of the other arrays depends on the configured
        detector period.
    """
    # Reshape IQ data into blocks
    block_size = int(params[TD_BIN_SIZE_MS] * params[SAMPLE_RATE] * 1e-3)
    n_blocks = len(iqdata) // block_size
    iqdata = iqdata.reshape((n_blocks, block_size))
    iq_pwr = calculate_power_watts(iqdata, IMPEDANCE_OHMS)

    # Apply max/mean detectors
    td_result = apply_power_detector(iq_pwr, TD_DETECTOR, axis=1)

    # Get single value median/max statistics
    td_channel_result = np.array([np.max(td_result[0]), np.median(td_result[1])])

    # Convert to dBm and account for RF/baseband power difference
    td_result, td_channel_result = (
        convert_watts_to_dBm(x) - 3.0 for x in [td_result, td_channel_result]
    )

    channel_max, channel_median = (np.array(a) for a in td_channel_result)

    # packed order of td_result is (max, mean)
    return td_result[0], td_result[1], channel_max, channel_median


@ray.remote
def get_periodic_frame_power(
    iqdata: np.ndarray,
    params: dict,
    detector_period_s: float = PFP_FRAME_RESOLUTION_S,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a time series of periodic frame power statistics.

    The time axis on the frame time elapsed spans [0, frame_period) binned with step size
    `detector_period`, for a total of `int(frame_period/detector_period)` samples.
    Mean and max power detector data are returned. For each type of detector, a time
    series is returned for (min, mean, max) statistics, which are computed across the
    number of frames (`frame_period/Ts`).

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include `SAMPLE_RATE` and
        `PFP_FRAME_PERIOD_MS` keys.
    :param detector_period_s: Sampling period (s) within the frame.
    :return: A tuple of 6 NumPy arrays for the 6 detector/statistic results: (mean_min,
        mean_max, mean_mean, max_min, max_max, max_mean).
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

    Nframes = int(round(frame_period_s / sampling_period_s))
    Npts = int(round(frame_period_s / detector_period_s))

    # set up dimensions to make the statistics fast
    chunked_shape = (iqdata.shape[0] // Nframes, Npts, Nframes // Npts) + tuple(
        [iqdata.shape[1]] if iqdata.ndim == 2 else []
    )
    iq_bins = iqdata.reshape(chunked_shape)
    power_bins = calculate_pseudo_power(iq_bins)

    # compute statistics first by cycle
    mean_power = power_bins.mean(axis=0)
    max_power = power_bins.max(axis=0)

    # then do the detector
    pfp = np.array(
        [
            apply_power_detector(p, PFP_M3_DETECTOR, axis=1)
            for p in [mean_power, max_power]
        ]
    ).reshape(6, Npts)

    # Finish conversion to power
    pfp /= IMPEDANCE_OHMS

    # Convert to dBm
    pfp = convert_watts_to_dBm(pfp)
    pfp -= 3  # RF/baseband
    return tuple(pfp)


@ray.remote
def generate_data_product(
    iqdata: np.ndarray, params: dict, iir_sos: np.ndarray
) -> list:
    """Process IQ data and generate the SEA data product."""
    iqdata = sosfilt(iir_sos, iqdata)

    # Explicitly call ray.put to pass filtered IQ to nested remote procs
    iqdata_id = ray.put(iqdata)
    del iqdata

    remote_procs = [
        get_fft_results.remote(iqdata_id, params),
        get_td_power_results.remote(iqdata_id, params),
        get_periodic_frame_power.remote(iqdata_id, params),
        get_apd_results.remote(iqdata_id, params),
    ]

    # Return identifiers to avoid waiting for processing to complete
    return remote_procs


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

        # Check that certain parameters are specified only once (not per-channel)
        assert not any(
            isinstance(self.parameters[k], list)
            for k in (
                IIR_GPASS,
                IIR_GSTOP,
                IIR_PB_EDGE,
                IIR_SB_EDGE,
                NUM_FFTS,
                SAMPLE_RATE,
                DURATION_MS,
                TD_BIN_SIZE_MS,
                PFP_FRAME_PERIOD_MS,
                APD_BIN_SIZE_DB,
                APD_MAX_BIN_DBM,
                APD_MIN_BIN_DBM,
            )
        )

        # Setup/pull config parameters
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
        self.iir_numerators, self.iir_denominators = sos2tf(self.iir_sos)

        # Remove IIR parameters which aren't needed after filter generation
        for key in [
            IIR_GPASS,
            IIR_GSTOP,
            IIR_PB_EDGE,
            IIR_SB_EDGE,
        ]:
            self.parameters.pop(key)

    def __call__(self, schedule_entry, task_id):
        """This is the entrypoint function called by the scheduler."""
        action_start_tic = perf_counter()

        _ = psutil.cpu_percent(interval=None)  # Initialize CPU usage monitor
        self.test_required_components()
        iteration_params = utils.get_iterable_parameters(self.parameters)
        self.configure_preselector(self.rf_path)

        # Initialize metadata object
        self.get_sigmf_builder(
            # Assumes the sigan correctly uses the configured sample rate.
            iteration_params[0][SAMPLE_RATE],
            task_id,
            schedule_entry,
            iteration_params,
        )
        self.create_global_data_product_metadata(self.parameters)

        # Collect all IQ data and spawn data product computation processes
        dp_procs, cap_meta, cap_entries, cpu_speed = [], [], [], []
        capture_tic = perf_counter()
        for parameters in iteration_params:
            measurement_result = self.capture_iq(parameters)
            # Start data product processing but do not block next IQ capture
            dp_procs.append(
                generate_data_product.remote(
                    measurement_result["data"], parameters, self.iir_sos
                )
            )
            # Generate capture metadata before sigan reconfigured
            cap_meta_tuple = self.create_channel_metadata(measurement_result)
            cap_meta.append(cap_meta_tuple[0])
            cap_entries.append(cap_meta_tuple[1])
            cpu_speed.append(get_current_cpu_clock_speed())
        capture_toc = perf_counter()
        logger.debug(
            f"Collected all IQ data and started all processing in {capture_toc-capture_tic:.2f} s"
        )

        # Collect processed data product results
        last_data_len = 0
        all_data, max_max_ch_pwrs, med_mean_ch_pwrs = [], [], []
        result_tic = perf_counter()
        for i, channel_data_process in enumerate(dp_procs):
            # Add capture metadata
            self.sigmf_builder.set_capture(
                cap_meta[i]["frequency"],
                cap_meta[i]["start_time"],
                sample_start=last_data_len,
                extra_entries=cap_entries[i],
            )

            # Now wait for channel data to be processed
            channel_data = []
            tic = perf_counter()
            for j, d in enumerate(ray.get(channel_data_process)):
                if j == 3:
                    # APD requires different handling
                    channel_data.append(ray.get(d))
                else:
                    channel_data.extend(ray.get(d))
            toc = perf_counter()
            logger.debug(f"Waited {toc-tic} s for channel {i} data")

            # Pull out single value channel powers (max of max, median of mean)
            max_max_ch_pwrs.append(DATA_TYPE(channel_data[4]))
            med_mean_ch_pwrs.append(DATA_TYPE(channel_data[5]))
            del channel_data[4:6]

            all_data.extend(NasctnSeaDataProduct.transform_data(channel_data))
            last_data_len = len(all_data)
        result_toc = perf_counter()
        logger.debug(f"Got all processed data in {result_toc-result_tic:.2f} s")

        del dp_procs

        # Build metadata and convert data to compressed bytes
        all_data = self.compress_bytes_data(np.array(all_data).tobytes())
        self.sigmf_builder.add_to_global(
            "ntia-nasctn-sea:max_of_max_channel_powers", max_max_ch_pwrs
        )
        self.sigmf_builder.add_to_global(
            "ntia-nasctn-sea:median_of_mean_channel_powers", med_mean_ch_pwrs
        )
        # Get diagnostics last to record action runtime
        self.capture_diagnostics(
            action_start_tic, cpu_speed
        )  # Add diagnostics to metadata
        self.sigmf_builder.build()

        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=all_data,
            metadata=self.sigmf_builder.metadata,
        )

    def capture_iq(self, params: dict) -> dict:
        """Acquire a single gap-free stream of IQ samples."""
        # Downselect params to limit meaningless logger warnings
        hw_params = {
            k: params[k]
            for k in [
                ATTENUATION,
                PREAMP_ENABLE,
                REFERENCE_LEVEL,
                SAMPLE_RATE,
                FREQUENCY,
            ]
        }
        start_time = utils.get_datetime_str_now()
        tic = perf_counter()
        # Configure signal analyzer
        self.configure_sigan(hw_params)
        # Get IQ capture parameters
        duration_ms = utils.get_parameter(DURATION_MS, params)
        nskip = utils.get_parameter(NUM_SKIP, params)
        num_samples = int(params[SAMPLE_RATE] * duration_ms * 1e-3)
        # Collect IQ data
        measurement_result = self.sigan.acquire_time_domain_samples(num_samples, nskip)
        # Store some metadata with the IQ
        measurement_result.update(params)
        measurement_result["start_time"] = start_time
        measurement_result["sensor_cal"] = self.sigan.sensor_calibration_data
        toc = perf_counter()
        logger.debug(
            f"IQ Capture ({duration_ms} ms @ {(params[FREQUENCY]/1e6):.1f} MHz) completed in {toc-tic:.2f} s."
        )
        return measurement_result

    def capture_diagnostics(self, action_start_tic: float, cpu_speeds: list) -> None:
        """
        Capture diagnostic sensor data.

        This method pulls diagnostic data from web relays in the
        sensor preselector and SPU as well as from the SPU computer
        itself. All temperatures are reported in degrees Celsius,
        and humidities in % relative humidity. The diagnostic data is
        added to the SigMF metadata as an object in the global namespace.

        The following diagnostic data is collected:

        From the preselector: LNA temperature, noise diode temperature,
            internal preselector temperature, internal preselector
            humidity, and preselector door open/closed status.

        From the SPU web relay: SPU RF tray power on/off, preselector
            power on/off, aux 28 V power on/off, SPU RF tray temperature,
            SPU power/control tray temperature, SPU power/control tray
            humidity.

        From the SPU computer: average CPU clock speed during action run,
            systemwide CPU utilization (%) averaged over the action runtime,
            system load (%) averaged over last 5 minutes, current memory
            usage (%), SSD SMART health check status, SSD usage percent,
            CPU temperature, CPU overheating status, CPU uptime, SCOS
            start time, and SCOS uptime.

        The total action runtime is also recorded.

        Preselector X410 Setup requires:
        internal temperature : oneWireSensor 1, set units to C
        noise diode temp : oneWireSensor 2, set units to C
        LNA temp : oneWireSensor 3, set units to C
        internal humidity : oneWireSensor 4
        door sensor: digitalInput 1

        SPU X410 requires:
        config file: name field must be "SPU X410"
        Power/Control Box Temperature: oneWireSensor 1, set units to C
        RF Box Temperature: oneWireSensor 2, set units to C
        Power/Control Box Humidity: oneWireSensor 3

        Various CPU metrics make assumptions about the system: only
        Intel CPUs are supported.

        Disk health check assumes the SSD is ``/dev/nvme0n1`` and
        requires the Docker container to have the required privileges
        or capabilities and device passthrough. For more information,
        see ``scos_actions.hardware.utils.get_disk_smart_data()``.

        :param action_start_tic: Action start timestamp, as would
             be returned by ``time.perf_counter()``
        """
        tic = perf_counter()
        # Read SPU sensors
        for switch in switches.values():
            if switch.name == "SPU X410":
                spu_diagnostics = switch.get_status()
                del spu_diagnostics["name"]
                del spu_diagnostics["healthy"]
                for sensor in SPU_SENSORS:
                    try:
                        value = switch.get_sensor_value(SPU_SENSORS[sensor])
                        spu_diagnostics[sensor] = value
                    except:
                        logger.warning(f"Unable to read {sensor} from SPU x410")
                        pass
                try:
                    spu_diagnostics["sigan_internal_temp"] = self.sigan.temperature
                except:
                    logger.warning("Unable to read internal sigan temperature")
                    pass

        # Read preselector sensors
        preselector_diagnostics = {}

        for sensor in PRESELECTOR_SENSORS:
            try:
                value = preselector.get_sensor_value(PRESELECTOR_SENSORS[sensor])
                preselector_diagnostics[sensor] = value
            except:
                logger.warning(f"Unable to read {sensor} from preselector")
                pass

        for inpt in PRESELECTOR_DIGITAL_INPUTS:
            try:
                value = preselector.get_digital_input_value(
                    PRESELECTOR_DIGITAL_INPUTS[inpt]
                )
                preselector_diagnostics[inpt] = value
            except:
                logger.warning(f"Unable to read {inpt} from preselector")
                pass

        # Read computer performance metrics
        computer_diagnostics = {}

        # CPU temperature
        try:
            cpu_temp_degC = get_current_cpu_temperature()
            computer_diagnostics["cpu_temp"] = round(cpu_temp_degC, 1)
        except:
            logger.warning("Failed to get current CPU temperature")

        # CPU overheating
        try:
            cpu_overheating = cpu_temp_degC > get_max_cpu_temperature()
            computer_diagnostics["cpu_overheating"] = cpu_overheating
        except:
            logger.warning("Failed to get CPU overheating status")

        # Computer uptime
        try:
            cpu_uptime_days = round(get_cpu_uptime_seconds() / (60 * 60 * 24), 2)
            computer_diagnostics["cpu_uptime"] = cpu_uptime_days
        except:
            logger.warning("Failed to get computer uptime")

        # CPU min/max/mean speeds
        computer_diagnostics.update(
            {
                "cpu_max_clock": round(max(cpu_speeds), 1),
                "cpu_min_clock": round(min(cpu_speeds), 1),
                "cpu_mean_clock": round(np.mean(cpu_speeds), 1),
            }
        )

        # Systemwide CPU utilization (%), averaged over current action runtime
        try:
            cpu_utilization = psutil.cpu_percent(interval=None)
            computer_diagnostics["action_cpu_usage"] = round(cpu_utilization, 1)
        except:
            logger.warning("Failed to get CPU utilization diagnostics")

        # Average system load (%) over last 5m
        try:
            load_avg_5m = (psutil.getloadavg()[1] / psutil.cpu_count()) * 100.0
            computer_diagnostics["system_load_5m"] = round(load_avg_5m, 1)
        except:
            logger.warning("Failed to get system load 5m average")

        # Memory usage
        try:
            mem_usage_pct = psutil.virtual_memory().percent
            computer_diagnostics["memory_usage"] = round(mem_usage_pct, 1)
        except:
            logger.warning("Failed to get memory usage")

        # SCOS start time
        try:
            computer_diagnostics[
                "scos_start_time"
            ] = convert_datetime_to_millisecond_iso_format(start_time)
        except:
            logger.warning("Failed to get SCOS start time")

        # SCOS uptime
        try:
            computer_diagnostics["scos_uptime"] = get_days_up()
        except:
            logger.warning("Failed to get SCOS uptime")

        # SSD SMART data
        try:
            computer_diagnostics["ssd_smart_data"] = get_disk_smart_data("/dev/nvme0n1")
        except:
            logger.warning("Failed to get SSD SMART data")

        toc = perf_counter()
        logger.debug(f"Got all diagnostics in {toc-tic} s")

        diagnostics = {
            "datetime": utils.get_datetime_str_now(),
            "preselector": preselector_diagnostics,
            "spu": spu_diagnostics,
            "computer": computer_diagnostics,
            "action_runtime": perf_counter() - action_start_tic,
        }

        # Make SigMF annotation from sensor data
        self.sigmf_builder.add_to_global("ntia-diagnostics:diagnostics", diagnostics)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            trigger_api_restart.send(sender=self.__class__)
            raise RuntimeError(msg)
        if "SPU X410" not in [s.name for s in switches.values()]:
            msg = "Configuration error: no switch configured with name 'SPU X410'"
            raise RuntimeError(msg)
        if not self.sigan.healthy():
            trigger_api_restart.send(sender=self.__class__)
        return None

    def create_global_data_product_metadata(self, params: dict) -> None:
        num_iq_samples = int(params[SAMPLE_RATE] * params[DURATION_MS] * 1e-3)
        dp_meta = {
            "digital_filter": "iir_1",
            "reference": "noise source output",
            "power_spectral_density": {
                "traces": [{"statistic": d.value.split("_")[0]} for d in FFT_DETECTOR],
                # Get sample count the same way that FFT processing truncates the result
                "length": int(FFT_SIZE * (5 / 7)),
                "equivalent_noise_bandwidth": round(
                    get_fft_enbw(FFT_WINDOW, params[SAMPLE_RATE]), 2
                ),
                "samples": FFT_SIZE,
                "ffts": int(params[NUM_FFTS]),
                "units": "dBm/Hz",
                "window": FFT_WINDOW_TYPE,
            },
            "time_series_power": {
                "traces": [{"detector": d.value.split("_")[0]} for d in TD_DETECTOR],
                # Get sample count the same way that TD power processing shapes result
                "length": int(params[DURATION_MS] / params[TD_BIN_SIZE_MS]),
                "samples": num_iq_samples,
                "units": "dBm",
            },
            "periodic_frame_power": {
                "traces": [
                    {
                        "detector": det,
                        "statistic": stat.value.split("_")[0],
                    }
                    for det in ["mean", "max"]
                    for stat in PFP_M3_DETECTOR
                ],
                # Get sample count the same way that data is reshaped for PFP calculation
                "length": int(
                    round((params[PFP_FRAME_PERIOD_MS] * 1e-3) / PFP_FRAME_RESOLUTION_S)
                ),
                "units": "dBm",
            },
            "amplitude_probability_distribution": {
                # Get sample count the same way that downsampling is applied
                "length": np.arange(
                    params[APD_MIN_BIN_DBM],
                    params[APD_MAX_BIN_DBM] + params[APD_BIN_SIZE_DB],
                    params[APD_BIN_SIZE_DB],
                ).size,
                "samples": num_iq_samples,
                "units": "dBm",
                "probability_units": "percent",
                "amplitude_bin_size": params[APD_BIN_SIZE_DB],
                "min_amplitude": params[APD_MIN_BIN_DBM],
                "max_amplitude": params[APD_MAX_BIN_DBM],
            },
        }
        self.sigmf_builder.add_to_global("ntia-algorithm:data_products", dp_meta)

        # Create DigitalFilter object
        iir_filter_meta = [
            {
                "id": "iir_1",
                "filter_type": "IIR",
                "IIR_numerator_coefficients": self.iir_numerators.tolist(),
                "IIR_denominator_coefficients": self.iir_denominators.tolist(),
                "ripple_passband": self.iir_gpass_dB,
                "attenuation_stopband": self.iir_gstop_dB,
                "frequency_stopband": self.iir_sb_edge_Hz,
                "frequency_cutoff": self.iir_pb_edge_Hz,
            },
        ]
        self.sigmf_builder.add_to_global(
            "ntia-algorithm:digital_filters", iir_filter_meta
        )

    def create_channel_metadata(
        self,
        measurement_result: dict,
    ) -> Tuple:
        """Add metadata corresponding to a single-frequency capture in the measurement"""
        # Construct dict of extra info to attach to capture
        entries_dict = {
            "ntia-sensor:overload": measurement_result["overload"],
            "ntia-sensor:duration": measurement_result[DURATION_MS],
            "ntia-sensor:sensor_calibration": {
                "noise_figure": round(
                    measurement_result["sensor_cal"]["noise_figure_sensor"], 3
                ),
                "gain": round(measurement_result["sensor_cal"]["gain_sensor"], 3),
                "temperature": round(
                    measurement_result["sensor_cal"]["temperature"], 1
                ),
                "datetime": measurement_result["sensor_cal"]["datetime"],
            },
            "ntia-sensor:sigan_settings": {
                "reference_level": self.sigan.reference_level,
                "attenuation": self.sigan.attenuation,
                "preamp_enable": self.sigan.preamp_enable,
            },
        }
        # Start time and frequency are needed when building
        # the capture metadata, but should not be in the capture_dict.
        capture_meta = {
            "start_time": measurement_result["start_time"],
            "frequency": self.sigan.frequency,
        }
        return capture_meta, entries_dict

    def get_sigmf_builder(
        self,
        sample_rate_Hz: float,
        task_id: int,
        schedule_entry: dict,
        iter_params: list,
    ) -> SigMFBuilder:
        """Build SigMF that applies to the entire capture (all channels)"""
        sigmf_builder = SigMFBuilder()
        try:
            loc = self.sensor_definition["location"]
            sigmf_builder.set_geolocation(loc["x"], loc["y"], loc["z"])
        except KeyError as e:
            logger.error("Set the sensor location in the SCOS admin web UI")
            raise e
        sigmf_builder.set_data_type(self.is_complex(), bit_width=16, endianness="")
        sigmf_builder.set_sample_rate(sample_rate_Hz)
        sigmf_builder.set_num_channels(len(iter_params))
        sigmf_builder.set_task(task_id)
        sigmf_builder.set_schedule(schedule_entry)

        # Add some (not all) ntia-sensor metadata
        sigmf_builder.set_sensor(
            {
                "id": self.sensor_definition["sensor_spec"]["id"],
                "sensor_spec": self.sensor_definition["sensor_spec"],
                "sensor_sha512": SENSOR_DEFINITION_HASH,
            }
        )

        # Mark data as UNCLASSIFIED
        sigmf_builder.add_to_global("ntia-core:classification", "UNCLASSIFIED")

        self.sigmf_builder = sigmf_builder

    @staticmethod
    def transform_data(channel_data_products: list):
        """Flatten data product list of arrays for a single channel."""
        # Flatten data product, reduce dtype
        return np.hstack(channel_data_products).astype(DATA_TYPE)

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
