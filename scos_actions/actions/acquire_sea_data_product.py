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
import psutil
import ray
from its_preselector.configuration_exception import ConfigurationException
from its_preselector.web_relay import WebRelay
from scipy.signal import sos2tf, sosfilt

from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
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
        workers=4,
    )
    fft_result = calculate_pseudo_power(fft_result)
    fft_result = apply_power_detector(fft_result, FFT_DETECTOR)  # (max, mean)
    ne.evaluate("fft_result/50", out=fft_result)  # Finish conversion to Watts
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
def get_apd_results(iqdata: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate downsampled APD result from IQ samples.

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include an
        `APD_BIN_SIZE_DB` key.
    :return: A tuple of NumPy arrays containing the APD amplitude and
        probability axes, ordered (probabilities, amplitudes). Probabilities
        are given as percentages, and amplitudes in dBm referenced to the
        calibration terminal.
    """
    p, a = get_apd(
        iqdata,
        params[APD_BIN_SIZE_DB],
        params[APD_MIN_BIN_DBM] - 27.0,
        params[APD_MAX_BIN_DBM] - 27.0,
        IMPEDANCE_OHMS,
    )
    # Convert dBW to dBm (+30) and account for RF/baseband conversion (-3)
    ne.evaluate("a+27.0", out=a)
    return p, a


@ray.remote
def get_td_power_results(
    iqdata: np.ndarray, params: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean/max time domain power statistics from IQ samples.

    Mean and max detectors are applied over a configurable time window.
    Mean and max power values are also reported as single values, with
    the detectors applied over the entire input IQ sample sequence.

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include
        `TD_BIN_SIZE_MS` and `SAMPLE_RATE` keys.
    :return: A tuple of NumPy arrays containing power detector results,
        in dBm referenced to the calibration terminal. The order of the
        results is (max, mean, max_single_value, mean_single_value). The
        single_value results will always contain only a single number,
        while the length of the other arrays depends on the configured
        detector period.
    """
    # Reshape IQ data into blocks
    block_size = int(params[TD_BIN_SIZE_MS] * params[SAMPLE_RATE] * 1e-3)
    n_blocks = len(iqdata) // block_size
    iqdata = iqdata.reshape((n_blocks, block_size))
    iq_pwr = calculate_power_watts(iqdata, IMPEDANCE_OHMS)

    # Apply mean/max detectors
    td_result = apply_power_detector(iq_pwr, TD_DETECTOR, axis=1)

    # Get single value mean/max statistics
    td_channel_result = np.array([np.mean(td_result[0]), np.median(td_result[1])])

    # Convert to dBm and account for RF/baseband power difference
    td_result, td_channel_result = (
        convert_watts_to_dBm(x) - 3.0 for x in [td_result, td_channel_result]
    )

    channel_max, channel_median = (np.array(a) for a in td_channel_result)

    # packed order is (max, mean)
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
    RMS and peak power detector data are returned. For each type of detector, a time
    series is returned for (min, mean, max) statistics, which are computed across the
    number of frames (`frame_period/Ts`).

    :param iqdata: Complex-valued input waveform samples.
    :param params: Action parameters from YAML, which must include `SAMPLE_RATE` and
        `PFP_FRAME_PERIOD_MS` keys.
    :param detector_period_s: Sampling period (s) within the frame.
    :return: A tuple of 6 NumPy arrays for the 6 detector results: (rms_min, rms_max,
        rms_mean, peak_min, peak_max, peak_mean).
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
    data_product = []
    iqdata = sosfilt(iir_sos, iqdata)

    remote_procs = []
    remote_procs.append(get_fft_results.remote(iqdata, params))
    remote_procs.append(get_td_power_results.remote(iqdata, params))
    remote_procs.append(get_periodic_frame_power.remote(iqdata, params))
    remote_procs.append(get_apd_results.remote(iqdata, params))
    all_results = ray.get(remote_procs)

    for dp in all_results:
        data_product.extend(dp)

    del iqdata
    gc.collect()

    # Flatten data product but retain component indices
    # Also, separate single value channel powers
    max_peak_chan_pwr = DATA_TYPE(data_product[4])
    med_rms_chan_pwr = DATA_TYPE(data_product[5])
    del data_product[4:6]
    data_product, dp_idx = NasctnSeaDataProduct.transform_data(data_product)

    return data_product, dp_idx, max_peak_chan_pwr, med_rms_chan_pwr


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

        # Collect all IQ data and spawn data product computation processes
        all_data, all_idx, dp_procs, cap_meta, cap_entries, cpu_speed = (
            [] for _ in range(6)
        )
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

        # Initialize metadata object
        self.get_sigmf_builder(
            # Assumes all sample rates are the same, and that the sigan
            # correctly uses the desired sample rate.
            iteration_params[0][SAMPLE_RATE],
            task_id,
            schedule_entry,  # Uses a single "last calibration time"
            iteration_params,
        )

        # Collect processed data product results
        last_data_len = 0
        results = ray.get(dp_procs)  # Ordering is retained
        max_peak_ch_pwrs, med_rms_ch_pwrs, apd_lengths = [], [], []
        for i, (dp, dp_idx, max_ch_pwr, med_ch_pwr) in enumerate(results):
            # Combine channel data
            all_data.extend(dp)
            all_idx.extend((dp_idx + last_data_len).tolist())

            # Add Capture
            self.sigmf_builder.set_capture(
                cap_meta[i]["frequency"],
                cap_meta[i]["start_time"],
                sample_start=last_data_len,
                extra_entries=cap_entries[i],
            )

            # Collect channel power statistics
            max_peak_ch_pwrs.append(max_ch_pwr)
            med_rms_ch_pwrs.append(med_ch_pwr)

            # Get APD result sizes for metadata
            apd_lengths.append(dp_idx[11] - dp_idx[10])

            # Increment start sample for data combination
            last_data_len = len(all_data)

        # Build metadata and convert data to compressed bytes
        all_data = self.compress_bytes_data(np.array(all_data).tobytes())
        self.sigmf_builder.add_to_global(
            "ntia-nasctn-sea:max_of_peak_channel_powers", max_peak_ch_pwrs
        )
        self.sigmf_builder.add_to_global(
            "ntia-nasctn-sea:median_of_rms_channel_powers", med_rms_ch_pwrs
        )
        self.create_global_data_product_metadata(self.parameters, apd_lengths)
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
        self.configure_sigan({k: v for k, v in hw_params.items() if k != RF_PATH})
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

    @staticmethod
    def read_sensor_if_available(relay: WebRelay, sensor_idx: int):
        try:
            value = relay.get_sensor_value(sensor_idx)
        except ConfigurationException:
            logger.debug(f"Could not read relay {relay.name} sensor {sensor_idx}")
            value = "Unavailable"
        except ValueError:
            logger.debug(
                f"Relay {relay.name} sensor {sensor_idx} returned an invalid value."
            )
            value = "Unavailable"
        return value

    @staticmethod
    def read_digital_input_if_available(relay: WebRelay, sensor_idx: int):
        try:
            value = relay.get_digital_input_value(sensor_idx)
        except ConfigurationException:
            logger.debug(
                f"Could not read relay {relay.name} digital input {sensor_idx}"
            )
            value = "Unavailable"
        return value

    def capture_diagnostics(self, action_start_tic: float, cpu_speeds: list) -> dict:
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
        spu_x410_sensor_values = "Unavailable"
        for switch in switches.values():
            if switch.name == "SPU X410":
                spu_x410_sensor_values = switch.get_status()
                del spu_x410_sensor_values["name"]
                del spu_x410_sensor_values["healthy"]
                spu_x410_sensor_values[
                    "pwr_box_temp_degC"
                ] = self.read_sensor_if_available(switch, 1)
                spu_x410_sensor_values[
                    "rf_box_temp_degC"
                ] = self.read_sensor_if_available(switch, 2)
                spu_x410_sensor_values[
                    "pwr_box_humidity_pct"
                ] = self.read_sensor_if_available(switch, 3)

        # Read preselector sensors
        preselector_sensor_values = {
            "internal_temp_degC": self.read_sensor_if_available(preselector, 1),
            "noise_diode_temp_degC": self.read_sensor_if_available(preselector, 2),
            "lna_temp_degC": self.read_sensor_if_available(preselector, 3),
            "internal_humidity_pct": self.read_sensor_if_available(preselector, 4),
            "door_closed": self.read_digital_input_if_available(preselector, 1),
        }

        # Read computer performance metrics

        # Systemwide CPU utilization (%), averaged over current action runtime
        cpu_utilization = psutil.cpu_percent(interval=None)

        # Average system load (%) over last 5m
        load_avg_5m = (psutil.getloadavg()[1] / psutil.cpu_count()) * 100.0

        # Memory usage
        mem_usage_pct = psutil.virtual_memory().percent

        # CPU temperature
        cpu_temp_degC = get_current_cpu_temperature()
        cpu_overheating = cpu_temp_degC > get_max_cpu_temperature()

        # Get computer uptime
        cpu_uptime_days = round(get_cpu_uptime_seconds() / (60 * 60 * 24), 4)

        computer_metrics = {
            "maximum_cpu_clock_speed_MHz": round(np.max(cpu_speeds), 2),
            "minimum_cpu_clock_speed_MHz": round(np.min(cpu_speeds), 2),
            "mean_cpu_clock_speed_MHz": round(np.mean(cpu_speeds), 2),
            "action_cpu_usage_pct": round(cpu_utilization, 2),
            "system_load_5m_pct": round(load_avg_5m, 2),
            "memory_usage_pct": round(mem_usage_pct, 2),
            "cpu_temperature_degC": round(cpu_temp_degC, 2),
            "cpu_overheating": cpu_overheating,
            "cpu_uptime_days": cpu_uptime_days,
            "scos_start_time": convert_datetime_to_millisecond_iso_format(start_time),
            "scos_uptime_days": get_days_up(),
            "ssd_smart_data": get_disk_smart_data("/dev/nvme0n1"),
        }
        toc = perf_counter()
        logger.debug(f"Got all diagnostics in {toc-tic} s")

        diag = {
            "diagnostics_datetime": utils.get_datetime_str_now(),
            "preselector": preselector_sensor_values,
            "spu_x410": spu_x410_sensor_values,
            "spu_computer": computer_metrics,
            "action_runtime_sec": perf_counter() - action_start_tic,
        }

        # Make SigMF annotation from sensor data
        self.sigmf_builder.add_to_global("ntia-nasctn-sea:diagnostics", diag)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sigan.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            raise RuntimeError(msg)
        if "SPU X410" not in [s.name for s in switches.values()]:
            msg = "Configuration error: no switch configured with name 'SPU X410'"
            raise RuntimeError(msg)
        if not self.sigan.healthy():
            trigger_api_restart.send(sender=self.__class__)
        return None

    def create_global_data_product_metadata(
        self, params: dict, apd_lengths: list
    ) -> None:
        # Assumes only one value is set for all channels of the following:
        # NUM_FFTS, SAMPLE_RATE, DURATION_MS, TD_BIN_SIZE_MS, PFP_FRAME_PERIOD_MS, APD_BIN_SIZE_DB
        num_iq_samples = int(params[SAMPLE_RATE] * params[DURATION_MS] * 1e-3)
        dp_meta = {
            "power_spectral_density": {
                # Approximately a FrequencyDomainDetection from ntia-algorithm
                "detector": [d.value for d in FFT_DETECTOR],
                # Get sample count the same way that FFT processing truncates the result
                "sample_count": int(FFT_SIZE * (5 / 7)),
                "equivalent_noise_bandwidth": round(
                    get_fft_enbw(FFT_WINDOW, params[SAMPLE_RATE]), 2
                ),
                "number_of_samples_in_fft": FFT_SIZE,
                "number_of_ffts": int(params[NUM_FFTS]),
                "units": "dBm/Hz",
                "window": FFT_WINDOW_TYPE,
                "reference": "noise source output",
            },
            "time_series_power": {
                # Approximately a TimeDomainDetection from ntia-algorithm
                "detector": [d.value for d in TD_DETECTOR],
                # Get sample count the same way that TD power processing shapes result
                "sample_count": int(params[DURATION_MS] / params[TD_BIN_SIZE_MS]),
                "number_of_samples": num_iq_samples,
                "units": "dBm",
                "reference": "noise source output",
            },
            "periodic_frame_power": {
                # Approximately a TimeDomainDetection from ntia-algorithm
                "detector": [
                    f"{m}_{d.value}" for m in ["rms", "peak"] for d in PFP_M3_DETECTOR
                ],
                # Get sample count the same way that data is reshaped for PFP calculation
                "sample_count": int(
                    np.round(
                        (params[PFP_FRAME_PERIOD_MS] * 1e-3) / PFP_FRAME_RESOLUTION_S
                    )
                ),
                "units": "dBm",
                "reference": "noise source output",
            },
            "amplitude_probability_distribution": {
                "sample_count": apd_lengths,
                "number_of_samples": num_iq_samples,
                "units": "dBm",
                "probability_units": "percent",
                "reference": "noise source output",
                "power_bin_size": round(2.0 * params[APD_BIN_SIZE_DB], 2),
            },
        }
        self.sigmf_builder.add_to_global("ntia-nasctn-sea:data_products", dp_meta)

        # Create DigitalFilter object
        iir_filter_meta = [
            {
                "filter_type": "IIR",
                "IIR_numerator_coefficients": self.iir_numerators.tolist(),
                "IIR_denominator_coefficients": self.iir_denominators.tolist(),
                "ripple_passband": self.iir_gpass_dB,
                "attenuation_stopband": self.iir_gstop_dB,
                "frequency_stopband": self.iir_sb_edge_Hz,
                "frequency_passband": self.iir_pb_edge_Hz,
            },
        ]
        self.sigmf_builder.add_to_global("digital_filters", iir_filter_meta)

    def create_channel_metadata(
        self,
        measurement_result: dict,
    ) -> Tuple:
        """Add metadata corresponding to a single-frequency capture in the measurement"""
        # Construct dict of extra info to attach to capture
        entries_dict = {
            "ntia-sensor:overload": measurement_result["overload"],
            "ntia-sensor:calibration": {
                "noise_figure_sensor": round(
                    measurement_result["sensor_cal"]["noise_figure_sensor"], 3
                ),
                "gain_sensor": round(
                    measurement_result["sensor_cal"]["gain_sensor"], 3
                ),
                "temperature": round(
                    measurement_result["sensor_cal"]["temperature"], 1
                ),
                "datetime": measurement_result["sensor_cal"]["datetime"],
            },
            "ntia-sensor:sigan_settings": {
                "class": "tekrsa_usb",
                "reference_level": self.sigan.reference_level,
                "attenuation": self.sigan.attenuation,
                "preamp_enable": self.sigan.preamp_enable,
                "auto_attenuation_enable": False,
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
        sigmf_builder.set_coordinate_system()
        sigmf_builder.set_data_type(self.is_complex(), bit_width=16, endianness="")
        sigmf_builder.set_sample_rate(sample_rate_Hz)
        sigmf_builder.set_num_channels(len(iter_params))
        sigmf_builder.set_task(task_id)
        sigmf_builder.set_schedule(schedule_entry)

        # Add some (not all) ntia-sensor metadata
        sensor_meta = {
            "id": self.sensor_definition["sensor_spec"]["id"],
            "sensor_spec": self.sensor_definition["sensor_spec"],
        }
        # Adding location requires the location to be set manually
        try:
            sensor_meta.update({"location": self.sensor_definition["location"]})
        except KeyError:
            logger.error("Set the sensor location in the SCOS admin web UI")
            sensor_meta.update({"location": "NOT CONFIGURED"})

        sigmf_builder.sigmf_md.set_global_field(
            "ntia-sensor:sensor",
            sensor_meta,
        )

        self.sigmf_builder = sigmf_builder

    @staticmethod
    def transform_data(data_product: list):
        """Flatten data product list of arrays (single channel), convert to bytes object, then compress"""
        # Get indices for start of each component in flattened result
        data_lengths = [d.size for d in data_product]
        idx = [0] + np.cumsum(data_lengths[:-1]).tolist()
        # Flatten data product, reduce dtypem
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
