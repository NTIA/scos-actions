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
import logging
import lzma
import platform
import sys
from enum import EnumMeta
from time import perf_counter
from typing import Tuple

import numpy as np
import psutil
import ray
from environs import Env
from its_preselector import __version__ as PRESELECTOR_API_VERSION
from scipy.signal import sos2tf, sosfilt

from scos_actions import __version__ as SCOS_ACTIONS_VERSION
from scos_actions import utils
from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor
from scos_actions.hardware.utils import (
    get_cpu_uptime_seconds,
    get_current_cpu_clock_speed,
    get_current_cpu_temperature,
    get_disk_smart_data,
    get_max_cpu_temperature,
)
from scos_actions.metadata.sigmf_builder import SigMFBuilder
from scos_actions.metadata.structs import (
    ntia_algorithm,
    ntia_core,
    ntia_diagnostics,
    ntia_scos,
    ntia_sensor,
)
from scos_actions.metadata.structs.capture import CaptureSegment
from scos_actions.metadata.utils import construct_geojson_point
from scos_actions.settings import SCOS_SENSOR_GIT_TAG
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
    apply_statistical_detector,
    calculate_power_watts,
    calculate_pseudo_power,
    create_statistical_detector,
)
from scos_actions.signals import measurement_action_completed, trigger_api_restart
from scos_actions.status import start_time
from scos_actions.utils import convert_datetime_to_millisecond_iso_format, get_days_up

env = Env()
logger = logging.getLogger(__name__)

if not ray.is_initialized():
    # Dashboard is only enabled if ray[default] is installed
    ray.init()

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
FFT_SIZE = 175  # 80 kHz resolution @ 14 MHz sampling rate
FFT_PERCENTILES = np.array([25, 75, 90, 95, 99, 99.9, 99.99])
FFT_WINDOW_TYPE = "flattop"
FFT_WINDOW = get_fft_window(FFT_WINDOW_TYPE, FFT_SIZE)
FFT_WINDOW_ECF = get_fft_window_correction(FFT_WINDOW, "energy")
IMPEDANCE_OHMS = 50.0
DATA_REFERENCE_POINT = "noise source output"
NUM_ACTORS = 3  # Number of ray actors to initialize

# Create power detectors
TD_DETECTOR = create_statistical_detector("TdMeanMaxDetector", ["max", "mean"])
FFT_M3_DETECTOR = create_statistical_detector(
    "FftM3Detector", ["max", "mean", "median"]
)
PFP_M3_DETECTOR = create_statistical_detector("PfpM3Detector", ["min", "max", "mean"])


@ray.remote
class PowerSpectralDensity:
    """
    Compute data product mean/max PSD results from IQ samples.

    The result is a PSD, with amplitudes in dBm/Hz referenced to the
    calibration terminal (assuming IQ data has been scaled for calibration).
    The result is truncated in frequency to the middle 10 MHz (middle 625 out
    of 875 total DFT samples).
    """

    def __init__(
        self,
        sample_rate_Hz: float,
        num_ffts: int,
        fft_size: int = FFT_SIZE,
        fft_window: np.ndarray = FFT_WINDOW,
        window_ecf: float = FFT_WINDOW_ECF,
        detector: EnumMeta = FFT_M3_DETECTOR,
        percentiles: np.ndarray = FFT_PERCENTILES,
        impedance_ohms: float = IMPEDANCE_OHMS,
    ):
        self.detector = detector
        self.percentiles = percentiles
        self.fft_size = fft_size
        self.fft_window = fft_window
        self.num_ffts = num_ffts
        # Get truncation points: truncate FFT result to middle 125 samples (middle 10 MHz from 14 MHz)
        self.bin_start = int(fft_size / 7)  # bin_start = 25 with FFT_SIZE 175
        self.bin_end = fft_size - self.bin_start  # bin_end = 150 with FFT_SIZE 175
        # Compute the amplitude shift for PSD scaling. The FFT result
        # is in pseudo-power log units and must be scaled to a PSD.
        self.fft_scale_factor = (
            -10.0 * np.log10(impedance_ohms)  # Pseudo-power to power
            + 27.0  # Watts to dBm (+30) and baseband to RF (-3)
            - 10.0 * np.log10(sample_rate_Hz * fft_size)  # PSD scaling
            + 20.0 * np.log10(window_ecf)  # Window energy correction
        )

    def run(self, iq: ray.ObjectRef) -> np.ndarray:
        """
        Compute power spectral densities from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: A 2D NumPy array of statistical detector results computed
            from PSD amplitudes, ordered (max, mean).
        """
        fft_amplitudes = get_fft(
            iq, self.fft_size, "backward", self.fft_window, self.num_ffts, False, 1
        )
        # Power in Watts
        fft_amplitudes = calculate_pseudo_power(fft_amplitudes)
        fft_result = apply_statistical_detector(
            fft_amplitudes, self.detector
        )  # (max, mean, median)
        percentile_result = np.percentile(fft_amplitudes, self.percentiles, axis=0)
        fft_result = np.vstack((fft_result, percentile_result))
        fft_result = np.fft.fftshift(fft_result, axes=(1,))  # Shift frequencies
        fft_result = fft_result[
            :, self.bin_start : self.bin_end
        ]  # Truncation to middle bins
        fft_result = 10.0 * np.log10(fft_result) + self.fft_scale_factor

        # Returned order is (max, mean, median, 25%, 75%, 90%, 95%, 99%, 99.9%, 99.99%)
        # Total of 10 arrays, each of length 125 (output shape (10, 125))
        # Percentile computation linearly interpolates. See numpy documentation.
        return fft_result


@ray.remote
class AmplitudeProbabilityDistribution:
    """
    Compute a downsampled amplitude probability distribution from IQ samples.

    The result is the probability axis of the downsampled APD. Downsampling is
    configurable through defining a bin size, minimum bin value, and maximum bin
    value for binning amplitudes. The amplitude data can be reconstructed if these
    three parameters are known.
    """

    def __init__(
        self,
        bin_size_dB: float,
        min_bin_dBm_baseband: float,
        max_bin_dBm_baseband: float,
        impedance_ohms: float = IMPEDANCE_OHMS,
    ):
        self.bin_size_dB = bin_size_dB
        self.impedance_ohms = impedance_ohms
        # get_apd requires amplitude bin edge values in dBW
        # Scale input to get_apd to account for:
        #     dBm -> dBW (-30)
        #     baseband -> RF power reference (+3)
        self.min_bin_dBW_RF = min_bin_dBm_baseband - 27.0
        self.max_bin_dBW_RF = max_bin_dBm_baseband - 27.0

    def run(self, iq: ray.ObjectRef) -> np.ndarray:
        """
        Generate the downsampled APD result from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: A NumPy array containing the APD probability axis as percentages.
        """
        p, _ = get_apd(
            iq,
            self.bin_size_dB,
            self.min_bin_dBW_RF,
            self.max_bin_dBW_RF,
            self.impedance_ohms,
        )
        return p


@ray.remote
class PowerVsTime:
    """
    Compute mean/max PVT results and summary statistics from IQ samples.

    The results are two NumPy arrays. The first contains the full PVT detector
    results (``detector`` results with ``bin_size_ms`` resolution). The second
    contains two values: the maximum of the first detector result, and the median
    of the second detector result. These are used to obtain max-of-max and median-of-
    mean channel power summary statistics.
    """

    def __init__(
        self,
        sample_rate_Hz: float,
        bin_size_ms: float,
        impedance_ohms: float = IMPEDANCE_OHMS,
        detector: EnumMeta = TD_DETECTOR,
    ):
        self.block_size = int(bin_size_ms * sample_rate_Hz * 1e-3)
        self.detector = detector
        self.impedance_ohms = impedance_ohms

    def run(self, iq: ray.ObjectRef) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power versus time results from IQ samples.

        :param iq: Complex-valued input waveform samples (which should
            already exist in the Ray object store).
        :return: Two NumPy arrays: the first has shape (2, 400) and
            the second is 1D with length 2. The first array contains
            the (max, mean) detector results and the second array contains
            the (max-of-max, median-of-mean, mean, median) single-valued
            summary statistics.
        """
        # Reshape IQ data into blocks and calculate power
        n_blocks = len(iq) // self.block_size
        iq_pwr = calculate_power_watts(
            iq.reshape((n_blocks, self.block_size)), self.impedance_ohms
        )
        # Get true median power
        pvt_median = np.median(iq_pwr.flatten())
        # Apply max/mean detectors
        pvt_result = apply_statistical_detector(iq_pwr, self.detector, axis=1)
        # Get single value statistics: (max-of-max, median-of-mean, mean, median)
        pvt_summary = np.array(
            [
                pvt_result[0].max(),
                np.median(pvt_result[1]),
                pvt_result[1].mean(),
                pvt_median,
            ]
        )
        # Convert to dBm and account for RF/baseband power difference
        # Note: convert_watts_to_dBm is not used to avoid NumExpr usage
        # for the relatively small arrays
        pvt_result, pvt_summary = (
            10.0 * np.log10(x) + 27.0 for x in [pvt_result, pvt_summary]
        )
        # Return order ((max array, mean array), (max-of-max, median-of-mean, mean, median))
        return pvt_result, pvt_summary


@ray.remote
class PeriodicFramePower:
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
        detector_period_s: float = PFP_FRAME_RESOLUTION_S,
        impedance_ohms: float = IMPEDANCE_OHMS,
        detector: EnumMeta = PFP_M3_DETECTOR,
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


@ray.remote
class IQProcessor:
    """
    Supervisor actor for IQ processing.

    Note: the current implementation in ``__call__`` makes the
    assumption that all ``params`` used are identical for each
    consecutive capture (i.e., each channel). The ``params`` which
    fall under this assumption are: ``SAMPLE_RATE``, ``NUM_FFTS``,
    ``TD_BIN_SIZE_MS``, ``PFP_FRAME_PERIOD_MS``, ``APD_BIN_SIZE_DB``,
    ``APD_MIN_BIN_DBM``, and ``APD_MAX_BIN_DBM``. A single set of IIR
    second-order-sections (``iir_sos``) is also used for filtering.

    Upon initialization of this supervisor actor, workers are spawned
    for each of the components of the data product. Initializing these
    processes allows for certain parts of the computation, which do not
    depend on the IQ data, to be performed ahead-of-time. The workers are
    stateful, and reuse the initialized values for each consecutive run.

    The ``run`` method can be called to filter and process IQ samples.
    Filtering happens before the remote workers are called, which run
    concurrently. The ``run`` method returns Ray object references
    immediately, which can be later used to retrieve the procesed results.
    """

    def __init__(self, params: dict, iir_sos: np.ndarray):
        # initialize worker processes
        self.iir_sos = iir_sos
        self.fft_worker = PowerSpectralDensity.remote(
            params[SAMPLE_RATE], params[NUM_FFTS]
        )
        self.pvt_worker = PowerVsTime.remote(
            params[SAMPLE_RATE], params[TD_BIN_SIZE_MS]
        )
        self.pfp_worker = PeriodicFramePower.remote(
            params[SAMPLE_RATE], params[PFP_FRAME_PERIOD_MS]
        )
        self.apd_worker = AmplitudeProbabilityDistribution.remote(
            params[APD_BIN_SIZE_DB], params[APD_MIN_BIN_DBM], params[APD_MAX_BIN_DBM]
        )
        self.workers = [
            self.fft_worker,
            self.pvt_worker,
            self.pfp_worker,
            self.apd_worker,
        ]
        del params

    def run(self, iqdata: np.ndarray) -> list:
        """
        Filter the input IQ data and concurrently compute FFT, PVT, PFP, and APD results.

        :param iqdata: Complex-valued input waveform samples.
        :return: A list of Ray object references which can be used to
            retrieve the processed results. The order is [FFT, PVT, PFP, APD].
        """
        # Filter IQ and place it in the object store
        iqdata = ray.put(sosfilt(self.iir_sos, iqdata))
        # Compute PSD, PVT, PFP, and APD concurrently.
        # Do not wait until they finish. Yield references to their results.
        yield [worker.run.remote(iqdata) for worker in self.workers]
        del iqdata


class NasctnSeaDataProduct(Action):
    """Acquire a stepped-frequency NASCTN SEA data product.

    :param parameters: The dictionary of parameters needed for
        the action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
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

        # Get iterable parameter list
        self.iteration_params = utils.get_iterable_parameters(self.parameters)

    def __call__(self, sensor, schedule_entry, task_id):
        """This is the entrypoint function called by the scheduler."""
        self._sensor = sensor
        action_start_tic = perf_counter()

        _ = psutil.cpu_percent(interval=None)  # Initialize CPU usage monitor
        self.test_required_components()
        self.configure_preselector(self.sensor, self.rf_path)

        # Initialize metadata object
        self.get_sigmf_builder(
            # Assumes the sigan correctly uses the configured sample rate.
            self.iteration_params[0][SAMPLE_RATE],
            task_id,
            schedule_entry,
            self.iteration_params,
        )
        self.create_global_sensor_metadata()
        self.create_global_data_product_metadata()

        # Initialize remote supervisor actors for IQ processing
        tic = perf_counter()
        # This uses iteration_params[0] because
        iq_processors = [
            IQProcessor.remote(self.iteration_params[0], self.iir_sos)
            for _ in range(NUM_ACTORS)
        ]
        toc = perf_counter()
        logger.debug(f"Spawned {NUM_ACTORS} supervisor actors in {toc-tic:.2f} s")

        # Collect all IQ data and spawn data product computation processes
        dp_procs, cpu_speed = [], []
        capture_tic = perf_counter()
        for i, parameters in enumerate(self.iteration_params):
            measurement_result = self.capture_iq(parameters)
            # Start data product processing but do not block next IQ capture
            tic = perf_counter()

            dp_procs.append(
                iq_processors[i % NUM_ACTORS].run.remote(measurement_result["data"])
            )
            del measurement_result["data"]
            toc = perf_counter()
            logger.debug(f"IQ data delivered for processing in {toc-tic:.2f} s")
            # Create capture segment with channel-specific metadata before sigan is reconfigured
            tic = perf_counter()
            self.create_capture_segment(i, measurement_result)
            toc = perf_counter()
            logger.debug(f"Created capture metadata in {toc-tic:.2f} s")
            cpu_speed.append(get_current_cpu_clock_speed())
        capture_toc = perf_counter()
        logger.debug(
            f"Collected all IQ data and started all processing in {capture_toc-capture_tic:.2f} s"
        )

        # Collect processed data product results
        all_data, max_max_ch_pwrs, med_mean_ch_pwrs, mean_ch_pwrs, median_ch_pwrs = (
            [],
            [],
            [],
            [],
            [],
        )
        result_tic = perf_counter()
        for channel_data_process in dp_procs:
            # Retrieve object references for channel data
            channel_data_refs = ray.get(channel_data_process)
            channel_data = []
            for i, data_ref in enumerate(channel_data_refs):
                # Now block until the data is ready
                data = ray.get(data_ref)
                if i == 1:
                    # Power-vs-Time results, a tuple of arrays
                    data, summaries = data  # Split the tuple
                    max_max_ch_pwrs.append(DATA_TYPE(summaries[0]))
                    med_mean_ch_pwrs.append(DATA_TYPE(summaries[1]))
                    mean_ch_pwrs.append(DATA_TYPE(summaries[2]))
                    median_ch_pwrs.append(DATA_TYPE(summaries[3]))
                    del summaries
                if i == 3:  # Separate condition is intentional
                    # APD result: append instead of extend,
                    # since the result is a single 1D array
                    channel_data.append(data)
                else:
                    # For 2D arrays (PSD, PVT, PFP)
                    channel_data.extend(data)
            toc = perf_counter()
            logger.debug(f"Waited {toc-tic} s for channel data")
            all_data.extend(NasctnSeaDataProduct.transform_data(channel_data))
        for ray_actor in iq_processors:
            ray.kill(ray_actor)
        result_toc = perf_counter()
        del dp_procs, iq_processors, channel_data, channel_data_refs
        logger.debug(f"Got all processed data in {result_toc-result_tic:.2f} s")

        # Build metadata and convert data to compressed bytes
        all_data = self.compress_bytes_data(np.array(all_data).tobytes())
        self.sigmf_builder.set_max_of_max_channel_powers(max_max_ch_pwrs)
        self.sigmf_builder.set_median_of_mean_channel_powers(med_mean_ch_pwrs)
        self.sigmf_builder.set_mean_channel_powers(mean_ch_pwrs)
        self.sigmf_builder.set_median_channel_powers(median_ch_pwrs)
        # Get diagnostics last to record action runtime
        self.capture_diagnostics(
            action_start_tic, cpu_speed
        )  # Add diagnostics to metadata

        measurement_action_completed.send(
            sender=self.__class__,
            task_id=task_id,
            data=all_data,
            metadata=self.sigmf_builder.metadata,
        )

    def capture_iq(self, params: dict) -> dict:
        """Acquire a single gap-free stream of IQ samples."""
        tic = perf_counter()
        # Configure signal analyzer
        self.configure_sigan(params)
        # Get IQ capture parameters
        duration_ms = utils.get_parameter(DURATION_MS, params)
        nskip = utils.get_parameter(NUM_SKIP, params)
        num_samples = int(params[SAMPLE_RATE] * duration_ms * 1e-3)
        # Collect IQ data
        measurement_result = self.sensor.signal_analyzer.acquire_time_domain_samples(
            num_samples, nskip
        )
        # Store some metadata with the IQ
        measurement_result.update(params)
        measurement_result[
            "sensor_cal"
        ] = self.sensor.signal_analyzer.sensor_calibration_data
        toc = perf_counter()
        logger.debug(
            f"IQ Capture ({duration_ms} ms @ {(params[FREQUENCY]/1e6):.1f} MHz) completed in {toc-tic:.2f} s."
        )
        return measurement_result

    def capture_diagnostics(
        self, sensor: Sensor, action_start_tic: float, cpu_speeds: list
    ) -> None:
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

        Software versions: the OS platform, Python version, scos_actions
            version, the preselector API version, the signal analyzer API
            version, and the signal analyzer firmware version.

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
        :param cpu_speeds: List of CPU speed values, recorded at
            consecutive points as the action has been running.
        """
        tic = perf_counter()
        switch_diag = {}
        all_switch_status = {}
        # Add status for any switch
        for switch in self.sensor.switches.values():
            switch_status = switch.get_status()
            del switch_status["name"]
            del switch_status["healthy"]
            all_switch_status.update(switch_status)

        self.set_ups_states(all_switch_status, switch_diag)
        self.add_temperature_and_humidity_sensors(all_switch_status, switch_diag)
        self.add_power_sensors(all_switch_status, switch_diag)
        self.add_power_states(all_switch_status, switch_diag)
        if "door_state" in all_switch_status:
            switch_diag["door_closed"] = not bool(all_switch_status["door_state"])

        # Read preselector sensors
        ps_diag = sensor.preselector.get_status()
        del ps_diag["name"]
        del ps_diag["healthy"]

        # Read computer performance metrics
        cpu_diag = {  # Start with CPU min/max/mean speeds
            "cpu_max_clock": round(max(cpu_speeds), 1),
            "cpu_min_clock": round(min(cpu_speeds), 1),
            "cpu_mean_clock": round(np.mean(cpu_speeds), 1),
            "action_runtime": round(perf_counter() - action_start_tic, 2),
        }
        try:  # Computer uptime (days)
            cpu_diag["cpu_uptime"] = round(get_cpu_uptime_seconds() / (60 * 60 * 24), 2)
        except:
            logger.warning("Failed to get computer uptime")
        try:  # System CPU utilization (%), averaged over current action runtime
            cpu_utilization = psutil.cpu_percent(interval=None)
            cpu_diag["action_cpu_usage"] = round(cpu_utilization, 1)
        except:
            logger.warning("Failed to get CPU utilization diagnostics")
        try:  # Average system load (%) over last 5m
            load_avg_5m = (psutil.getloadavg()[1] / psutil.cpu_count()) * 100.0
            cpu_diag["system_load_5m"] = round(load_avg_5m, 1)
        except:
            logger.warning("Failed to get system load 5m average")
        try:  # Memory usage
            mem_usage_pct = psutil.virtual_memory().percent
            cpu_diag["memory_usage"] = round(mem_usage_pct, 1)
        except:
            logger.warning("Failed to get memory usage")
        try:  # CPU temperature
            cpu_temp_degC = get_current_cpu_temperature()
            cpu_diag["cpu_temp"] = round(cpu_temp_degC, 1)
        except:
            logger.warning("Failed to get current CPU temperature")
        try:  # CPU overheating
            cpu_diag["cpu_overheating"] = cpu_temp_degC > get_max_cpu_temperature()
        except:
            logger.warning("Failed to get CPU overheating status")
        try:  # SCOS start time
            cpu_diag["software_start"] = convert_datetime_to_millisecond_iso_format(
                start_time
            )
        except:
            logger.warning("Failed to get SCOS start time")
        try:  # SCOS uptime
            cpu_diag["software_uptime"] = get_days_up()
        except:
            logger.warning("Failed to get SCOS uptime")
        try:  # SSD SMART data
            smart_data = get_disk_smart_data("/dev/nvme0n1")
            cpu_diag["ssd_smart_data"] = ntia_diagnostics.SsdSmartData(**smart_data)
        except:
            logger.warning("Failed to get SSD SMART data")

        # Get software versions
        software_diag = {
            "system_platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "scos_sensor_version": SCOS_SENSOR_GIT_TAG,
            "scos_actions_version": SCOS_ACTIONS_VERSION,
            "scos_sigan_plugin": ntia_diagnostics.ScosPlugin(
                name="scos_tekrsa", version=self.sensor.signal_analyzer.plugin_version
            ),
            "preselector_api_version": PRESELECTOR_API_VERSION,
            "sigan_firmware_version": self.sensor.signal_analyzer.firmware_version,
            "sigan_api_version": self.sensor.signal_analyzer.api_version,
        }

        toc = perf_counter()
        logger.debug(f"Got all diagnostics in {toc-tic} s")
        diagnostics = {
            "datetime": utils.get_datetime_str_now(),
            "preselector": ntia_diagnostics.Preselector(**ps_diag),
            "spu": ntia_diagnostics.SPU(**switch_diag),
            "computer": ntia_diagnostics.Computer(**cpu_diag),
            "software": ntia_diagnostics.Software(**software_diag),
        }

        # Add diagnostics to SigMF global object
        self.sigmf_builder.set_diagnostics(ntia_diagnostics.Diagnostics(**diagnostics))

    def set_ups_states(self, all_switch_status: dict, switch_diag: dict):
        if "ups_power_state" in all_switch_status:
            switch_diag["battery_backup"] = not all_switch_status["ups_power_state"]
        else:
            logger.warning("No ups_power_state found in switch status.")

        if "ups_battery_level" in all_switch_status:
            switch_diag["low_battery"] = not all_switch_status["ups_battery_level"]
        else:
            logger.warning("No ups_battery_level found in switch status.")

        if "ups_state" in all_switch_status:
            switch_diag["ups_healthy"] = not all_switch_status["ups_state"]
        else:
            logger.warning("No ups_state found in switch status.")

        if "ups_battery_state" in all_switch_status:
            switch_diag["replace_battery"] = not all_switch_status["ups_battery_state"]
        else:
            logger.warning("No ups_battery_state found in switch status")

    def add_temperature_and_humidity_sensors(
        self, all_switch_status: dict, switch_diag: dict
    ):
        switch_diag["temperature_sensors"] = []
        if "internal_temp" in all_switch_status:
            switch_diag["temperature_sensors"].append(
                {"name": "internal_temp", "value": all_switch_status["internal_temp"]}
            )
        else:
            logger.warning("No internal_temp found in switch status.")
        try:
            switch_diag["temperature_sensors"].append(
                {
                    "name": "sigan_internal_temp",
                    "value": self.sensor.signal_analyzer.temperature,
                }
            )
        except:
            logger.warning("Unable to read internal sigan temperature")

        if "tec_intake_temp" in all_switch_status:
            switch_diag["temperature_sensors"].append(
                {
                    "name": "tec_intake_temp",
                    "value": all_switch_status["tec_intake_temp"],
                }
            )
        else:
            logger.warning("No tec_intake_temp found in switch status.")

        if "tec_exhaust_temp" in all_switch_status:
            switch_diag["temperature_sensors"].append(
                {
                    "name": "tec_exhaust_temp",
                    "value": all_switch_status["tec_exhaust_temp"],
                }
            )
        else:
            logger.warning("No tec_exhaust_temp found in switch status.")

        if "internal_humidity" in all_switch_status:
            switch_diag["humidity_sensors"] = [
                {
                    "name": "internal_humidity",
                    "value": all_switch_status["internal_humidity"],
                }
            ]
        else:
            logger.warning("No internal_humidity found in switch status.")

    def add_power_sensors(self, all_switch_status: dict, switch_diag: dict):
        switch_diag["power_sensors"] = []
        if "power_monitor5V" in all_switch_status:
            switch_diag["power_sensors"].append(
                {
                    "name": "5V Monitor",
                    "value": all_switch_status["power_monitor5V"],
                    "expected_value": 5.0,
                }
            )
        else:
            logger.warning("No power_monitor5V found in switch status")

        if "power_monitor15V" in all_switch_status:
            switch_diag["power_sensors"].append(
                {
                    "name": "15V Monitor",
                    "value": all_switch_status["power_monitor15V"],
                    "expected_value": 15.0,
                }
            )
        else:
            logger.warning("No power_monitor15V found in switch status.")

        if "power_monitor24V" in all_switch_status:
            switch_diag["power_sensors"].append(
                {
                    "name": "24V Monitor",
                    "value": all_switch_status["power_monitor24V"],
                    "expected_value": 24.0,
                }
            )
        else:
            logger.warning("No power_monitor24V found in switch status")

        if "power_monitor28V" in all_switch_status:
            switch_diag["power_sensors"].append(
                {
                    "name": "28V Monitor",
                    "value": all_switch_status["power_monitor28V"],
                    "expected_value": 28.0,
                }
            )
        else:
            logger.warning("No power_monitor28V found in switch status")

    def add_heating_cooling(self, all_switch_status: dict, switch_diag: dict):
        if "heating" in all_switch_status:
            switch_diag["heating"] = all_switch_status["heating"]
        else:
            logger.warning("No heating found in switch status.")

        if "cooling" in all_switch_status:
            switch_diag["cooling"] = all_switch_status["cooling"]
        else:
            logger.warning("No cooling found in switch status")

    def add_power_states(self, all_switch_status: dict, switch_diag: dict):
        if "sigan_powered" in all_switch_status:
            switch_diag["sigan_powered"] = all_switch_status["sigan_powered"]
        else:
            logger.warning("No sigan_powered found in switch status.")

        if "temperature_control_powered" in all_switch_status:
            switch_diag["temperature_control_powered"] = all_switch_status[
                "temperature_control_powered"
            ]
        else:
            logger.warning("No temperature_control_powered found in switch status.")

        if "preselector_powered" in all_switch_status:
            switch_diag["preselector_powered"] = all_switch_status[
                "preselector_powered"
            ]
        else:
            logger.warning("No preselector_powered found in switch status.")

    def create_global_sensor_metadata(self, sensor: Sensor):
        # Add (minimal) ntia-sensor metadata to the sigmf_builder:
        #   sensor ID, serial numbers for preselector, sigan, and computer
        #   overall sensor_spec version, e.g. "Prototype Rev. 3"
        #   sensor definition hash, to link to full sensor definition
        self.sigmf_builder.set_sensor(
            ntia_sensor.Sensor(
                sensor_spec=ntia_core.HardwareSpec(
                    id=self.sensor.capabilities["sensor"]["sensor_spec"]["id"],
                ),
                sensor_sha512=sensor.capabilities["sensor"]["sensor_sha512"],
            )
        )

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.sensor.signal_analyzer.is_available:
            msg = "Acquisition failed: signal analyzer is not available"
            trigger_api_restart.send(sender=self.__class__)
            raise RuntimeError(msg)
        if not self.sensor.signal_analyzer.healthy():
            trigger_api_restart.send(sender=self.__class__)
        return None

    def create_global_data_product_metadata(self) -> None:
        p = self.parameters
        num_iq_samples = int(p[SAMPLE_RATE] * p[DURATION_MS] * 1e-3)
        iir_obj = ntia_algorithm.DigitalFilter(
            id="iir_1",
            filter_type=ntia_algorithm.FilterType.IIR,
            feedforward_coefficients=self.iir_numerators.tolist(),
            feedback_coefficients=self.iir_denominators.tolist(),
            attenuation_cutoff=self.iir_gstop_dB,
            frequency_cutoff=self.iir_sb_edge_Hz,
            description="5 MHz lowpass filter used as complex 10 MHz bandpass for channelization",
        )
        self.sigmf_builder.set_processing([iir_obj.id])

        dft_obj = ntia_algorithm.DFT(
            id="psd_fft",
            equivalent_noise_bandwidth=round(
                get_fft_enbw(FFT_WINDOW, p[SAMPLE_RATE]), 2
            ),
            samples=FFT_SIZE,
            dfts=int(p[NUM_FFTS]),
            window=FFT_WINDOW_TYPE,
            baseband=True,
            description=f"First and last {int(FFT_SIZE / 7)} samples from {FFT_SIZE}-point FFT discarded",
        )
        self.sigmf_builder.set_processing_info([iir_obj, dft_obj])

        psd_length = int(FFT_SIZE * (5 / 7))
        psd_bin_start = int(FFT_SIZE / 7)  # bin_start = 125 with FFT_SIZE 875
        psd_bin_end = FFT_SIZE - psd_bin_start  # bin_end = 750 with FFT_SIZE 875
        psd_x_axis__Hz = get_fft_frequencies(FFT_SIZE, p[SAMPLE_RATE], 0.0)  # Baseband
        psd_graph = ntia_algorithm.Graph(
            name="Power Spectral Density",
            series=[d.value for d in FFT_M3_DETECTOR]
            + [
                f"{int(p)}th_percentile" if p.is_integer() else f"{p}th_percentile"
                for p in FFT_PERCENTILES
            ],  # ["max", "mean", "median", "25th_percentile", "75th_percentile", ... "99.99th_percentile"]
            length=int(FFT_SIZE * (5 / 7)),
            x_units="Hz",
            x_start=[psd_x_axis__Hz[psd_bin_start]],
            x_stop=[psd_x_axis__Hz[psd_bin_end - 1]],  # -1 for zero-indexed array
            x_step=[p[SAMPLE_RATE] / FFT_SIZE],
            y_units="dBm/Hz",
            processing=[dft_obj.id],
            reference=DATA_REFERENCE_POINT,
            description=(
                "Results of statistical detectors (max, mean, median, 25th_percentile, 75th_percentile, "
                + "90th_percentile, 95th_percentile, 99th_percentile, 99.9th_percentile, 99.99th_percentile) "
                + f"applied to power spectral density samples, with the first and last {int(FFT_SIZE / 7)} "
                + "samples discarded. FFTs computed on IIR-filtered data."
            ),
        )

        pvt_length = int(p[DURATION_MS] / p[TD_BIN_SIZE_MS])
        pvt_x_axis__s = np.arange(pvt_length) * (p[DURATION_MS] / 1e3 / pvt_length)
        pvt_graph = ntia_algorithm.Graph(
            name="Power vs. Time",
            series=[d.value for d in TD_DETECTOR],  # ["max", "mean"]
            length=pvt_length,
            x_units="s",
            x_start=[pvt_x_axis__s[0]],
            x_stop=[pvt_x_axis__s[-1]],
            x_step=[pvt_x_axis__s[1] - pvt_x_axis__s[0]],
            y_units="dBm",
            reference=DATA_REFERENCE_POINT,
            description=(
                "Max- and mean-detected channel power vs. time, with "
                + f"an integration time of {p[TD_BIN_SIZE_MS]} ms. "
                + "Each data point represents the result of a statistical "
                + f"detector applied over the previous {p[TD_BIN_SIZE_MS]}."
                + f" In total, {num_iq_samples} IQ samples were used as the input."
            ),
        )

        pfp_length = int(round((p[PFP_FRAME_PERIOD_MS] / 1e3) / PFP_FRAME_RESOLUTION_S))
        pfp_x_axis__s = np.arange(pfp_length) * (
            p[DURATION_MS] / 1e3 / pfp_length / pvt_length
        )
        pfp_graph = ntia_algorithm.Graph(
            name="Periodic Frame Power",
            series=[
                f"{det}_{stat.value}"
                for det in ["mean", "max"]
                for stat in PFP_M3_DETECTOR
            ],
            length=pfp_length,
            x_units="s",
            x_start=[pfp_x_axis__s[0]],
            x_stop=[pfp_x_axis__s[-1]],
            x_step=[pfp_x_axis__s[1] - pfp_x_axis__s[0]],
            y_units="dBm",
            reference=DATA_REFERENCE_POINT,
            description=(
                "Channelized periodic frame power statistics reported over"
                + f" a {p[PFP_FRAME_PERIOD_MS]} ms frame period, with frame resolution"
                + f" of {PFP_FRAME_RESOLUTION_S} s. Mean and max detectors are first "
                + f"applied over the frame resolution, then {[d.value for d in PFP_M3_DETECTOR]} statistics"
                + " are computed on samples sharing the same index within the frame period."
            ),
        )

        apd_y_axis__dBm = np.arange(
            p[APD_MIN_BIN_DBM],
            p[APD_MAX_BIN_DBM] + p[APD_BIN_SIZE_DB],
            p[APD_BIN_SIZE_DB],
        )
        apd_graph = ntia_algorithm.Graph(
            name="Amplitude Probability Distribution",
            length=apd_y_axis__dBm.size,
            x_units="percent",
            y_units="dBm",
            y_start=[apd_y_axis__dBm[0]],
            y_stop=[apd_y_axis__dBm[-1]],
            y_step=[apd_y_axis__dBm[1] - apd_y_axis__dBm[0]],
            description=(
                f"Estimate of the APD, using a {p[APD_BIN_SIZE_DB]} dB "
                + "bin size for amplitude values. The data payload includes"
                + " probability values, as percentages, indicating the "
                + "probability that a given IQ sample exceeds the corresponding"
                + " amplitudes, the y-axis values recorded by this metadata object."
            ),
        )

        self.sigmf_builder.set_data_products(
            [psd_graph, pvt_graph, pfp_graph, apd_graph]
        )
        self.total_channel_data_length = (
            psd_length * (len(FFT_M3_DETECTOR) + len(FFT_PERCENTILES))
            + pvt_length * len(TD_DETECTOR)
            + pfp_length * len(PFP_M3_DETECTOR) * 2
            + apd_graph.length
        )

    def create_capture_segment(
        self,
        channel_idx: int,
        measurement_result: dict,
    ) -> None:
        """Add metadata corresponding to a single-frequency capture in the measurement"""

        capture_segment = CaptureSegment(
            sample_start=channel_idx * self.total_channel_data_length,
            frequency=self.sensor.signal_analyzer.frequency,
            datetime=measurement_result["capture_time"],
            duration=measurement_result[DURATION_MS],
            overload=measurement_result["overload"],
            sensor_calibration=ntia_sensor.Calibration(
                datetime=measurement_result["sensor_cal"]["datetime"],
                gain=round(measurement_result["sensor_cal"]["gain"], 3),
                noise_figure=round(measurement_result["sensor_cal"]["noise_figure"], 3),
                temperature=round(measurement_result["sensor_cal"]["temperature"], 1),
                reference=DATA_REFERENCE_POINT,
            ),
            sigan_settings=ntia_sensor.SiganSettings(
                reference_level=self.sensor.signal_analyzer.reference_level,
                attenuation=self.sensor.signal_analyzer.attenuation,
                preamp_enable=self.sensor.signal_analyzer.preamp_enable,
            ),
        )
        self.sigmf_builder.add_capture(capture_segment)

    def get_sigmf_builder(
        self,
        sample_rate_Hz: float,
        task_id: int,
        schedule_entry: dict,
        iter_params: list,
    ) -> None:
        """Build SigMF that applies to the entire capture (all channels)"""
        sigmf_builder = SigMFBuilder()

        # Keep only keys supported by ntia-scos ScheduleEntry SigMF
        schedule_entry_cleaned = {
            k: v
            for k, v in schedule_entry.items()
            if k in ["id", "name", "start", "stop", "interval", "priority", "roles"]
        }
        if "id" not in schedule_entry_cleaned:
            # If there is no ID, reuse the "name" as the ID
            schedule_entry_cleaned["id"] = schedule_entry_cleaned["name"]
        schedule_entry_obj = ntia_scos.ScheduleEntry(**schedule_entry_cleaned)
        sigmf_builder.set_schedule(schedule_entry_obj)

        action_obj = ntia_scos.Action(
            name=self.name,
            summary=self.summary,
            # Do not include lengthy description
        )
        sigmf_builder.set_action(action_obj)
        if self.sensor.location is not None:
            sigmf_builder.set_geolocation(self.sensor.location)
        else:
            raise Exception("Sensor does not have a location defined.")
        sigmf_builder.set_data_type(self.is_complex(), bit_width=16, endianness="")
        sigmf_builder.set_sample_rate(sample_rate_Hz)
        sigmf_builder.set_num_channels(len(iter_params))
        sigmf_builder.set_task(task_id)

        # Mark data as CUI (basic)
        sigmf_builder.set_classification("CUI")

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
