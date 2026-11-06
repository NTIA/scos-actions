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
r"""Acquire generalized SEA data products.

"""
import logging
import lzma
import platform
import sys
from time import perf_counter
from typing import Optional

import numpy as np
import psutil
import ray
from environs import Env
from its_preselector import __version__ as PRESELECTOR_API_VERSION
from scipy.signal import sos2tf

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
from scos_actions.data_product.distributed import IQProcessorProductActors
from scos_actions.settings import SCOS_SENSOR_GIT_TAG, SSD_DEVICE
from scos_actions.signal_processing.filtering import (
    generate_elliptic_iir_low_pass_filter,
)
from scos_actions.signal_processing.power_analysis import (
    create_statistical_detector,
)
from scos_actions.signals import measurement_action_completed, trigger_api_restart
from scos_actions.utils import (
    convert_datetime_to_millisecond_iso_format,
    get_days_up,
    get_disk_usage,
)

env = Env()
logger = logging.getLogger(__name__)

# Define parameter keys
RF_PATH = "rf_path"
IIR_GPASS = "iir_gpass_dB"
IIR_GSTOP = "iir_gstop_dB"
IIR_PB_EDGE = "iir_pb_edge_Hz"
IIR_SB_EDGE = "iir_sb_edge_Hz"
IIR_NUMERATORS = "iir_numerators"
IIR_DENOMINATORS = "iir_denominators"
IIR_SOS = "iir_sos"
FFT_SIZE = "fft_size"  # Optional; defaults to DEFAULT_FFT_SIZE
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

# Generalized action parameter keys
DATA_PRODUCTS = "data_products"
CAPTURES = "captures"

# Constants
DATA_TYPE = np.half
PFP_FRAME_RESOLUTION_S = (1e-3 * (1 + 1 / (14)) / 15) / 4
DEFAULT_FFT_SIZE = 175  # 80 kHz resolution @ 14 MHz sampling rate
FFT_PERCENTILES = np.array([25, 75, 90, 95, 99, 99.9, 99.99])
FFT_WINDOW_TYPE = "flattop"
FFT_WINDOW_CORRECTION_TYPE = "energy"
IMPEDANCE_OHMS = 50.0
ALL_DATA_PRODUCTS = ["psd", "pvt", "pfp", "apd"]
NUM_ACTORS = env.int("RAY_WORKERS", default=3)  # Number of ray actors to initialize

# Create power detectors
TD_DETECTOR = create_statistical_detector("TdMeanMaxDetector", ["max", "mean"])
FFT_M3_DETECTOR = create_statistical_detector(
    "FftM3Detector", ["max", "mean", "median"]
)
PFP_M3_DETECTOR = create_statistical_detector("PfpM3Detector", ["min", "max", "mean"])


class GeneralizedSeaDataProduct(Action):
    """Acquire a stepped-frequency Generalized SEA data product.

    :param parameters: The dictionary of parameters needed for
        the action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        # Use fft_size from the YAML if present, otherwise the constant default
        self.parameters.setdefault(FFT_SIZE, DEFAULT_FFT_SIZE)
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
                FFT_SIZE,
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

        # Remove IIR parameters which aren't needed after filter generation
        for key in [
            IIR_GPASS,
            IIR_GSTOP,
            IIR_PB_EDGE,
            IIR_SB_EDGE,
        ]:
            self.parameters.pop(key)

        # Use data_products from the YAML if present, otherwise the constant default
        self.parameters.setdefault(DATA_PRODUCTS, ALL_DATA_PRODUCTS)
        self.parameters[DATA_PRODUCTS] = tuple(self.parameters[DATA_PRODUCTS])

        # Get iterable parameter list
        if CAPTURES in self.parameters:
            self.iteration_params = []
            for grouped_captures in self.parameters[CAPTURES]:
                grouped_captures_parameters = self.parameters.copy()
                grouped_captures_parameters[FREQUENCY] = grouped_captures[FREQUENCY]
                if FFT_SIZE in grouped_captures:
                    grouped_captures_parameters[FFT_SIZE] = grouped_captures[FFT_SIZE]
                if SAMPLE_RATE in grouped_captures:
                    grouped_captures_parameters[SAMPLE_RATE] = grouped_captures[SAMPLE_RATE]
                if DURATION_MS in grouped_captures:
                    grouped_captures_parameters[DURATION_MS] = grouped_captures[DURATION_MS]
                if DATA_PRODUCTS in grouped_captures:
                    grouped_captures_parameters[DATA_PRODUCTS] = tuple(grouped_captures[DATA_PRODUCTS])
                del grouped_captures_parameters[CAPTURES]
                self.update_iir_parameters(grouped_captures_parameters)
                # Using original frequency sequence in YAML, do not re-sort
                self.iteration_params.extend(utils.get_iterable_parameters(grouped_captures_parameters, None))
        else:
            self.update_iir_parameters(self.parameters)
            # Using original frequency sequence in YAML, do not re-sort
            self.iteration_params = utils.get_iterable_parameters(self.parameters, None)

    def build_constants(self) -> dict:
        """Data product constants owned by this action, passed to IQ Processor as a dict."""
        return {
            "FFT_PERCENTILES": FFT_PERCENTILES,
            "FFT_WINDOW_TYPE": FFT_WINDOW_TYPE,
            "FFT_WINDOW_CORRECTION_TYPE": FFT_WINDOW_CORRECTION_TYPE,
            "FFT_M3_DETECTOR": FFT_M3_DETECTOR,
            "TD_DETECTOR": TD_DETECTOR,
            "PFP_M3_DETECTOR": PFP_M3_DETECTOR,
            "PFP_FRAME_RESOLUTION_S": PFP_FRAME_RESOLUTION_S,
            "IMPEDANCE_OHMS": IMPEDANCE_OHMS,
        }

    def __call__(self, sensor: Sensor, schedule_entry: dict, task_id: int):
        """This is the entrypoint function called by the scheduler."""
        self._sensor = sensor
        action_start_tic = perf_counter()
        # Ray should have already been initialized within scos-sensor,
        # but check and initialize just in case.

        if not ray.is_initialized():
            logger.info("Initializing ray.")
            logger.info("Set RAY_INIT=true to avoid initializing within " + __name__)
            # Dashboard is only enabled if ray[default] is installed
            ray.init()

        _ = psutil.cpu_percent(interval=None)  # Initialize CPU usage monitor
        self.test_required_components()
        self.configure_preselector(self.rf_path)

        # Initialize metadata objects, each channel has one SigMF
        sigmf_builders = []

        # Initialize remote supervisor actors for IQ processing
        tic = perf_counter()
        # Each supervisor spawns its own data_product workers; constants are identical
        # for every channel.
        constants = self.build_constants()
        iq_processors = [
            IQProcessorProductActors.remote(actor_id)
            for actor_id in range(NUM_ACTORS)
        ]
        toc = perf_counter()
        logger.debug(f"Spawned {NUM_ACTORS} supervisor actors in {toc - tic:.2f} s")

        # Capture each channel and dispatch its IQ data while later captures proceed.
        data_products_refs, cpu_speed, reference_points = [], [], []
        capture_tic = perf_counter()
        #self.total_channel_data_length = 0

        for i, parameters in enumerate(self.iteration_params):
            parameters["recording_id"] = i
            measurement_result = self.capture_iq(parameters)

            # Start data product processing but do not block next IQ capture
            tic = perf_counter()
            data_products_refs.append(
                iq_processors[i % NUM_ACTORS]
                .run.options(num_returns=len(parameters[DATA_PRODUCTS]))
                .remote(measurement_result["data"], parameters, parameters[IIR_SOS], constants)
            )

            del measurement_result["data"]
            toc = perf_counter()
            logger.debug(f"IQ data delivered for processing in {toc - tic:.2f} s")

            # Each channel has its own sigmf with recording id
            sigmf_builder = self.create_sigmf_builder(
                task_id,
                schedule_entry,
                parameters[SAMPLE_RATE],
                num_channels=1,
                recording=i,
            )
            self.total_channel_data_length = 0
            # Create capture segment with channel-specific metadata before sigan is reconfigured
            self.create_capture_segment(sigmf_builder, 0, measurement_result)
            sigmf_builders.append(sigmf_builder)

            # Query CPU speed for later averaging in diagnostics metadata
            cpu_speed.append(get_current_cpu_clock_speed())
            # Append list of data reference points; later we require these to be identical
            reference_points.append(measurement_result["reference"])
        capture_toc = perf_counter()
        logger.debug(
            f"Collected all IQ data and started all processing in {capture_toc - capture_tic:.2f} s"
        )

        # One channel's metadata can represent the result only if every capture
        # used the same reference point.
        assert (
                len(set(reference_points)) == 1
        ), "Channel data were scaled to different reference points. Cannot build metadata."

        result_tic = perf_counter()
        channel_count = len(data_products_refs)
        logger.debug(f"Have {channel_count} channel results")
        for index, parameters in enumerate(self.iteration_params):
            logger.debug(f"Working on channel {index}")
            all_data, max_max_ch_pwrs, med_mean_ch_pwrs, mean_ch_pwrs, median_ch_pwrs = (
                [],
                [],
                [],
                [],
                [],
            )
            channel_data = []
            channel_data_length = 0

            tic = perf_counter()
            # Now block until the data is ready
            product_refs = ray.get(data_products_refs[index])
            if product_refs:
                _, pending_product_refs = ray.wait(
                    list(product_refs), num_returns=len(product_refs)
                )
                if pending_product_refs:
                    raise RuntimeError(
                        f"Ray left data_product refs pending for channel {index}."
                    )

            sigmf_builder = sigmf_builders[index]
            self.create_global_sensor_metadata(sigmf_builder, self.sensor)

            channel_graph = []
            channel_dft = None
            # Product refs follow DATA_PRODUCTS order; zip preserves each result's name.
            for product_name, product_ref in zip(parameters[DATA_PRODUCTS], product_refs):
                data, summaries, graph_objects = ray.get(product_ref)
                product_length, product_total_length, product_graph, dft_obj = graph_objects
                product_graph.reference = reference_points[index]
                channel_graph.append(product_graph)

                if product_name == "psd":
                    if dft_obj is None:
                        raise RuntimeError(
                            "PSD data_product result is missing DFT data"
                        )
                    else:
                        channel_dft = dft_obj

                if product_name == "pvt":
                    if summaries is None:
                        raise RuntimeError(
                            "PVT data_product result is missing summary data"
                        )
                    max_max_ch_pwrs.append(DATA_TYPE(summaries[0]))
                    med_mean_ch_pwrs.append(DATA_TYPE(summaries[1]))
                    mean_ch_pwrs.append(DATA_TYPE(summaries[2]))
                    median_ch_pwrs.append(DATA_TYPE(summaries[3]))

                    sigmf_builder.set_max_of_max_channel_powers(max_max_ch_pwrs)
                    sigmf_builder.set_median_of_mean_channel_powers(med_mean_ch_pwrs)
                    sigmf_builder.set_mean_channel_powers(mean_ch_pwrs)
                    sigmf_builder.set_median_channel_powers(median_ch_pwrs)

                channel_data.extend(data)
                channel_data_length += product_total_length

            toc = perf_counter()
            logger.debug(f"Waited {toc - tic} s for channel data")
            all_data.extend(self.transform_data(channel_data))

            # Build metadata and convert data to compressed bytes
            all_data = self.compress_bytes_data(np.array(all_data).tobytes())

            self.create_processing_metadata(sigmf_builder, parameters, channel_dft)
            sigmf_builder.set_data_products(channel_graph)

            logger.debug(f"Channel data product length: {channel_data_length}")

            # Get diagnostics last to record action runtime
            if sensor.preselector is not None:
                self.capture_diagnostics(
                    sigmf_builder, self.sensor, action_start_tic, cpu_speed
                )  # Add diagnostics to metadata

            measurement_action_completed.send(
                sender=self.__class__,
                task_id=task_id,
                data=all_data,
                metadata=sigmf_builder.metadata,
            )

            sigmf_builder = None

        for ray_actor in iq_processors:
            ray.kill(ray_actor)
        result_toc = perf_counter()
        del iq_processors, channel_data
        logger.debug(f"Got all processed data in {result_toc - result_tic:.2f} s")

    def update_iir_parameters(self, params: dict):
        "Update iir parameters for each channel"
        sample_rate_Hz = params[SAMPLE_RATE]
        # Construct IIR filter
        iir_sos = generate_elliptic_iir_low_pass_filter(
            self.iir_gpass_dB,
            self.iir_gstop_dB,
            self.iir_pb_edge_Hz,
            self.iir_sb_edge_Hz,
            sample_rate_Hz,
        )
        iir_numerators, iir_denominators = sos2tf(sos=iir_sos)

        params[IIR_NUMERATORS] = iir_numerators
        params[IIR_DENOMINATORS] = iir_denominators
        params[IIR_SOS] = iir_sos

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
        measurement_result = self.sensor.acquire_time_domain_samples(
            num_samples, nskip, cal_params=params
        )
        # Store some metadata with the IQ
        measurement_result.update(params)
        toc = perf_counter()
        logger.debug(
            f"IQ Capture ({duration_ms} ms @ {(params[FREQUENCY] / 1e6):.1f} MHz) completed in {toc - tic:.2f} s."
        )
        return measurement_result

    def capture_diagnostics(
            self, sigmf_builder: SigMFBuilder, sensor: Sensor, action_start_tic: float, cpu_speeds: list
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

        Disk health check collects SMART diagnostics on the
        SSD specified by the `SSD_DEVICE` environment variable. This
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
                self.sensor.start_time
            )
        except:
            logger.warning("Failed to get SCOS start time")
        try:  # SCOS uptime
            cpu_diag["software_uptime"] = get_days_up(self.sensor.start_time)
        except:
            logger.warning("Failed to get SCOS uptime")
        try:  # SSD SMART data
            smart_data = get_disk_smart_data(SSD_DEVICE)
            cpu_diag["ssd_smart_data"] = ntia_diagnostics.SsdSmartData(**smart_data)
        except:
            logger.warning("Failed to get SSD SMART data")
        try:  # Disk usage
            disk_usage = get_disk_usage()
            cpu_diag["disk_usage"] = disk_usage
        except:
            logger.warning("Failed to get disk usage")

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
        logger.debug(f"Got all diagnostics in {toc - tic} s")
        diagnostics = {
            "datetime": utils.get_datetime_str_now(),
            "preselector": ntia_diagnostics.Preselector(**ps_diag),
            "spu": ntia_diagnostics.SPU(**switch_diag),
            "computer": ntia_diagnostics.Computer(**cpu_diag),
            "software": ntia_diagnostics.Software(**software_diag),
        }

        # Add diagnostics to SigMF global object
        sigmf_builder.set_diagnostics(ntia_diagnostics.Diagnostics(**diagnostics))

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

    def create_global_sensor_metadata(self, sigmf_builder: SigMFBuilder, sensor: Sensor):
        # Add (minimal) ntia-sensor metadata to the sigmf_builder:
        #   sensor ID, serial numbers for preselector, sigan, and computer
        #   overall sensor_spec version, e.g. "Prototype Rev. 3"
        #   sensor definition hash, to link to full sensor definition
        sigmf_builder.set_sensor(
            ntia_sensor.Sensor(
                sensor_spec=ntia_core.HardwareSpec(
                    id=sensor.capabilities["sensor"]["sensor_spec"]["id"],
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

    def create_processing_metadata(self, sigmf_builder: SigMFBuilder, params: dict,
                                   dft_obj: Optional[ntia_algorithm.DFT]) -> None:
        iir_obj = ntia_algorithm.DigitalFilter(
            id="iir_1",
            filter_type=ntia_algorithm.FilterType.IIR,
            feedforward_coefficients=params[IIR_NUMERATORS].tolist(),
            feedback_coefficients=params[IIR_DENOMINATORS].tolist(),
            attenuation_cutoff=self.iir_gstop_dB,
            frequency_cutoff=self.iir_sb_edge_Hz,
            description="5 MHz lowpass filter used as complex 10 MHz bandpass for channelization",
        )
        sigmf_builder.set_processing([iir_obj.id])

        if dft_obj is not None:
            sigmf_builder.set_processing_info([iir_obj, dft_obj])
        else:
            sigmf_builder.set_processing_info([iir_obj])

    def create_capture_segment(
            self,
            sigmf_builder: SigMFBuilder,
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
                datetime=self.sensor.sensor_calibration_data["datetime"],
                gain=round(measurement_result["applied_calibration"]["gain"], 3),
                noise_figure=round(
                    measurement_result["applied_calibration"]["noise_figure"], 3
                ),
                temperature=round(
                    self.sensor.sensor_calibration_data["temperature"], 1
                ),
                reference=measurement_result["reference"],
            ),
            sigan_settings=ntia_sensor.SiganSettings(
                reference_level=self.sensor.signal_analyzer.reference_level,
                attenuation=self.sensor.signal_analyzer.attenuation,
                preamp_enable=self.sensor.signal_analyzer.preamp_enable,
            ),
        )
        if "compression_point" in measurement_result["applied_calibration"]:
            capture_segment.sensor_calibration.compression_point = measurement_result[
                "applied_calibration"
            ]["compression_point"]

        sigmf_builder.add_capture(capture_segment)

    def create_sigmf_builder(
            self,
            task_id: int,
            schedule_entry: dict,
            sample_rate_Hz: float,
            num_channels: int,
            recording: Optional[int] = None,
    ) -> SigMFBuilder:
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
        sigmf_builder.set_task(task_id)
        sigmf_builder.set_classification("Unclassified")

        sigmf_builder.set_sample_rate(sample_rate_Hz)
        sigmf_builder.set_num_channels(num_channels)

        # Set the recording, if provided
        if recording is not None:
            sigmf_builder.set_recording(recording)

        return sigmf_builder

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
