from typing import Optional, List

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class Preselector(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `Preselector` objects.

    :param temp: Temperature inside the preselector enclosure, in degrees Celsius.
    :param noise_diode_temp: Temperature of the noise diode, in degrees Celsius.
    :param noise_diode_powered: Boolean indicating if the noise diode is powered.
    :param lna_powered: Boolean indicating if the lna is powered.
    :param lna_temp: Temparature of the low noise amplifier, in degrees Celsius.
    :param antenna_path_enabled: Boolean indicating if the antenna path is enabled.
    :param noise_diode_path_enabled: Boolean indicating if the noise diode path is enabled.
    :param humidity: Relative humidity inside the preselector enclosure, as a percentage.
    :param door_closed: Indicates whether the door of the enclosure is closed.
    """

    temp: Optional[float] = None
    noise_diode_temp: Optional[float] = None
    noise_diode_powered: Optional[bool] = None
    lna_powered: Optional[bool] = None
    lna_temp: Optional[float] = None
    antenna_path_enabled: Optional[bool] = None
    noise_diode_path_enabled: Optional = None
    humidity: Optional[float] = None
    door_closed: Optional[bool] = False

class DiagnosticSensor(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `DiagnosticSensor` objects.

    :param name: The name of the sensor
    :param value: The value provided by the sensor
    :param maximum_allowed: The maximum value allowed from the sensor before action should be taken
    :param mimimum_allowed: The minimum value allowed from the sensor before action should be taken
    :param description: A description of the sensor
    """
    name: str
    value: float
    maximum_allowed: Optional[float] = None
    minimum_allowed: Optional[float] = None
    expected_value: Optional[float] = None
    description: Optional[str] = None

class SPU(
    msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `SPU` objects.

    :param cooling: Boolean indicating if the cooling is enabled.
    :param heating: Boolean indicating if the heat is enabled.
    :param preselector_powered: Indicates if the preselector is powered.
    :param sigan_powered: Boolean indicating if the signal analyzer is powered.
    :param temperature_control_powered: Boolean indicating TEC AC power.
    :param battery_backup: Boolean indicating if it is running on battery backup.
    :param low_battery: Boolean indicating if the battery is low.
    :param ups_healthy: Indicates trouble with UPS.
    :param replace_battery: Boolean indicating if the ups battery needs replacing.
    :param temperature_sensors: List of temperature sensor values
    :param humidity_sensors: List of humidity sensor values
    :param power_sensors: List of power sensor values
    :param door_closed: Boolean indicating if the door is closed
    """
    cooling: Optional[bool] = None
    heating: Optional[bool] = None
    sigan_powered: Optional[bool] = None
    temperature_control_powered: Optional[bool] = None
    preselector_powered: Optional[bool] = None

    battery_backup: Optional[bool] = None
    low_battery: Optional[bool] = None
    ups_healthy: Optional[bool] = None
    replace_battery: Optional[bool] = None

    temperature_sensors: Optional[List[DiagnosticSensor]] = None
    humidity_sensors: Optional[List[DiagnosticSensor]] = None
    power_sensors: Optional[List[DiagnosticSensor]] = None
    door_closed: Optional[bool] = None


class SsdSmartData(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `SsdSmartData` objects.

    :param test_passed: SMART overall-health self-assessment test result.
    :param critical_warning: Critical warning message from `smartctl`, a
        string containing a hexadecimal value, e.g. `"0x00"`.
    :param temp: Drive temperature, in degrees Celsius.
    :param available_spare: Normalized percentage (0 to 100) of the remaining
        spare capacity available.
    :param available_spare_threshold: When the `available_spare` falls below
        this threshold, an aynchronous event completion may occur. Indicated as
        a normalized percentage (0 to 100).
    :param percentage_used: Contains a vendor specific estimate of the percentage
        of NVM subsystem life used based on the actual usage and the manufacturer’s
        prediction of NVM life. A value of 100 indicates that the estimated endurance
        of the NVM in the NVM subsystem has been consumed, but may not indicate an NVM
        subsystem failure. Values may exceed 100 and percentages greater than 254 shall
        be represented as 255.
    :param unsafe_shutdowns: Number of unsafe shutdowns.
    :param integrity_errors: Number of occurrences where the controller detected an
        unrecovered data integrity error.
    """

    test_passed: Optional[bool] = None
    critical_warning: Optional[str] = None
    temp: Optional[float] = None
    available_spare: Optional[float] = None
    available_spare_threshold: Optional[float] = None
    percentage_used: Optional[float] = None
    unsafe_shutdowns: Optional[int] = None
    integrity_errors: Optional[int] = None


class Computer(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `Computer` objects.

    :param cpu_min_clock: Minimum sampled clock speed, in MHz.
    :param cpu_max_clock: Maximum sampled clock speed, in MHz.
    :param cpu_mean_clock: Mean sampled clock speed, in MHz.
    :param cpu_uptime: Number of days since the computer started.
    :param action_cpu_usage: CPU utilization during action execution, as a percentage.
    :param system_load_5m: Number of processes in a runnable state over the
        previous 5 minutes as a percentage of the number of CPUs.
    :param memory_usage: Average percent of memory used during action execution.
    :param cpu_overheating: Whether the CPU is overheating.
    :param cpu_temp: CPU temperature, in degrees Celsius.
    :param scos_start: The time at which the SCOS API container started. Must be
        an ISO 8601 formatted string.
    :param scos_uptime: Number of days since the SCOS API container started.
    :param ssd_smart_data: Information provided by the drive Self-Monitoring,
        Analysis, and Reporting Technology.
    """

    cpu_min_clock: Optional[float] = None
    cpu_max_clock: Optional[float] = None
    cpu_mean_clock: Optional[float] = None
    cpu_uptime: Optional[float] = None
    action_cpu_usage: Optional[float] = None
    system_load_5m: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_overheating: Optional[bool] = None
    cpu_temp: Optional[float] = None
    software_start: Optional[str] = None
    software_uptime: Optional[float] = None
    ssd_smart_data: Optional[SsdSmartData] = None


class ScosPlugin(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `ScosPlugin` objects.

    :param name: The Python package name as it is imported, e.g., `"scos_tekrsa"`
    :param version: Version of the SCOS plugin.
    """

    name: Optional[str] = None
    version: Optional[str] = None


class Software(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `Software` objects.

    :param system_platform: The underlying platform, as returned by `platform.platform()`
    :param python_version: The Python version, as returned by `sys.version()`.
    :param scos_sensor_version: The SCOS Sensor version, as returned by `git describe --tags`.
    :param scos_actions_version: Version of `scos_actions` plugin.
    :param scos_sigan_plugin: `ScosPlugin` object describing the plugin which defines the
        signal analyzer interface.
    :param preselector_api_version: Version of the NTIA `preselector` package.
    """

    system_platform: Optional[str] = None
    python_version: Optional[str] = None
    scos_sensor_version: Optional[str] = None
    scos_actions_version: Optional[str] = None
    scos_sigan_plugin: Optional[ScosPlugin] = None
    preselector_api_version: Optional[str] = None


class Diagnostics(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `Diagnostics` objects.

    :param datetime: The time at which the diagnostics were gathered. Must be
        an ISO 8601 formatted string.
    :param preselector: Metadata to capture preselector diagnostics.
    :param spu: Metadata to capture signal processing unit diagnostics.
    :param computer: Metadata to capture computer diagnostics.
    :param action_runtime: Total action execution time, in seconds.
    """

    datetime: Optional[str] = None
    preselector: Optional[Preselector] = None
    spu: Optional[SPU] = None
    computer: Optional[Computer] = None
    software: Optional[Software] = None
    action_runtime: Optional[float] = None
