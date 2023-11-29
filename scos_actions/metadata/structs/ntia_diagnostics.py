from typing import Optional

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


class SPU(
    msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `SPU` objects.

    :param rf_tray_powered: Indicates if the RF tray is powered.
    :param preselector_powered: Indicates if the preselector is powered.
    :param aux_28v_powered: Indicates if the 28V aux power is on.
    :param pwr_box_temp: Ambient temperature in power distribution box,
        in degrees Celsius..
    :param pwr_box_humidity: Humidity in power distribution box, as a
        percentage.
    :param rf_box_temp: Ambient temperature in the RF box (around the signal
        analyzer), in degrees Celsius.
    :param internal_temp: Ambient temperature in the SPU,
        in degrees Celsius
    :param internal_humidity: Humidity in the SPU.
    :param tec_intake_temp: Temperature at the TEC intake.
    :param tec_exhaust_temp: Temperature at the TEC exhaust.
    :param sigan_internal_temp: Internal temperature reported by the signal analyzer.
    :param cooling_enabled: Boolean indicating if the cooling is enabled.
    :param heat_enabled: Boolean indicating if the heat is enabled.
    :param rsa_powered: Boolean indicating if the RSA is powered.
    :param nuc_powered: Boolean indicating if NUC is powered.
    :param tec_ac_powered: Boolean indicating TEC AC power.
    :param ups_power: Boolean indicating UPS power.
    :param ups_battery_level: UPS batery level warning.
    :param ups_trouble: Indicates trouble with UPS.
    :param ups_battery_replace: Boolean indicating if the ups battery needs replacing.
    :param door_sensor: Indicates if the door is open.
    :param power_5vdc: 5V DC power supplied.
    :param power_15vdc: 15V DC power supplied.
    :param power_24vdc: 24V DC power supplied.
    :param power_28vdc: 28V DC power supplied.
    """

    rf_tray_powered: Optional[bool] = None
    preselector_powered: Optional[bool] = None
    aux_28v_powered: Optional[bool] = None
    pwr_box_temp: Optional[float] = None
    pwr_box_humidity: Optional[float] = None
    rf_box_temp: Optional[float] = None
    internal_temp: Optional[float] = None
    internal_humidity: Optional[float] = None
    tec_intake_temp: Optional[float] = None
    tek_exhaust_temp: Optional[float] = None
    sigan_internal_temp: Optional[float] = None
    cooling_enabled: Optional[bool] = None
    heat_enabled: Optional[bool] = None
    rsa_powered: Optional[bool] = None
    nuc_powered: Optional[bool] = None
    tec_ac_powered: Optional[bool] = None
    ups_power: Optional[bool] = None
    ups_battery_level: Optional[bool] = None
    ups_trouble: Optional[bool] = None
    ups_battery_replace: Optional[bool] = None
    door_sensor: Optional[float] = None
    power_5vdc: Optional[float] = None
    power_15vdc: Optional[float] = None
    power_24vdc: Optional[float] = None
    power_28vdc: Optional[float] = None



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
    scos_start: Optional[str] = None
    scos_uptime: Optional[float] = None
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
