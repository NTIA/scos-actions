from typing import Optional

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class Preselector(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `Preselector` objects.

    :param temp: Temperature inside the preselector enclosure, in degrees Celsius.
    :param noise_diode_temp: Temperature of the noise diode, in degrees Celsius.
    :param lna_temp: Temparature of the low noise amplifier, in degrees Celsius.
    :param humidity: Relative humidity inside the preselector enclosure, as a percentage.
    :param door_closed: Indicates whether the door of the enclosure is closed.
    """

    temp: Optional[float] = None
    noise_diode_temp: Optional[float] = None
    lna_temp: Optional[float] = None
    humidity: Optional[float] = None
    door_closed: Optional[bool] = False


class SPU(
    msgspec.Struct, rename={"aux_28v_powered": "28v_aux_powered"}, **SIGMF_OBJECT_KWARGS
):
    """
    Interface for generating `ntia-diagnostics` `SPU` objects.

    :param rf_tray_powered: Indicates if the RF tray is powered.
    :param preselector_powered: Indicates if the preselector is powered.
    :param aux_28v_powered: Indicates if the 28V aux power is on.
    :param pwr_box_temp: Ambient temperature in power distribution box,
        in degrees Celsius.
    :param pwr_box_humidity: Humidity in power distribution box, as a
        percentage.
    :param rf_box_temp: Ambient temperature in the RF box (around the signal
        analyzer), in degrees Celsius.
    :param sigan_internal_temp: Internal temperature reported by the signal analyzer.
    """

    rf_tray_powered: Optional[bool] = None
    preselector_powered: Optional[bool] = None
    aux_28v_powered: Optional[bool] = None
    pwr_box_temp: Optional[float] = None
    pwr_box_humidity: Optional[float] = None
    rf_box_temp: Optional[float] = None
    sigan_internal_temp: Optional[float] = None


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


class Software(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-diagnostics` `Software` objects.

    :param system_platform: The underlying platform, as returned by `platform.platform()`
    :param python_version: The Python version, as returned by `sys.version()`.
    :param scos_actions_version: Version of `scos_actions` plugin.
    :param preselector_api_version: Version of the NTIA `preselector` package.
    """

    system_platform: Optional[str] = None
    python_version: Optional[str] = None
    scos_actions_version: Optional[str] = None
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
