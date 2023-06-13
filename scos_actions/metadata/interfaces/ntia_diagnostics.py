from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

from scos_actions.metadata.interfaces.sigmf_object import SigMFObject
from scos_actions.utils import convert_datetime_to_millisecond_iso_format


@dataclass
class Preselector(SigMFObject):
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

    def __post_init__(self):
        super().__post_init__()
        # Define SigMF key names
        self.obj_keys.update(
            {
                "temp": "temp",
                "noise_diode_temp": "noise_diode_temp",
                "lna_temp": "lna_temp",
                "humidity": "humidity",
                "door_closed": "door_closed",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class SPU(SigMFObject):
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

    def __post_init__(self):
        super().__post_init__()
        # Define SigMF key names
        self.obj_keys.update(
            {
                "rf_tray_powered": "rf_tray_powered",
                "preselector_powered": "preselector_powered",
                "aux_28v_powered": "28v_aux_powered",
                "pwr_box_temp": "pwr_box_temp",
                "pwr_box_humidity": "pwr_box_humidity",
                "rf_box_temp": "rf_box_temp",
                "sigan_internal_temp": "sigan_internal_temp",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class SsdSmartData(SigMFObject):
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

    def __post_init__(self):
        super().__post_init__()
        # Define SigMF key names
        self.obj_keys.update(
            {
                "test_passed": "test_passed",
                "critical_warning": "critical_warning",
                "temp": "temp",
                "available_spare": "available_spare",
                "available_spare_threshold": "available_spare_threshold",
                "percentage_used": "percentage_used",
                "unsafe_shutdowns": "unsafe_shutdowns",
                "integrity_errors": "integrity_errors",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class Computer(SigMFObject):
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
    :param scos_start: The time at which the SCOS API container started.
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
    scos_start: Optional[Union[datetime, str]] = None
    scos_uptime: Optional[float] = None
    ssd_smart_data: Optional[SsdSmartData] = None

    def __post_init__(self):
        super().__post_init__()
        # Convert datetime to string if needed
        if isinstance(self.scos_start, datetime):
            self.scos_start = convert_datetime_to_millisecond_iso_format(
                self.scos_start
            )
        # Define SigMF key names
        self.obj_keys.update(
            {
                "cpu_min_clock": "cpu_min_clock",
                "cpu_max_clock": "cpu_max_clock",
                "cpu_mean_clock": "cpu_mean_clock",
                "cpu_uptime": "cpu_uptime",
                "action_cpu_usage": "action_cpu_usage",
                "system_load_5m": "system_load_5m",
                "memory_usage": "memory_usage",
                "cpu_overheating": "cpu_overheating",
                "cpu_temp": "cpu_temp",
                "scos_start": "scos_start",
                "scos_uptime": "scos_uptime",
                "ssd_smart_data": "ssd_smart_data",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class Diagnostics(SigMFObject):
    """
    Interface for generating `ntia-diagnostics` `Diagnostics` objects.

    :param datetime: The time at which the diagnostics were gathered.
    :param preselector: Metadata to capture preselector diagnostics.
    :param spu: Metadata to capture signal processing unit diagnostics.
    :param computer: Metadata to capture computer diagnostics.
    :param action_runtime: Total action execution time, in seconds.
    """

    datetime: Optional[Union[datetime, str]] = None
    preselector: Optional[Preselector] = None
    spu: Optional[SPU] = None
    computer: Optional[Computer] = None
    action_runtime: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        # Convert datetime to string if needed
        if isinstance(self.datetime, datetime):
            self.datetime = convert_datetime_to_millisecond_iso_format(self.datetime)
        # Define SigMF key names
        self.obj_keys.update(
            {
                "datetime": "datetime",
                "preselector": "preselector",
                "spu": "spu",
                "computer": "computer",
                "action_runtime": "action_runtime",
            }
        )
        # Create metadata object
        super().create_json_object()
