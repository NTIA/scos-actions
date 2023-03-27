import logging
import subprocess

import psutil

from scos_actions.hardware import switches
from scos_actions.hardware.hardware_configuration_exception import (
    HardwareConfigurationException,
)
from scos_actions.settings import SIGAN_POWER_CYCLE_STATES, SIGAN_POWER_SWITCH

logger = logging.getLogger(__name__)


def get_cpu_uptime_seconds() -> float:
    """
    Get the current uptime, in seconds, of the CPU running SCOS.

    The uptime is pulled from ``/proc/uptime``.

    :return: The CPU uptime, in seconds.
    """
    try:
        with open("/proc/uptime") as f:
            uptime_seconds = float(f.readline().split()[0])
        return uptime_seconds
    except Exception as e:
        logger.error("Unable to get CPU uptime from system.")
        raise e


def get_current_cpu_clock_speed() -> float:
    """
    Get the current clock speed of the CPU running SCOS.

    The clock speed is queried with ``lscpu`` and the returned
    value is in MHz.

    :return:
    """
    try:
        out = subprocess.check_output(["lscpu | grep 'MHz'"]).decode("utf-8")
        spd = [l.split()[2] for l in out.split("\n") if l.startswith("CPU MHz:")][0]
        return float(spd)
    except Exception as e:
        logger.error("Unable to retrieve current CPU speed")
        raise e


def get_current_cpu_temperature(fahrenheit: bool = False) -> float:
    """
    Get the current temperature of the CPU running SCOS.

    Only Intel CPUs are currently supported.

    :param farenheit: If True, return the temperature in degrees
        Fahrenheit instead of Celsius, defaults to False.
    :return: The current CPU temperature, in the units specified.
    """
    try:
        cpu_temp = psutil.sensors_temperatures(fahrenheit)["coretemp"][0].current
        return cpu_temp
    except Exception as e:
        logger.error("Unable to get current CPU temperature.")
        raise e


def get_disk_smart_data(disk: str) -> dict:
    """
    Get selected SMART data for the chosen disk.

    This method requires that ``smartmontools`` be installed on the host OS,
    and that the disk is properly accessible (i.e., available to the Docker
    container, if running in Docker). This method was written specifically
    for NVMe SSDs used by NASCTN SEA SPU computers, and results may vary when
    using other hardware.

    More details can be found in Figure 194 (p. 122) of the NVMe standard,
    Revision 1.4, available here (accessed 3/7/2023):
    https://nvmexpress.org/wp-content/uploads/NVM-Express-1_4-2019.06.10-Ratified.pdf

    :param disk: The desired disk, e.g., ``/dev/nvme0n1``.
    :return: A dictionary containing the retrieved data from the SMART report.
    """
    try:
        report = subprocess.check_output(["smartctl", "-a", disk]).decode("utf-8")
    except Exception:
        logger.exception(f"Unable to get SMART data for disk {disk}")
        return "Unavailable"
    disk_info = {}
    for line in report.split("\n"):
        if line.startswith("SMART overall-health self-assessment test result:"):
            disk_info["test_passed"] = "PASSED" in line.split()[5]
        elif line.startswith("Critical Warning:"):
            disk_info["critical_warning"] = line.split()[2]
        elif line.startswith("Temperature:"):
            disk_info["temperature_degC"] = int(line.split()[1])
        elif line.startswith("Available Spare:"):
            disk_info["available_spare"] = int(line.split()[2].rstrip("%"))
        elif line.startswith("Available Spare Threshold:"):
            disk_info["available_spare_threshold"] = int(line.split()[3].rstrip("%"))
        elif line.startswith("Percentage Used:"):
            disk_info["percentage_used"] = int(line.split()[2].rstrip("%"))
        elif line.startswith("Unsafe Shutdowns:"):
            disk_info["unsafe_shutdown_count"] = int(line.split()[2])
        elif line.startswith("Media and Data Integrity Errors:"):
            disk_info["media_data_integrity_error_count"] = int(line.split()[5])
    return disk_info


def get_max_cpu_temperature(fahrenheit: bool = False) -> float:
    """
    Get the maximum temperature of the CPU running SCOS.

    Only Intel CPUs are currently supported.

    :param fahrenheit: If True, return the temperature in degrees
        Fahrenheit instead of Celsius, defaults to False.
    :return: The maximum safe CPU temperature, in the units specified.
    """
    try:
        max_temp = psutil.sensors_temperatures(fahrenheit)["coretemp"][0].high
        return max_temp
    except Exception as e:
        logger.error("Unable to get maximum CPU temperature.")
        raise e


def power_cycle_sigan():
    """
    Performs a hard power cycle of the signal analyzer. This method requires power to the signal analyzer is
    controlled by a Web_Relay (see https://www.github.com/ntia/Preselector) and that the switch id of that
    switch is specified in scos-sensor settings as SIGAN_POWER_SWITCH and the sequence of states is specified as
    a comma delimited list of states in SIGAN_POWER_CYCLE_STATES. This method will raise HardwareConfigurationException
    if there are no switches configured, the switch specified to control power to the signal analyzer doesn't exist, or
    if either SIGAN_POWER_SWITCH or SIGAN_POWER_CYCLE_STATES are not set.
    """
    if switches is None:
        raise HardwareConfigurationException(
            "No switches are configured. Unable to power cycle signal analyzer "
        )

    if SIGAN_POWER_SWITCH and SIGAN_POWER_CYCLE_STATES:
        logger.debug(f"searching for {SIGAN_POWER_SWITCH}")
        if SIGAN_POWER_SWITCH in switches:
            power_switch = switches[SIGAN_POWER_SWITCH]
            if SIGAN_POWER_CYCLE_STATES is None:
                raise HardwareConfigurationException(
                    "SIGAN_POWER_CYCLE_STATES not specified in settings"
                )
            else:
                states = SIGAN_POWER_CYCLE_STATES.split(",")
                for state in states:
                    logger.debug(f"Setting state: {state} in power switch")
                    power_switch.set_state(state)
        else:
            raise HardwareConfigurationException(
                f"Switch {SIGAN_POWER_SWITCH} does not exist. Unable to restart signal analyzer"
            )
    else:
        raise HardwareConfigurationException(
            "Call to power cycle sigan, but no power switch or power cycle states specified "
        )
