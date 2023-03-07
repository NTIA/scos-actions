import logging
import subprocess

import psutil

from scos_actions.hardware import switches
from scos_actions.hardware.hardware_configuration_exception import (
    HardwareConfigurationException,
)
from scos_actions.settings import SIGAN_POWER_CYCLE_STATES, SIGAN_POWER_SWITCH

logger = logging.getLogger(__name__)


def get_cpu_overheating() -> bool:
    """
    Get a boolean indicating whether the CPU running SCOS is overheating.

    Only Intel CPUs are currently supported.

    :return: True if the current CPU temperature is above its maximum,
        False otherwise.
    """
    return get_current_cpu_temperature() > get_max_cpu_temperature()


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


def get_disk_smart_healthy_status(disk: str) -> bool:
    """
    Get the result of the SMART overall-assessment health test for the chosen disk.

    This method requires that ``smartmontools`` be installed on the host OS.

    :param disk: The desired disk, e.g., ``/dev/sda1``.
    :raises RuntimeError: If the ``smartctl`` call fails and disk health cannot be
        determined.
    :return: True if the SMART overall-health self-assessment health test is passed.
        False otherwise.
    """
    try:
        result = subprocess.run(["smartctl", "-H", disk], capture_output=True)
        if result.returncode == 0:
            return "PASSED" in str(result.stdout)
        else:
            logger.error(
                f"Call to smartctl failed with return code {result.returncode}"
            )
            raise RuntimeError(str(result.stdout))
    except Exception as e:
        logger.error(f"Unable to get SMART health test result for disk {disk}")
        raise e


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
