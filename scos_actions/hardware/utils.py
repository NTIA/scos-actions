import logging

import psutil

from scos_actions.hardware import switches
from scos_actions.hardware.hardware_configuration_exception import (
    HardwareConfigurationException,
)
from scos_actions.settings import SIGAN_POWER_CYCLE_STATES, SIGAN_POWER_SWITCH

logger = logging.getLogger(__name__)


def get_base_cpu_clock_speed_kHz(policy_num: int = 0) -> float:
    """
    Get the base clock speed, in kHz, of the CPU running SCOS.

    By default, the base clock speed is queried from:
    ``/sys/devices/system/cpu/cpufreq/policy0/base_frequency``

    The ``policy_num`` parameter allows selection of a specific cpufreq
    policy, if desired. Typically, all cores will have the same base
    clock speed, and it is not necessary to specify this parameter.

    :param policy_num: The cpufreq policy number for which to query
        the base clock speed. Defaults to 0.
    :return: The base CPU clock speed in kHz.
    """
    try:
        with open(
            f"/sys/devices/system/cpu/cpufreq/policy{policy_num}/base_frequency"
        ) as f:
            base_clock_speed_kHz = float(f.readline())
        return base_clock_speed_kHz
    except Exception as e:
        logger.error("Unable to read base CPU clock speed from system.")
        raise e


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


def get_current_cpu_clock_speeds_MHz() -> list:
    """
    Get current speeds of each logical core, in MHz, on the CPU running SCOS.

    The current clock speeds are pulled from ``/proc/cpuinfo``.

    :return: The clock speeds of each logical CPU, in MHz. Ordering
        is preserved from ``/proc/cpuinfo``.
    """
    try:
        with open("/proc/cpuinfo") as f:
            s = [float(k.split(": ")[1]) for k in f.readlines() if "cpu MHz" in k]
        return s
    except Exception as e:
        logger.error("Unable to read current CPU clock speeds from system.")
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


def get_max_cpu_clock_speed_kHz(policy_num: int = 0) -> float:
    """
    Get the maximum clock speed, in kHz, of the CPU running SCOS.

    By default, the maximum clock speed is queried from:
    ``/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq``

    The maximum clock speed typically corresponds to the "turbo boost"
    speed of a CPU.

    The ``policy_num`` parameter allows selection of a specific cpufreq
    policy, if desired. Typically, all cores will have the same maximum
    clock speed, and it is not necessary to specify this parameter.

    :param policy_num: The cpufreq policy number for which to query
        the maximum clock speed. Defaults to 0.
    :return: The maximum CPU clock speed in kHz.
    """
    try:
        with open(
            f"/sys/devices/system/cpu/cpufreq/policy{policy_num}/scaling_max_freq"
        ) as f:
            max_clock_speed_kHz = float(f.readline())
        return max_clock_speed_kHz
    except Exception as e:
        logger.error("Unable to read max. CPU clock speed from system.")
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
