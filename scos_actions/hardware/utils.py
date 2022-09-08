import logging

from scos_actions.hardware import switches
from scos_actions.hardware.hardware_configuration_exception import (
    HardwareConfigurationException,
)
from scos_actions.settings import SIGAN_POWER_CYCLE_STATES, SIGAN_POWER_SWITCH

logger = logging.getLogger(__name__)


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
