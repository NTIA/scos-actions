import logging

from scos_actions.hardware import switches
from scos_actions.settings import SIGAN_POWER_CYCLE_STATES, SIGAN_POWER_SWITCH

logger = logging.getLogger(__name__)


def power_cycle_sigan():
    """
    Performs a hard power cycle of the signal analyzer. This method requires power to the signal analyzer is
    controlled by a Web_Relay (see https://www.github.com/ntia/Preselector) and that the switch id of that
    switch is specified in scos-sensor settings as SIGAN_POWER_SWITCH and the sequence of states is specified as
    a comma delimited list of states in SIGAN_POWER_CYCLE_STATES. This method will raise Excwptions if the nn
    """
    if SIGAN_POWER_SWITCH and SIGAN_POWER_CYCLE_STATES:
        for switch in switches:
            if switch.id == SIGAN_POWER_SWITCH:
                power_switch = switch
                break
        if power_switch is None:
            raise Exception(
                "Switch {switch_id} does not exist. Unable to restart signal analyzer"
            )
        else:
            if SIGAN_POWER_CYCLE_STATES is None:
                raise Exception("SIGAN_POWER_CYCLE_STATES not specified in settings")
            else:
                states = SIGAN_POWER_CYCLE_STATES.split(",")
                for state in states:
                    power_switch.set_state(state)
    else:
        logger.error(
            "Call to power cycle sigan, but no power switch or power cycle states specified "
        )
