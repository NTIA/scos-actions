"""Set the preselector to noise diode off"""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware import preselector

logger = logging.getLogger(__name__)


class EnableNoiseDiodeOff(Action):
    """Set the preselector to noise diode off"""

    def __init__(self, sigan):
        super().__init__(parameters={'name': 'enable_noise_diode_off'}, sigan=sigan)

    def __call__(self, schedule_entry_json, task_id):
        logger.debug("Setting noise diode OFF")
        preselector.set_state('noise_diode_off')
