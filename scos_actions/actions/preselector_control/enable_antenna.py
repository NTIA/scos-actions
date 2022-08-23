"""Set the preselector to the antenna"""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware import preselector

logger = logging.getLogger(__name__)


class EnableAntenna(Action):
    """Set the preselector to the antenna"""

    def __init__(self, sigan):
        super().__init__(parameters={"name": "enable_antenna"}, sigan=sigan)

    def __call__(self, schedule_entry_json, task_id):
        logger.debug("Enabling antenna")
        preselector.set_state("antenna")
