"""A simple action that raises SystemExit"""

import logging
from typing import Optional

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor

logger = logging.getLogger(__name__)


class SystemExitAction(Action):
    """
    Raise a SystemExit.
    This is useful for testing and debugging. Note: this action
    is currently not loaded in any scenario and must be manually
    added to use.

    """

    def __init__(self):
        super().__init__(parameters={"name": "SystemExitAction"})

    def __call__(self, sensor: Optional[Sensor], schedule_entry: dict, task_id: int):
        msg = "Raising SystemExit {name}/{tid}"
        schedule_entry_name = schedule_entry["name"]
        logger.log(msg=msg.format(name=schedule_entry_name, tid=task_id))
        raise SystemExit("SystemExit produced by SystemExitAction. ")
