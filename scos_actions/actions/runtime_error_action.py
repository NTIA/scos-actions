"""A simple example action that raises a RuntimeError."""

import logging
from typing import Optional

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor

logger = logging.getLogger(__name__)


class RuntimeErrorAction(Action):
    """
    Raise a runtime error.
    This is useful for testing and debugging.

    """

    def __init__(self):
        super().__init__(parameters={"name": "RuntimeErrorAction"})

    def __call__(self, sensor: Optional[Sensor], schedule_entry: dict, task_id: int):
        msg = "Raising RuntimeError {name}/{tid}"
        schedule_entry_name = schedule_entry["name"]
        logger.log(msg=msg.format(name=schedule_entry_name, tid=task_id))
        raise RuntimeError("RuntimeError from RuntimeErrorAction")
