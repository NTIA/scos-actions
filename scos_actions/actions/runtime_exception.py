"""A simple example action that logs a message."""

import logging
from typing import Optional

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor

logger = logging.getLogger(__name__)

LOGLVL_INFO = 20
LOGLVL_ERROR = 40


class RuntimeException(Action):
    """Raise a runtime exception".

    This is useful for testing and debugging.

    """

    def __init__(self, loglvl=LOGLVL_INFO):
        super().__init__(parameters={"name": "RuntimeException"})
        self.loglvl = loglvl

    def __call__(self, sensor: Optional[Sensor], schedule_entry: dict, task_id: int):
        msg = "Raising runtime exception {name}/{tid}"
        schedule_entry_name = schedule_entry["name"]
        logger.log(
            level=self.loglvl, msg=msg.format(name=schedule_entry_name, tid=task_id)
        )
        raise RuntimeError("RuntimeError from RuntimeErrorAction")
