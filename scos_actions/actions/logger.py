"""A simple example action that logs a message."""

import logging

from scos_actions.actions.interfaces.action import Action

logger = logging.getLogger(__name__)

LOGLVL_INFO = 20
LOGLVL_ERROR = 40


class Logger(Action):
    """Log the message "running test {name}/{tid}".

    This is useful for testing and debugging.

    `{name}` will be replaced with the parent schedule entry's name, and
    `{tid}` will be replaced with the sequential task id.

    """

    def __init__(self, loglvl=LOGLVL_INFO):
        super().__init__(parameters={"name": "logger"})
        self.sigan = sigan
        self.gps = gps
        self.loglvl = loglvl

    def __call__(self,sigan, gps, schedule_entry, task_id):
        msg = "running test {name}/{tid}"
        schedule_entry_name = schedule_entry["name"]
        logger.log(
            level=self.loglvl, msg=msg.format(name=schedule_entry_name, tid=task_id)
        )
