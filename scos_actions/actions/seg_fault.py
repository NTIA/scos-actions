"""A simple action to create a segfault."""
import ctypes
import logging
from typing import Optional

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor

logger = logging.getLogger(__name__)

LOGLVL_INFO = 20
LOGLVL_ERROR = 40


class SegFault(Action):
    """Raise a runtime exception".

    This is useful for testing and debugging.

    """

    def __init__(self, loglvl=LOGLVL_INFO):
        super().__init__(parameters={"name": "RuntimeException"})
        self.loglvl = loglvl

    def __call__(self, sensor: Optional[Sensor], schedule_entry: dict, task_id: int):
        i = ctypes.c_char('a')
        j = ctypes.pointer(i)
        c = 0
        while True:
                j[c] = 'a'
                c += 1
        j
