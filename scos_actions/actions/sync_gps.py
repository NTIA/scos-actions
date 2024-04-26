"""Monitor the on-board USRP and touch or remove an indicator file."""

import logging
import subprocess

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor
from scos_actions.signals import location_action_completed

logger = logging.getLogger(__name__)


class SyncGps(Action):
    """Query the GPS and synchronize time and location."""

    def __init__(self, parameters: dict):
        super().__init__(parameters=parameters)

    def __call__(self, sensor: Sensor, schedule_entry: dict, task_id: int):
        logger.debug("Syncing to GPS")
        self.sensor = sensor
        dt = self.sensor.gps.get_gps_time(self.sensor)
        date_cmd = ["date", "-s", "{:}".format(dt.strftime("%Y/%m/%d %H:%M:%S"))]
        subprocess.check_output(date_cmd, shell=True)
        logger.info(f"Set system time to GPS time {dt.ctime()}")

        location = sensor.gps.get_location(self.sensor)
        if location is None:
            raise RuntimeError("Unable to synchronize to GPS")

        latitude, longitude, height = location
        location_action_completed.send(
            self.__class__,
            latitude=latitude,
            longitude=longitude,
            height=height,
            gps=True,
        )
