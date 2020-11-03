"""Monitor the on-board USRP and touch or remove an indicator file."""

import logging
import subprocess

from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import location_action_completed

logger = logging.getLogger(__name__)


class SyncGps(Action):
    """Query the GPS and synchronize time and location."""

    def __init__(self, gps, admin_only=True):
        super(SyncGps, self).__init__(admin_only=admin_only)

        self.gps = gps

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        logger.debug("Syncing to GPS")

        dt = self.gps.get_gps_time()
        date_cmd = ["date", "-s", "{:}".format(dt.strftime("%Y/%m/%d %H:%M:%S"))]
        subprocess.check_output(date_cmd, shell=True)
        logger.info("Set system time to GPS time {}".format(dt.ctime()))

        location = self.gps.get_lat_long()
        if location is None:
            raise RuntimeError("Unable to synchronize to GPS")

        latitude, longitude = location
        location_action_completed.send(
            self.__class__, latitude=latitude, longitude=longitude
        )
