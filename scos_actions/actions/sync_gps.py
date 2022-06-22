"""Monitor the on-board USRP and touch or remove an indicator file."""

import logging
import subprocess

from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import location_action_completed
from scos_actions.hardware import sigan as mock_sigan

logger = logging.getLogger(__name__)


class SyncGps(Action):
    """Query the GPS and synchronize time and location."""

    def __init__(self,gps, parameters={'name': 'SyncGps'}, sigan=mock_sigan):
        super().__init__(parameters=parameters, sigan=sigan, gps=gps)

    def execute(self, schedule_entry, task_id):
        logger.debug("Syncing to GPS")

        dt = self.gps.get_gps_time()
        date_cmd = ["date", "-s", "{:}".format(dt.strftime("%Y/%m/%d %H:%M:%S"))]
        subprocess.check_output(date_cmd, shell=True)
        logger.info("Set system time to GPS time {}".format(dt.ctime()))

        location = self.gps.get_location()
        if location is None:
            raise RuntimeError("Unable to synchronize to GPS")

        latitude, longitude, height = location
        measurement_result = {'latitude':latitude, 'longitude': longitude, 'height': height}
        return measurement_result


    def send_signals(self, measurement_result):
        location_action_completed.send(
            self.__class__,
            latitude=measurement_result['latitude'],
            longitude=measurement_result['longitude'],
            height=measurement_result['height'],
            gps=True,
        )

    def add_metadata_generators(self, measurement_result):
        pass


    def create_metadata(self, schedule_entry, measurement_result):
        pass

    def test_required_components(self):
        pass
