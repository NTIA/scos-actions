"""Monitor the signal analyzer."""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.sensor import Sensor
from scos_actions.signals import trigger_api_restart

logger = logging.getLogger(__name__)


class MonitorSignalAnalyzer(Action):
    """Monitor signal analyzer connection and restart container if unreachable."""

    def __init__(self, parameters={"name": "monitor_sigan"}, gps=None):
        super().__init__(parameters=parameters)

    def __call__(self, sensor: Sensor, schedule_entry: dict, task_id: int):
        logger.debug("Performing signal analyzer health check")
        self._sensor = sensor
        healthy = self.sensor.signal_analyzer.healthy()

        if healthy:
            logger.info("signal analyzer healthy")
        else:
            logger.warning("signal analyzer unhealthy")
            trigger_api_restart.send(sender=self.__class__)
