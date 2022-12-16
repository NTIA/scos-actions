"""Monitor the signal analyzer."""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.signals import trigger_api_restart

logger = logging.getLogger(__name__)


class MonitorSignalAnalyzer(Action):
    """Monitor signal analyzer connection and restart container if unreachable."""

    def __init__(self, sigan, parameters={"name": "monitor_sigan"}, gps=None):
        if gps is None:
            gps = MockGPS()
        super().__init__(parameters=parameters, sigan=sigan, gps=gps)

    def __call__(self, schedule_entry_json, task_id):
        logger.debug("Performing signal analyzer health check")

        healthy = self.sigan.healthy

        if healthy:
            trigger_api_restart.send(sender=self.__class__, sigan_healthy=True)
            logger.info("signal analyzer healthy")
        else:
            logger.warning("signal analyzer unhealthy")
            trigger_api_restart.send(sender=self.__class__, sigan_healthy=False)
