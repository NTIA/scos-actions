"""Monitor the signal analyzer."""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import monitor_action_completed
from scos_actions.hardware import gps as mock_gps

logger = logging.getLogger(__name__)


class MonitorSignalAnalyzer(Action):
    """Monitor signal analyzer connection and restart container if unreachable."""

    def __init__(self, sigan, parameters={}, gps=mock_gps):
        super().__init__(parameters=parameters,sigan=sigan, gps=gps)

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        logger.debug("Performing signal analyzer health check")

        healthy = True

        if not self.sigan.is_available:
            healthy = False
        else:
            healthy = self.sigan.healthy

        if healthy:
            monitor_action_completed.send(sender=self.__class__, sigan_healthy=True)
            logger.info("signal analyzer healthy")
        else:
            logger.warning("signal analyzer unhealthy")
            monitor_action_completed.send(sender=self.__class__, sigan_healthy=False)
