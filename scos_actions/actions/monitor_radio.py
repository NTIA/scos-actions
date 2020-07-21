"""Monitor the signal analyzer."""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import monitor_action_completed

logger = logging.getLogger(__name__)


class RadioMonitor(Action):
    """Monitor signal analyzer connection and restart container if unreachable."""

    def __init__(self, radio, admin_only=True):
        super(RadioMonitor, self).__init__(admin_only=admin_only)

        self.radio = radio

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        logger.debug("Performing signal analyzer health check")

        healthy = True

        if not self.radio.is_available:
            healthy = False
        else:
            healthy = self.radio.healthy

        if healthy:
            monitor_action_completed.send(sender=self.__class__, radio_healthy=True)
            logger.info("signal analyzer healthy")
        else:
            logger.warning("signal analyzer unhealthy")
            monitor_action_completed.send(sender=self.__class__, radio_healthy=False)
