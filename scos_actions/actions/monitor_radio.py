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
        detail = ""

        try:
            self.test_required_components()
        except RuntimeError as err:
            healthy = False
            detail = str(err)

        requested_samples = 100000  # Issue #42 hit error at ~70k, so test more

        if healthy:
            try:
                measurement_result = self.radio.acquire_time_domain_samples(requested_samples)
            except Exception:
                detail = "Unable to acquire samples from the signal analyzer"
                healthy = False

        if healthy:
            data = measurement_result["data"]
            if not len(data) == requested_samples:
                detail = "signal analyzer data doesn't match request"
                healthy = False

        if healthy:
            monitor_action_completed.send(sender=self.__class__, radio_healthy=True)
            logger.info("signal analyzer healthy")
        else:
            logger.warning("signal analyzer unhealthy")
            monitor_action_completed.send(sender=self.__class__, radio_healthy=False)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.radio.is_available:
            msg = "acquisition failed: signal analyzer required but not available"
            raise RuntimeError(msg)
