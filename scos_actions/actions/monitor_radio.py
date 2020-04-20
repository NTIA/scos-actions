"""Monitor the on-board USRP and touch or remove an indicator file."""

import logging

from scos_actions.actions.interfaces.action import Action
from scos_actions.actions.interfaces.signals import monitor_action_completed

logger = logging.getLogger(__name__)


class RadioMonitor(Action):
    """Monitor USRP connection and restart container if unreachable."""

    def __init__(self, radio, admin_only=True):
        super(RadioMonitor, self).__init__(admin_only=admin_only)

        self.radio = radio

    def __call__(self, schedule_entry_json, task_id, sensor_definition):
        logger.debug("Performing USRP health check")

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
                data = self.radio.acquire_time_domain_samples(requested_samples)
            except Exception:
                detail = "Unable to acquire USRP"
                healthy = False

        if healthy:
            if not len(data) == requested_samples:
                detail = "USRP data doesn't match request"
                healthy = False

        if healthy:
            try:
                monitor_action_completed.send(sender=self.__class__, radio_healthy=True)
                logger.info("USRP healthy")
            except FileNotFoundError:
                pass
        else:
            logger.warning("USRP unhealthy")
            monitor_action_completed.send(sender=self.__class__, radio_healthy=False)
            raise RuntimeError(detail)

    def test_required_components(self):
        """Fail acquisition if a required component is not available."""
        if not self.radio.is_available:
            msg = "acquisition failed: USRP required but not available"
            raise RuntimeError(msg)
