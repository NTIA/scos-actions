import logging

from scos_actions.core import gps_monitor

logger = logging.getLogger(__name__)


def gps_registration_handler(sender, **kwargs):
    try:
        gps = kwargs["gps"]
        logger.debug(f"Registering GPS: {gps}")
        gps_monitor.register_gps(gps)
    except:
        logger.exception("Error registering gps")
