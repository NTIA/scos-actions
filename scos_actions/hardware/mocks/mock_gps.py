import logging
from datetime import datetime

from scos_actions.hardware.gps_iface import GPSInterface

logger = logging.getLogger(__name__)


class MockGPS(GPSInterface):

    def get_location(self, sigan, timeout_s=1):
        logger.warning("Using mock GPS!")
        return 39.995118, -105.261572, 1651.0

    def get_gps_time(self, sigan):
        logger.warning("Using mock GPS!")
        return datetime.now()
