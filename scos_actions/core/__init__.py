import logging

from .gps_monitor import GpsMonitor
from .signal_analyzer_monitor import SignalAnalyzerMonitor
from .status_monitor import StatusMonitor

logger = logging.getLogger(__name__)
logger.debug("********** Initializing scos-actions.core **********")
signal_analyzer_monitor = SignalAnalyzerMonitor()
status_registrar = StatusMonitor()
gps_monitor = GpsMonitor()
