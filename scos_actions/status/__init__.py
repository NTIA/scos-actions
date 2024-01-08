import datetime

from .signal_analyzer_monitor import SignalAnalyzerMonitor
from .status_monitor import StatusMonitor

status_registrar = StatusMonitor()
signal_analyzer_monitor = SignalAnalyzerMonitor()
start_time = datetime.datetime.utcnow()
