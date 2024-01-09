import logging

logger = logging.getLogger(__name__)


class SignalAnalyzerMonitor:
    def __init__(self):
        logger.debug("Initializing Signal Analyzer Monitor")
        self._signal_analyzer = None

    def register_signal_analyzer(self, sigan):
        logger.debug(f"Setting Signal Analyzer to {sigan}")
        self._signal_analyzer = sigan

    @property
    def signal_analyzer(self):
        return self._signal_analyzer
