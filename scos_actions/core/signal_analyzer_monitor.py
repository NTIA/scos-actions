import logging

logger = logging.getLogger(__name__)


class SignalAnalyzerMonitor:
    def __init__(self):
        logger.debug("Initializing Signal Analyzer Monitor")
        self.signal_analyzer = None

    def register_signal_analyzer(self, sigan):
        logger.debug(f"Setting Signal Analyzer to {sigan}")
        self.signal_analyzer = sigan

    @property
    def signal_analyzer(self):
        return self.signal_analyzer
