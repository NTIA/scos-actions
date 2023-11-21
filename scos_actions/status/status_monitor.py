import logging

logger = logging.getLogger(__name__)


class StatusMonitor:
    def __init__(self):
        logger.debug("Initializing StatusMonitor")
        self.status_components = []

    def add_component(self, component):
        self.status_components.append(component)
