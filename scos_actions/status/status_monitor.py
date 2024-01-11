import logging

logger = logging.getLogger(__name__)


class StatusMonitor:
    def __init__(self):
        logger.debug("Initializing StatusMonitor")
        self.status_components = []

    def add_component(self, component):
        """
        Allows objects to be registered to provide status. Any object registered will
        be included in scos-sensors status endpoint. All objects registered must
        implement a get_status() method that returns a dictionary.

        :param component: the object to add to the list of status providing objects.
        """
        if hasattr(component, "get_status"):
            self.status_components.append(component)
