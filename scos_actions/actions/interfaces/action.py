import logging
from abc import ABC, abstractmethod

from scos_actions.hardware import gps as mock_gps
from scos_actions.hardware import preselector
from scos_actions.hardware import sigan as mock_sigan

logger = logging.getLogger(__name__)


class Action(ABC):
    """The action base class.

    To create an action, create a subclass of `Action` with a descriptive
    docstring and override the `__call__` method.

    The scheduler reports the 'success' or 'failure' of an action by the
    following convention:

      * If at any point or for any reason that `__call__` function raises an
        exception, the task is marked a 'failure' and `str(err)` is provided
        as a detail to the user, where `err` is the raised Exception object.

      * If the `__call__` function returns normally, the task was a 'success',
        and if the value returned to the scheduler is a string, it will be
        added to the task result's detail field.

    """
    PRESELECTOR_PATH_KEY='rf_path'

    def __init__(self, parameters={}, sigan=mock_sigan, gps=mock_gps):
        self.parameters = parameters
        self.sigan = sigan
        self.gps = gps

    @abstractmethod
    def __call__(self, schedule_entry_json, sensor_definition):
        pass

    @property
    def summary(self):
        try:
            return self.description.splitlines()[0]
        except IndexError:
            return "Summary not provided."

    @property
    def description(self):
        return self.__doc__

    def configure(self, measurement_params):
        self.configure_sigan(measurement_params)
        self.configure_preselector(measurement_params)

    def configure_sigan(self, measurement_params):
        for key, value in measurement_params.items():
            if hasattr(self.sigan, key):
                setattr(self.sigan, key, value)
            else:
                logger.warning(f"radio does not have attribute {key}")

    def configure_preselector(self, measurement_params):
        if self.PRESELECTOR_PATH_KEY in measurement_params:
            path = measurement_params[self.PRESELECTOR_PATH_KEY]
            preselector.set_rf_path(path)
