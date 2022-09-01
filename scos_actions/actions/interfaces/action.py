import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from scos_actions.capabilities import capabilities
from scos_actions.hardware import gps as mock_gps
from scos_actions.hardware import preselector
from scos_actions.hardware import sigan as mock_sigan
from scos_actions.utils import ParameterException, get_parameter

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

    PRESELECTOR_PATH_KEY = "rf_path"

    def __init__(self, parameters, sigan=mock_sigan, gps=mock_gps):
        self.parameters = deepcopy(parameters)
        self.sigan = sigan
        self.gps = gps
        self.sensor_definition = capabilities["sensor"]
        self.has_preselector = (
            True if "preselector" in self.sensor_definition else False
        )

    def configure(self, measurement_params: dict):
        self.configure_sigan(measurement_params)
        self.configure_preselector(measurement_params)

    def configure_sigan(self, measurement_params: dict):
        for key, value in measurement_params.items():
            if hasattr(self.sigan, key):
                logger.debug(f"Applying setting to sigan: {key}: {value}")
                setattr(self.sigan, key, value)
            else:
                logger.warning(f"Sigan does not have attribute {key}")

    def configure_preselector(self, measurement_params: dict):
        if self.PRESELECTOR_PATH_KEY in measurement_params:
            path = measurement_params[self.PRESELECTOR_PATH_KEY]
            logger.debug(f"Setting preselector RF path: {path}")
            preselector.set_state(path)
        elif self.has_preselector:
            # Set RF path automatically if only one exists.
            if len(preselector.rf_paths) != 1:
                logger.debug(
                    f"No {self.PRESELECTOR_PATH_KEY} specified, but only one is "
                    + f" available. Setting the {self.PRESELECTOR_PATH_KEY} to "
                    + f"{preselector.__get_rf_paths()[0]}."
                )
            else:
                # Otherwise, require the RF path to be specified.
                raise ParameterException(
                    f"No {self.PRESELECTOR_PATH_KEY} value specified in the YAML config."
                )
        else:
            # No preselector in use, so do not require an RF path
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

    @property
    def name(self):
        return get_parameter("name", self.parameters)

    @abstractmethod
    def __call__(self, schedule_entry, task_id):
        pass
