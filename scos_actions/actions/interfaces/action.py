import copy
import logging
from abc import ABC, abstractmethod

from scos_actions.hardware import gps as mock_gps
from scos_actions.hardware import sigan as mock_sigan
from scos_actions.capabilities import capabilities
from scos_actions.actions.sigmf_builder import SigMFBuilder

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

    def __init__(self, parameters, sigan=mock_sigan, gps=mock_gps):
        self.parameters = parameters
        self.sigan = sigan
        self.gps = gps
        self.sensor_definition = capabilities['sensor']
        self.parameter_map = self.get_parameter_map(self.parameters)
        self.metadata_generators = {}
        self.sigmf_builder = SigMFBuilder()



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
        return self.parameter_map['name']

    def get_parameter_map(self, params):
        if isinstance(params, list):
            key_map = {}
            for param in params:
                for key, value in param.items():
                    key_map[key] = value
            return key_map
        elif isinstance(params, dict):
           return copy.deepcopy(params)


    @abstractmethod
    def __call__(self, schedule_entry, task_id):
        pass



