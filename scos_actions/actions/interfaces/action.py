import copy
import logging
from abc import ABC, abstractmethod

from scos_actions.hardware import gps as mock_gps
from scos_actions.hardware import preselector
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

    def __init__(self, parameters={}, sigan=mock_sigan, gps=mock_gps):
        self.parameters = parameters
        self.sigan = sigan
        self.gps = gps
        self.sensor_definition = capabilities['sensor']
        self.parameter_map = self.get_parameter_map(self.parameters)
        self.metadata_generators = {}
        self.sigmf_builder = SigMFBuilder()

    @abstractmethod
    def __call__(self, schedule_entry_json, task_id):
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
        if isinstance(measurement_params, list):
            for item in measurement_params:
                self.configure_sigan_with_dictionary(item)

        elif isinstance(measurement_params, dict):
            self.configure_sigan_with_dictionary(measurement_params)

    def configure_sigan_with_dictionary(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self.sigan, key):
                setattr(self.sigan, key, value)
            else:
                logger.warning(f"radio does not have attribute {key}")

    def configure_preselector(self, measurement_params):
        if self.PRESELECTOR_PATH_KEY in measurement_params:
            path = measurement_params[self.PRESELECTOR_PATH_KEY]
            preselector.set_state(path)

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
    def add_metadata_generators(self, measurement_result):
        pass

    @abstractmethod
    def create_metadata(self, schedule_entry, measurement_result):
        pass

    @abstractmethod
    def execute(self, schedule_entry, task_id):
        pass

    def __call__(self, schedule_entry, task_id):
        self.test_required_components()
        self.configure(self.parameters)
        measurement_result = self.execute(schedule_entry, task_id)
        self.add_metadata_generators(measurement_result)
        self.create_metadata(schedule_entry, measurement_result)
        self.send_signals()

    @abstractmethod
    def test_required_components(self):
        pass

    @abstractmethod
    def send_signals(self, measurement_result):
        pass

