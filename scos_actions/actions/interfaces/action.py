import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from scos_actions.capabilities import SENSOR_LOCATION, capabilities
from scos_actions.hardware import preselector
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.metadata.interfaces import ntia_scos, ntia_sensor
from scos_actions.metadata.sigmf_builder import SigMFBuilder
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

    def __init__(self, parameters, sigan, gps=None):
        if gps is None:
            gps = MockGPS()
        self.parameters = deepcopy(parameters)
        self.sigan = sigan
        self.gps = gps
        self.sensor_definition = capabilities["sensor"]
        self.sigmf_builder = None
        if (
            "preselector" in self.sensor_definition
            and "rf_paths" in self.sensor_definition["preselector"]
        ):
            self.has_configurable_preselector = True
        else:
            self.has_configurable_preselector = False

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
        elif self.has_configurable_preselector:
            # Require the RF path to be specified if the sensor has a preselector.
            raise ParameterException(
                f"No {self.PRESELECTOR_PATH_KEY} value specified in the YAML config."
            )
        else:
            # No preselector in use, so do not require an RF path
            pass

    def get_sigmf_builder(self, schedule_entry: dict) -> None:
        """
        Set the `sigmf_builder` instance variable to an initialized SigMFBuilder.

        Schedule entry and action information will be populated using `ntia-scos`
        fields, and sensor metadata will be filled using `ntia-sensor` fields.
        """
        sigmf_builder = SigMFBuilder()

        schedule_entry_cleaned = {
            k: v
            for k, v in schedule_entry.items()
            if k in ["id", "name", "start", "stop", "interval", "priority", "roles"]
        }
        if "id" not in schedule_entry_cleaned:
            # If there is no ID, reuse the "name" as the ID as well
            schedule_entry_cleaned["id"] = schedule_entry_cleaned["name"]
        schedule_entry_obj = ntia_scos.ScheduleEntry(**schedule_entry_cleaned)
        sigmf_builder.set_schedule(schedule_entry_obj)

        action_obj = ntia_scos.Action(
            name=self.name,
            description=self.description,
            summary=self.summary,
        )
        sigmf_builder.set_action(action_obj)

        if SENSOR_LOCATION is not None:
            sigmf_builder.set_geolocation(SENSOR_LOCATION)
        sigmf_builder.set_sensor(ntia_sensor.Sensor(**self.sensor_definition))

        self.sigmf_builder = sigmf_builder

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
