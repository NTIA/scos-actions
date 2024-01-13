import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from scos_actions.hardware.gps_iface import GPSInterface
from scos_actions.hardware.sensor import Sensor
from scos_actions.hardware.sigan_iface import SIGAN_SETTINGS_KEYS
from scos_actions.metadata.sigmf_builder import SigMFBuilder
from scos_actions.metadata.structs import ntia_scos, ntia_sensor
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

    def __init__(self, parameters):
        self._sensor = None
        self.parameters = deepcopy(parameters)
        self.sigmf_builder = None

    def configure(self, params: dict):
        self.configure_sigan(params)
        self.configure_preselector(self.sensor, params)

    @property
    def sensor(self):
        return self._sensor

    @sensor.setter
    def sensor(self, value):
        self._sensor = value

    def configure_sigan(self, params: dict):
        sigan_params = {k: v for k, v in params.items() if k in SIGAN_SETTINGS_KEYS}
        for key, value in sigan_params.items():
            if hasattr(self.sensor.signal_analyzer, key):
                logger.debug(f"Applying setting to sigan: {key}: {value}")
                setattr(self.sensor.signal_analyzer, key, value)
            else:
                logger.warning(f"Sigan does not have attribute {key}")

    def configure_preselector(self, sensor: Sensor, params: dict):
        preselector = sensor.preselector
        if self.PRESELECTOR_PATH_KEY in params:
            path = params[self.PRESELECTOR_PATH_KEY]
            logger.debug(f"Setting preselector RF path: {path}")
            preselector.set_state(path)
        elif sensor.has_configurable_preselector:
            # Require the RF path to be specified if the sensor has a preselector.
            raise ParameterException(
                f"No {self.PRESELECTOR_PATH_KEY} value specified in the YAML config."
            )
        else:
            # No preselector in use, so do not require an RF path
            pass

    def get_sigmf_builder(self, sensor: Sensor, schedule_entry: dict) -> None:
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

        if sensor.location is not None:
            sigmf_builder.set_geolocation(sensor.location)
        if self.sensor.capabilities is not None and hasattr(
            self.sensor.capabilities, "sensor"
        ):
            sigmf_builder.set_sensor(
                ntia_sensor.Sensor(**self.sensor.capabilities["sensor"])
            )

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
    def __call__(self, sensor=None, schedule_entry=None, task_id=None):
        pass
