from dataclasses import dataclass
from typing import Optional

from scos_actions.metadata.interfaces.sigmf_object import SigMFObject


@dataclass
class Environment(SigMFObject):
    """
    Interface for generating a `ntia-environment` `Environment` object.

    :param category: Categorical description of the environment where
        the sensor is mounted, e.g., `"indoor"`, `"outdoor urban"`,
        `"outdoor rural"`.
    :param temperature: Environmental temperature, in degrees Celsius.
    :param humidity: Relative humidity, as a percentage.
    :param weather: Weather around the sensor, e.g., `"rain"`, `"snow"`.
    :param description: A description of the environment.
    """

    category: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    weather: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Define SigMF key names
        self.obj_keys.update(
            {
                "category": "category",
                "temperature": "temperature",
                "humidity": "humidity",
                "weather": "weather",
                "description": "description",
            }
        )
        # Create metadata object
        super().create_json_object()
