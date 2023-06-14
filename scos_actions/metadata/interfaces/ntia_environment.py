from typing import Optional

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class Environment(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
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
