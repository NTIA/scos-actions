from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

from scos_actions.metadata.interfaces.sigmf_object import SigMFObject
from scos_actions.utils import convert_datetime_to_millisecond_iso_format


@dataclass
class Action(SigMFObject):
    """
    Interface for generating `ntia-scos` `Action` objects.

    The `name` parameter is required.

    :param name: Name of the action assigned to the schedule entry. MUST
        be unique on the sensor.
    :param description: A detailed description of what the action does.
    :param summary: A short summary of what the action does.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.name, "name")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "name": "name",
                "description": "description",
                "summary": "summary",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class ScheduleEntry(SigMFObject):
    """
    Interface for generating `ntia-scos` `ScheduleEntry` objects.

    The `id` and `name` parameters are required.

    :param id: Unique identifier for the `ScheduleEntry`.
    :param name: User-specified name of the schedule.
    :param start: Requested time to schedule the first task.
    :param stop: Requested time to end execution of tasks under the schedule.
    :param interval: Seconds between tasks, in seconds.
    :param priority: The priority of the schedule. Lower numbers indicate
        higher priority.
    :param roles: The user roles that are allowed to access acquisitions from
        the schedule.
    """

    id: Optional[str] = None
    name: Optional[str] = None
    start: Optional[Union[datetime, str]] = None
    stop: Optional[Union[datetime, str]] = None
    interval: Optional[int] = None
    priority: Optional[int] = None
    roles: Optional[List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.id, "id")
        self.check_required(self.name, "name")
        # Convert datetime to string if needed
        if isinstance(self.start, datetime):
            self.start = convert_datetime_to_millisecond_iso_format(self.start)
        if isinstance(self.stop, datetime):
            self.stop = convert_datetime_to_millisecond_iso_format(self.stop)
        # Define SigMF key names
        self.obj_keys.update(
            {
                "id": "id",
                "name": "name",
                "start": "start",
                "stop": "stop",
                "interval": "interval",
                "priority": "priority",
                "roles": "roles",
            }
        )
        # Create metadata object
        super().create_json_object()
