from typing import Optional

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class Action(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-scos` `Action` objects.

    :param name: Name of the action assigned to the schedule entry. MUST
        be unique on the sensor.
    :param description: A detailed description of what the action does.
    :param summary: A short summary of what the action does.
    """

    name: str
    description: Optional[str] = None
    summary: Optional[str] = None


class ScheduleEntry(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-scos` `ScheduleEntry` objects.

    :param id: Unique identifier for the `ScheduleEntry`.
    :param name: User-specified name of the schedule.
    :param start: Requested time to schedule the first task. Must be
        an ISO 8601 formatted string.
    :param stop: Requested time to end execution of tasks under the schedule.
        Must be an ISO 8601 formatted string.
    :param interval: Seconds between tasks, in seconds.
    :param priority: The priority of the schedule. Lower numbers indicate
        higher priority.
    :param roles: The user roles that are allowed to access acquisitions from
        the schedule.
    """

    id: str
    name: str
    start: Optional[str] = None
    stop: Optional[str] = None
    interval: Optional[int] = None
    priority: Optional[int] = None
    roles: Optional[list[str]] = None
