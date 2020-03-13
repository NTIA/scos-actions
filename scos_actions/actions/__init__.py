import logging

from . import logger as logger_action
from .acquire_single_freq_fft import SingleFrequencyFftAcquisition
from .acquire_stepped_freq_tdomain_iq import SteppedFrequencyTimeDomainIqAcquisition

logger = logging.getLogger(__name__)


# Actions initialized here are made available through the API
registered_actions = {
    "logger": logger_action.Logger(),
    "admin_logger": logger_action.Logger(
        loglvl=logger_action.LOGLVL_ERROR, admin_only=True
    ),
}

by_name = registered_actions


# Map a class name to an action class
# The YAML loader can key an object with parameters on these class names
action_classes = {
    "logger": logger_action.Logger,
    "single_frequency_fft": SingleFrequencyFftAcquisition,
    "stepped_frequency_time_domain_iq": SteppedFrequencyTimeDomainIqAcquisition,
}


def get_action_with_summary(action):
    """Given an action, return the string 'action_name - summary'."""
    action_fn = registered_actions[action]
    summary = get_summary(action_fn)
    action_with_summary = action
    if summary:
        action_with_summary += " - {}".format(summary)

    return action_with_summary


def get_summary(action_fn):
    """Extract the first line of the action's description as a summary."""
    description = action_fn.description
    summary = None
    if description:
        summary = description.splitlines()[0]

    return summary





MAX_LENGTH = 50
VALID_ACTIONS = []
CHOICES = []
ADMIN_CHOICES = []


# def init():
#     """Allows re-initing VALID_ACTIONS if `registered_actions` is modified."""
#     global VALID_ACTIONS
#     global CHOICES
#
#     parsed_actions = load_from_yaml()
#     registered_actions.update(parsed_actions)
#
#     VALID_ACTIONS = sorted(registered_actions.keys())
#     for action in VALID_ACTIONS:
#         if registered_actions[action].admin_only:
#             ADMIN_CHOICES.append((action, get_action_with_summary(action)))
#         else:
#             CHOICES.append((action, get_action_with_summary(action)))
#
#
# init()
