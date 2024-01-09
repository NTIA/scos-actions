import logging

from scos_actions.core import status_registrar

logger = logging.getLogger(__name__)


def status_registration_handler(sender, **kwargs):
    try:
        logger.debug(f"Registering {sender} as status provider")
        status_registrar.add_component(kwargs["component"])
    except:
        logger.exception("Error registering status component")
