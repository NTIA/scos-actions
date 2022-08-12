from scos_actions.status.status_registration_handler import status_registration_handler
from scos_actions.actions.interfaces.signals import register_component_with_status

import logging

logger = logging.getLogger(__name__)

register_component_with_status.connect(status_registration_handler)
logger.info('Connected status registration handler')