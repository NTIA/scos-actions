from . import status_registrar

import logging

logger = logging.getLogger(__name__)


def status_registration_handler(sender, **kwargs):
    try:
        logger.info("Registering " + str(sender) + ' as status provider')
        status_registrar.add_component(kwargs['component'])
    except:
        logger.exception('Error registering status component')
