import logging

from scos_actions.hardware import signa_analyzer_monitor

logger = logging.getLogger(__name__)


def signal_analyzer_registration_handler(sender, **kwargs):
    try:
        logger.debug(f"Registering {sender} as status provider")
        signa_analyzer_monitor.register_signal_analyzer(kwargs["signal_analyzer"])
    except:
        logger.exception("Error registering signal analyzer")
