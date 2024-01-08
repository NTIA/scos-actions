import importlib
import logging
from pathlib import Path

from its_preselector.configuration_exception import ConfigurationException
from its_preselector.controlbyweb_web_relay import ControlByWebWebRelay

from scos_actions import utils
from scos_actions.capabilities import capabilities
from scos_actions.settings import (
    PRESELECTOR_CLASS,
    PRESELECTOR_CONFIG_FILE,
    PRESELECTOR_MODULE,
    SWITCH_CONFIGS_DIR,
)
from scos_actions.signals import (
    register_component_with_status,
    register_signal_analyzer,
)
from scos_actions.status.status_registration_handler import status_registration_handler
from scos_actions.status.signal_analyzer_registration_handler import (
    signal_analyzer_registration_handler,
)

logger = logging.getLogger(__name__)


def load_switches(switch_dir: Path) -> dict:
    switch_dict = {}
    if switch_dir is not None and switch_dir.is_dir():
        for f in switch_dir.iterdir():
            file_path = f.resolve()
            logger.debug(f"loading switch config {file_path}")
            conf = utils.load_from_json(file_path)
            try:
                switch = ControlByWebWebRelay(conf)
                logger.debug(f"Adding {switch.id}")

                switch_dict[switch.id] = switch
                logger.debug(f"Registering switch status for {switch.name}")
                register_component_with_status.send(__name__, component=switch)
            except ConfigurationException:
                logger.error(f"Unable to configure switch defined in: {file_path}")

    return switch_dict


def load_preselector_from_file(preselector_config_file: Path):
    if preselector_config_file is None:
        return None
    else:
        try:
            preselector_config = utils.load_from_json(preselector_config_file)
            return load_preselector(
                preselector_config, PRESELECTOR_MODULE, PRESELECTOR_CLASS
            )
        except ConfigurationException:
            logger.exception(
                f"Unable to create preselector defined in: {preselector_config_file}"
            )
    return None


def load_preselector(preselector_config, module, preselector_class_name):
    if module is not None and preselector_class_name is not None:
        preselector_module = importlib.import_module(module)
        preselector_constructor = getattr(preselector_module, preselector_class_name)
        ps = preselector_constructor(capabilities["sensor"], preselector_config)
        if ps and ps.name:
            logger.debug(f"Registering {ps.name} as status provider")
            register_component_with_status.send(__name__, component=ps)
    else:
        ps = None
    return ps


register_signal_analyzer.connect(signal_analyzer_registration_handler)
register_component_with_status.connect(status_registration_handler)
logger.debug("Connected status registration handler")
preselector = load_preselector_from_file(PRESELECTOR_CONFIG_FILE)
switches = load_switches(SWITCH_CONFIGS_DIR)
logger.debug(f"Loaded {(len(switches))} switches.")
