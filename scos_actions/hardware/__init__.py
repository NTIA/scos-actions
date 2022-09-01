import importlib
import logging
import os

from its_preselector.configuration_exception import ConfigurationException
from its_preselector.controlbyweb_web_relay import ControlByWebWebRelay

from scos_actions import utils
from scos_actions.actions.interfaces.signals import register_component_with_status
from scos_actions.capabilities import capabilities
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.settings import (
    PRESELECTOR_CLASS,
    PRESELECTOR_CONFIG_FILE,
    PRESELECTOR_MODULE,
    SWITCH_CONFIGS_DIR,
)
from scos_actions.status.status_registration_handler import status_registration_handler

logger = logging.getLogger(__name__)


def load_switches(switch_dir):
    switch_dict = {}
    if os.path.isdir(switch_dir):
        files = os.listdir(switch_dir)
        for f in files:
            file_path = os.path.join(switch_dir, f)
            logger.info("loading switch config " + file_path)
            conf = utils.load_from_json(file_path)
            try:
                switch = ControlByWebWebRelay(conf)
                switch_dict[switch.id] = switch
                logger.info("Registering switch status for " + switch.name)
                register_component_with_status.send(__name__, component=switch)
            except (ConfigurationException):
                logger.error("Unable to configure switch defined in: " + file_path)

    return switch_dict


def load_preslector_from_file(preselector_config_file):
    if preselector_config_file is None:
        return None
    else:
        try:
            preselector_config = utils.load_from_json(preselector_config_file)
            return load_preselector(preselector_config)
        except ConfigurationException:
            logger.error(
                "Unable to create preselector defined in: " + preselector_config_file
            )
    return None


def load_preselector(preselector_config, module, preselector_class_name):
    if module is not None and preselector_class_name is not None:
        preselector_module = importlib.import_module(module)
        preselector_constructor = getattr(preselector_module, preselector_class_name)
        ps = preselector_constructor(capabilities["sensor"], preselector_config)
        if ps and ps.name:
            logger.info("Registering " + ps.name + " as status provider")
            register_component_with_status.send(__name__, component=ps)
    else:
        ps = None
    return ps


register_component_with_status.connect(status_registration_handler)
logger.info("Connected status registration handler")
sigan = MockSignalAnalyzer(randomize_values=True)
gps = MockGPS()
preselector = load_preslector_from_file(
    PRESELECTOR_CONFIG_FILE, PRESELECTOR_MODULE, PRESELECTOR_CLASS
)
switches = load_switches(SWITCH_CONFIGS_DIR)
