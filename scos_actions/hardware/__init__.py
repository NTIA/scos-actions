import importlib
import os

from scos_actions import utils
from scos_actions.capabilities import capabilities
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from its_preselector.controlbyweb_web_relay import ControlByWebWebRelay
from scos_actions.status.status_registration_handler import status_registration_handler
from scos_actions.actions.interfaces.signals import register_component_with_status
from scos_actions.settings import (
    PRESELECTOR_CLASS,
    PRESELECTOR_CONFIG_FILE,
    PRESELECTOR_MODULE,
    SWITCH_CONFIGS_DIR
)

import logging

logger = logging.getLogger(__name__)


def load_switches(switch_dir):
    switch_dict = {}
    if os.path.isdir(switch_dir):
        files = os.listdir(switch_dir)
        for f in files:
            file_path = os.path.join(switch_dir, f)
            logger.info('loading switch config ' + file_path)
            conf = utils.load_from_json(file_path)
            switch = ControlByWebWebRelay(conf)
            switch_dict[switch.id] = switch
            logger.info('Registering switch status for ' + switch.name)
            register_component_with_status.send(__name__, component=switch)
    return switch_dict


def load_preselector(preselector_config_file):
    if preselector_config_file is None:
        preselector_config = {}
    else:
        preselector_config = utils.load_from_json(preselector_config_file)

    preselector_module = importlib.import_module(PRESELECTOR_MODULE)
    preselector_class = getattr(preselector_module, PRESELECTOR_CLASS)
    ps = preselector_class(capabilities['sensor'], preselector_config)
    logger.info('Registering ' + preselector.name + ' as status provider')
    register_component_with_status.send(__name__, component=preselector)
    return ps


register_component_with_status.connect(status_registration_handler)
logger.info('Connected status registration handler')
sigan = MockSignalAnalyzer(randomize_values=True)
gps = MockGPS()
preselector = load_preselector(PRESELECTOR_CONFIG_FILE)
switches = load_switches(SWITCH_CONFIGS_DIR)
