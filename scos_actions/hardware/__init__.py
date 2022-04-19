import importlib

from scos_actions import utils
from scos_actions.capabilities import capabilities
from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.settings import PRESELECTOR_CLASS
from scos_actions.settings import PRESELECTOR_CONFIG_FILE
from scos_actions.settings import PRESELECTOR_MODULE


def load_preselector(preselector_config_file):
    if preselector_config_file is None:
        preselector_config = {}
    else:
        preselector_config = utils.load_from_json(preselector_config_file)

    preselector_module = importlib.import_module(PRESELECTOR_MODULE)
    preselector_class = getattr(preselector_module, PRESELECTOR_CLASS)
    preselector = preselector_class(capabilities, preselector_config)

    return preselector


from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
sigan = MockSignalAnalyzer(randomize_values=True)
gps = MockGPS()
preselector = load_preselector(PRESELECTOR_CONFIG_FILE)

