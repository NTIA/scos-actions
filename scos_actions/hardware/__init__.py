from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_radio import MockRadio
from scos_actions.capabilities import capabilities
from scos_actions.settings import PRESELECTOR_CONFIG_FILE
from scos_actions import utils
from its_preselector.preselector import Preselector


def load_preselector(preselector_config_file):
    preselector_config = utils.load_from_json(preselector_config_file)
    preselector = Preselector(capabilities, preselector_config)
    return preselector


radio = MockRadio(randomize_values=True)
gps = MockGPS()
preselector = load_preselector(PRESELECTOR_CONFIG_FILE)