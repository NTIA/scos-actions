import logging
import sys
from os import path
from pathlib import Path

from environs import Env

logger = logging.getLogger(__name__)
env = Env()

logger.debug("Initializing scos-actions settings")
CONFIG_DIR = Path(__file__).parent.resolve() / "configs"
logger.debug(f"scos-actions: CONFIG_DIR:{CONFIG_DIR}")
ACTION_DEFINITIONS_DIR = CONFIG_DIR / "actions"
logger.debug(f"scos-actions: ACTION_DEFINITIONS_DIR:{ACTION_DEFINITIONS_DIR}")
SWITCH_CONFIGS_DIR = env("SWITCH_CONFIGS_DIR", default=None)
if SWITCH_CONFIGS_DIR:
    SWITCH_CONFIGS_DIR = Path(SWITCH_CONFIGS_DIR)
logger.debug(f"scos-actions: SWITCH_CONFIGS_DIR:{SWITCH_CONFIGS_DIR}")
SCOS_SENSOR_GIT_TAG = env("SCOS_SENSOR_GIT_TAG", default="unknown")
logger.debug(f"scos-actions: SCOS_SENSOR_GIT_TAG:{SCOS_SENSOR_GIT_TAG}")
MOCK_SIGAN = env.bool("MOCK_SIGAN", True)
logger.debug(f"scos-actions: MOCK_SIGAN:{MOCK_SIGAN}")
MOCK_SIGAN_RANDOM = env.bool("MOCK_SIGAN_RANDOM", default=False)
logger.debug(f"scos-actions: MOCK_SIGAN_RANDOM:{MOCK_SIGAN_RANDOM}")
__cmd = path.split(sys.argv[0])[-1]
RUNNING_TESTS = env.bool("RUNNING_TESTS", "test" in __cmd)
logger.debug(f"scos-actions: RUNNING_TESTS:{RUNNING_TESTS}")
FQDN = env("FQDN", None)
logger.debug(f"scos-actions: FQDN:{FQDN}")

SIGAN_MODULE = env.str("SIGAN_MODULE", default=None)
if RUNNING_TESTS:
    SIGAN_MODULE = "scos_actions.hardware.mocks.mock_sigan"
logger.debug(f"scos-actions: SIGAN_MODULE:{SIGAN_MODULE}")
SIGAN_CLASS = env.str("SIGAN_CLASS", default=None)
if RUNNING_TESTS:
    SIGAN_CLASS = "MockSignalAnalyzer"
logger.debug(f"scos-actions: SIGAN_CLASS:{SIGAN_CLASS}")
SIGAN_POWER_SWITCH = env("SIGAN_POWER_SWITCH", default=None)
logger.debug(f"scos-actions: SIGAN_POWER_SWITCH:{SIGAN_POWER_SWITCH}")
SIGAN_POWER_CYCLE_STATES = env("SIGAN_POWER_CYCLE_STATES", default=None)
logger.debug(f"scos-actions: SIGAN_POWER_CYCLE_STATES:{SIGAN_POWER_CYCLE_STATES}")
PRESELECTOR_MODULE = env("PRESELECTOR_MODULE", default=None)
logger.debug(f"scos-actions: PRESELECTOR_MODULE:{PRESELECTOR_MODULE}")
PRESELECTOR_CLASS = env("PRESELECTOR_CLASS", default=None)
logger.debug(f"scos-actions: PRESELECTOR_CLASS:{PRESELECTOR_CLASS}")

SSD_DEVICE = env("SSD_DEVICE", default="/dev/nvme0n1")
logger.debug(f"scos-actions: SSD-DEVICE:{SSD_DEVICE}")
