from .acquire_single_freq_fft import SingleFrequencyFftAcquisition
from .acquire_single_freq_tdomain_iq import SingleFrequencyTimeDomainIqAcquisition
from .acquire_stepped_freq_tdomain_iq import SteppedFrequencyTimeDomainIqAcquisition
from scos_actions.settings import SENSOR_CALIBRATION_FILE
from scos_actions.settings import  SIGAN_CALIBRATION_FILE
from scos_actions.settings import MOCK_SIGAN
from scos_actions.settings import RUNNING_TESTS
from scos_actions.actions import calibration
from scos_actions.actions.tests.resources.utils import create_dummy_calibration
import logging

logger = logging.getLogger(__name__)



def get_calibration(self, sensor_cal_file, sigan_cal_file):
    """Get calibration data from sensor_cal_file and sigan_cal_file."""

    # Try and load sensor/sigan calibration data
    if not RUNNING_TESTS and not MOCK_SIGAN:
        try:
            sensor_calibration = calibration.load_from_json(sensor_cal_file)
        except Exception as err:
            logger.error(
                "Unable to load sensor calibration data, reverting to none"
            )
            logger.exception(err)
            sensor_calibration = None
        try:
            sigan_calibration = calibration.load_from_json(sigan_cal_file)
        except Exception as err:
            logger.error("Unable to load sigan calibration data, reverting to none")
            logger.exception(err)
            sigan_calibration = None
    else:  # If in testing, create our own test files
        dummy_calibration = create_dummy_calibration()
        sensor_calibration = dummy_calibration
        sigan_calibration = dummy_calibration

    return sigan_calibration, sensor_calibration

# Map a class name to an action class
# The YAML loader can key an object with parameters on these class names
action_classes = {
    "single_frequency_fft": SingleFrequencyFftAcquisition,
    "stepped_frequency_time_domain_iq": SteppedFrequencyTimeDomainIqAcquisition,
    "single_frequency_time_domain_iq": SingleFrequencyTimeDomainIqAcquisition,
}

sigan_calibration, sensor_calibration = get_calibration(SENSOR_CALIBRATION_FILE, SIGAN_CALIBRATION_FILE)