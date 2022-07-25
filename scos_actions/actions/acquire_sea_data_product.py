# What follows is a parameterizable description of the algorithm used by this
# action. The first line is the summary and should be written in plain text.
# Everything following that is the extended description, which can be written
# in Markdown and MathJax. Each name in curly brackets '{}' will be replaced
# with the value specified in the `description` method which can be found at
# the very bottom of this file. Since this parameterization step affects
# everything in curly brackets, math notation such as {m \over n} must be
# escaped to {{m \over n}}.
#
# To print out this docstring after parameterization, see
# scos-sensor/scripts/print_action_docstring.py. You can then paste that into the
# SCOS Markdown Editor (link below) to see the final rendering.
#
# Resources:
# - MathJax reference: https://math.meta.stackexchange.com/q/5020
# - Markdown reference: https://commonmark.org/help/
# - SCOS Markdown Editor: https://ntia.github.io/scos-md-editor/
#
r"""Acquire a NASCTN SEA data product.

Currently in development.
"""
import logging
import numpy as np
import numexpr as ne
from typing import Tuple
from scipy.signal import sosfilt

from scos_actions import utils
from scos_actions.actions.metadata.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.actions.interfaces.measurement_action import MeasurementAction
from scos_actions.hardware import gps as mock_gps
from scos_actions.signal_processing.filtering import generate_elliptic_iir_low_pass_filter
from scos_actions.signal_processing.fft import get_fft, get_fft_window, get_fft_window_correction
from scos_actions.signal_processing.power_analysis import apply_power_detector, create_power_detector, calculate_pseudo_power
from scos_actions.signal_processing.unit_conversion import convert_watts_to_dBm

logger = logging.getLogger(__name__)

# Define parameter keys
IIR_APPLY = 'iir_apply'
RP_DB = 'iir_rp_dB'
RS_DB = 'iir_rs_dB'
IIR_CUTOFF_HZ = 'iir_cutoff_Hz'
IIR_WIDTH_HZ = 'iir_width_Hz'
QFILT_APPLY = 'qfilt_apply'
Q_LO = 'qfilt_qlo'
Q_HI = 'qfilt_qhi'
FFT_SIZE = "fft_size"
NUM_FFTS = 'nffts'
FFT_WINDOW_TYPE = "fft_window_type"
APD_BIN_SIZE_DB = 'apd_bin_size_dB'
TD_BIN_SIZE_MS = 'td_bin_size_ms'
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"


class NasctnSeaDataProduct(MeasurementAction):
    """Acquire a stepped-frequency NASCTN SEA data product.

    :param parameters: The dictionary of parameters needed for
        the action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """
    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)

        self.sigan = sigan  # make instance variable to allow mocking

        # Setup/pull config parameters
        # TODO: All parameters in this section should end up hard-coded
        self.iir_rp_dB = utils.get_parameter(RP_DB, self.parameters)
        self.iir_rs_dB = utils.get_parameter(RS_DB, self.parameters)
        self.iir_cutoff_Hz = utils.get_parameter(IIR_CUTOFF_HZ, self.parameters)
        self.iir_width_Hz = utils.get_parameter(IIR_WIDTH_HZ, self.parameters)
        self.qfilt_qlo = utils.get_parameter(Q_LO, self.parameters)
        self.qfilt_qhi = utils.get_parameter(Q_HI, self.parameters)
        self.fft_window_type = utils.get_parameter(FFT_WINDOW_TYPE, self.parameters)

        # TODO: These parameters should not be hard-coded
        # None of these should be lists - all single values
        self.iir_apply = utils.get_parameter(IIR_APPLY, self.parameters)
        self.qfilt_apply = utils.get_parameter(QFILT_APPLY, self.parameters)
        self.fft_size = utils.get_parameter(FFT_SIZE, self.parameters)
        self.nffts = utils.get_parameter(NUM_FFTS, self.parameters)
        self.apd_bin_size_dB = utils.get_parameter(APD_BIN_SIZE_DB, self.parameters)
        self.td_bin_size_ms = utils.get_parameter(TD_BIN_SIZE_MS, self.parameters)
        self.sample_rate_Hz = utils.get_parameter(SAMPLE_RATE, self.parameters)

        # Construct IIR filter
        self.iir_sos = generate_elliptic_iir_low_pass_filter(self.iir_rp_dB, self.iir_rs_dB, self.iir_cutoff_Hz, self.iir_width_Hz, self.sample_rate_Hz)

        # Generate FFT window and get its energy correction factor
        self.fft_window = get_fft_window(self.fft_window_type, self.fft_size)
        self.fft_window_ecf = get_fft_window_correction(self.fft_window, 'energy')

        # Create power detectors
        self.fft_detector = create_power_detector("FftMeanMaxDetector", ["mean", "max"])
        self.td_detector = create_power_detector("TdMeanMaxDetector", ["mean", "max"])


    def __call__(self, schedule_entry, task_id):
        """This is the entrypoint function called by the scheduler."""
        # Temporary: remove config parameters which will be hard-coded eventually
        for key in [RP_DB, RS_DB, IIR_CUTOFF_HZ, IIR_WIDTH_HZ, Q_LO, Q_HI, FFT_WINDOW_TYPE]:
            self.parameters.pop(key)
        self.test_required_components()

        iteration_params = utils.get_iterable_parameters(self.parameters)

        # Handle single-channel case
        if len(iteration_params) == 1:
            # Capture IQ and generate data product
            pass
        else:
            for i, p in enumerate(iteration_params):
                # Capture and save IQ data
                self.capture_iq(schedule_entry, task_id, i, p)
                pass
            for i, p in enumerate(iteration_params):
                # Load IQ data, generate data product, save
                pass

    def capture_iq(self, schedule_entry, task_id, recording_id, params):
        start_time = utils.get_datetime_str_now()
        # Configure signal analyzer + preselector
        self.configure(params)
        # Get IQ capture parameters
        sample_rate = self.sigan.sample_rate
        duration_ms = utils.get_parameter(DURATION_MS, params)
        nskip = utils.get_parameter(NUM_SKIP, params)
        num_samples = int(sample_rate * duration_ms * 1e-3)
        # Collect IQ data
        measurement_result = super().acquire_data(num_samples, nskip)
        end_time = utils.get_datetime_str_now()
        # TODO: Store some metadata?
        # TODO: Save the IQ data and return the file name
        return
    
    def generate_data_product(self):
        # Load IQ, process, return data product, for single channel
        # TODO
        iq = [] # Placeholder for iq data
        rec_id = '' # Placeholder identifier

        # Filter IQ data
        logger.debug(f"Applying IIR low-pass filter to IQ capture {rec_id}")
        iq = sosfilt(self.iir_sos, iq)

        # It should be possible to parallelize each of these
        mean_fft, max_fft = self.get_fft_results()
        apd_result = self.get_apd_results()
        td_pwr_result = self.get_td_power_results()

        return

    def get_fft_results(self, measurement_result: dict) -> Tuple[np.ndarray, np.ndarray]:
        # IQ data already scaled for calibrated gain
        fft_result = get_fft(
            time_data = measurement_result["data"],
            fft_size = self.fft_size,
            norm = 'forward',
            fft_window = self.fft_window,
            num_ffts = self.nffts,
            shift = False,
            workers = 1  # Configurable for parallelization
        )
        fft_result = calculate_pseudo_power(fft_result)
        fft_result = apply_power_detector(fft_result, self.fft_detector)  # First array is mean, second is max
        ne.evaluate("fft_result/50", out=fft_result)  # Finish conversion to Watts
        # Shift frequencies of reduced result
        fft_result = np.fft.fftshift(fft_result, axes=(1,))
        fft_result = convert_watts_to_dBm(fft_result)
        fft_result -= 3 # Baseband/RF power conversion
        fft_result -= 10. * np.log10(self.sample_rate_Hz)  # PSD scaling # TODO: Assure this is the correct sample rate
        fft_result += 20. * np.log10(self.fft_window_ecf)  # Window energy correction
        return fft_result[0], fft_result[1]

    def get_apd_results(self, measurement_result: dict):
        # TODO
        return None

    def get_td_power_results(self, measurement_result: dict):
        # TODO
        return None

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        # TODO (low-priority)
        return __doc__

    def get_sigmf_builder(self, measurement_result) -> SigMFBuilder:
        # TODO (low-priority)
        return super().get_sigmf_builder(measurement_result)

    def is_complex(self) -> bool:
        return False