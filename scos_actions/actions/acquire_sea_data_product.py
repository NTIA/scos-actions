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
from scos_actions.actions.sigmf_builder import SigMFBuilder
from scos_actions.actions.interfaces.measurement_action import MeasurementAction
from scos_actions.actions.action_utils import get_param
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
        self.sorted_measurement_parameters = []
        self.num_center_frequencies = len(parameters[FREQUENCY])
        # convert dictionary of lists from yaml file to list of dictionaries
        longest_length = 0
        for key, value in parameters.items():
            if key == "name":
                continue
            if len(value) > longest_length:
                longest_length = len(value)
        for i in range(longest_length):
            sorted_params = {}
            for key in parameters.keys():
                if key == "name":
                    continue
                sorted_params[key] = parameters[key][i]
            self.sorted_measurement_parameters.append(sorted_params)
        self.sorted_measurement_parameters.sort(key=lambda params: params[FREQUENCY])

        self.sigan = sigan  # make instance variable to allow mocking

        # Setup/pull config parameters
        # TODO: Create ability to define single values for some params, multiple for others
        self.iir_apply = get_param(IIR_APPLY, self.sorted_measurement_parameters)
        self.iir_rp_dB = get_param(RP_DB, self.sorted_measurement_parameters)
        self.iir_rs_dB = get_param(RS_DB, self.sorted_measurement_parameters)
        self.iir_cutoff_Hz = get_param(IIR_CUTOFF_HZ, self.sorted_measurement_parameters)
        self.iir_width_Hz = get_param(IIR_WIDTH_HZ, self.sorted_measurement_parameters)
        self.qfilt_apply = get_param(QFILT_APPLY, self.sorted_measurement_parameters)
        self.qfilt_qlo = get_param(Q_LO, self.sorted_measurement_parameters)
        self.qfilt_qhi = get_param(Q_HI, self.sorted_measurement_parameters)
        self.fft_size = get_param(FFT_SIZE, self.sorted_measurement_parameters)
        self.nffts = get_param(NUM_FFTS, self.sorted_measurement_parameters)
        self.fft_window_type = get_param(FFT_WINDOW_TYPE, self.sorted_measurement_parameters)
        self.apd_bin_size_dB = get_param(APD_BIN_SIZE_DB, self.sorted_measurement_parameters)
        self.td_bin_size_ms = get_param(TD_BIN_SIZE_MS, self.sorted_measurement_parameters)
        # TODO: check that all sample rates are the same
        # TODO: overall- how to handle parameters that are the same for all frequencies?
        self.sample_rate_Hz = get_param(SAMPLE_RATE, self.sorted_measurement_parameters[0])

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
        self.test_required_components()

        # Step through frequencies and acquire IQ data
        for recording_id, measurement_params in enumerate(self.sorted_measurement_parameters, start=1):
            start_time = utils.get_datetime_str_now()
            # Apply sigan configuration
            self.configure(measurement_params)
            duration_ms = get_param(DURATION_MS, measurement_params)
            nskip = get_param(NUM_SKIP, measurement_params)
            sample_rate = self.sigan.sample_rate
            num_samples = int(sample_rate * duration_ms * 1e-3)
            # Acquire IQ data
            measurement_result = super().acquire_data(num_samples, nskip)
            # Save IQ data
            # TODO


        # Loop through acquired data and generate data product
        for recording_id, measurement_params in enumerate(self.sorted_measurement_parameters, start=1):
            # TODO
            iq = [] # Placeholder for loaded IQ data
            logger.debug(f"Applying IIR low-pass filter to IQ capture {recording_id}")
            iq = sosfilt(self.iir_sos, iq)

            # It should be possible to parallelize these tasks
            mean_fft_result, max_fft_result = self.get_fft_results(iq)
            apd_result = self.get_apd_results(iq)
            td_pwr_result = self.get_td_power_results(iq)

        # Save data product
        # TODO

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