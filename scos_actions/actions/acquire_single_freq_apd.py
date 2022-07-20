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
r"""Estimate the amplitude probability distribution of {num_samps} continuous time domain power samples at {center_frequency:.2f} MHz.

# To-do: write comprehensive module docstring

# {name}

## Signal Analyzer setup and sample acquisition

Each time this task runs, the following process is followed:
{acquisition_plan}

## Time-domain processing

First, the {num_samps} continuous samples are acquired from the signal analyzer. If
specified, a voltage scaling factor is applied to the time-domain signals. The time
domain complex voltage samples are converted to real-valued amplitudes which are used to
generate the APD. After the APD is generated, the amplitude values are further converted
to power values in dBm, with calculations assuming a 50 Ohm system. 

If no downsampling is specified (by either not providing the apd_bin_size_dB parameter, or
setting its value to be less than or equal to zero), the APD is estimated using the routine
provided in [IEEE P802.15-04-0428-00](https://www.ieee802.org/15/pub/2004/15-04-0428-00-003a-estimating-and-graphing-amplitude-probability-distribution-function.pdf).

If downsampling is specified, the APD is estimated by sampling the complementary
cumulative distribution function (CCDF) of the sample amplitudes. Bin edges are generated based on the specified value
of the apd_bin_size_dB parameter and the minimum and maximum recorded amplitude samples. The fraction of
amplitude samples exceeding each bin edge is computed, and used to construct an estimate of the APD.

## Power Conversion

After the APD is estimated, the amplitude values are converted to power values in dBm, assuming
a 50 Ohm system. A 3 dB power correction is applied to account for RF/baseband conversion. The resulting
power values are therefore referenced to the calibration point.

## Results

The current implementation stores a concatenated data array containing both the probability
and amplitude axes of the estimated APD. This should be altered as future updates to SigMF extensions
provide better ways to store such data.
"""
import logging
import numexpr as ne
import numpy as np
from scos_actions.actions.interfaces.measurement_action import (MeasurementAction)
from scos_actions.actions.action_utils import get_param, ParameterException
from scos_actions.actions.sigmf_builder import Domain, MeasurementType, SigMFBuilder
from scos_actions.signal_processing.apd import get_apd
from scos_actions.signal_processing.unit_conversion import convert_linear_to_dB
from scos_actions.hardware import gps as mock_gps
from scos_actions import utils


logger = logging.getLogger(__name__)

# Define parameter keys
FREQUENCY = "frequency"
SAMPLE_RATE = "sample_rate"
DURATION_MS = "duration_ms"
NUM_SKIP = "nskip"
APD_BIN_SIZE = "apd_bin_size_dB"


class SingleFrequencyApdAcquisition(MeasurementAction):
    """Performs an APD analysis of the requested number of samples at the specified sample rate.

    The action will set any matching attributes found in the signal analyzer object. The following
    parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        duration_ms: duration to acquire in ms
        apd_bin_size_dB: amplitude resolution of IQ data, reduces data size. defaults to 0.5
         
    For the parameters required by the signal analyzer, see the documentation from the Python
    package for the signal analyzer being used.

    :param parameters: Dictionary of parameters needed for the action and signal analyzer.
    :param sigan: instance of SignalAnalyzerInterface
    """
    def __init__(self, parameters, sigan, gps=mock_gps):
        super().__init__(parameters, sigan, gps)
        # Pull parameters from action config
        self.nskip = get_param(NUM_SKIP, self.parameter_map)
        self.duration_ms = get_param(DURATION_MS, self.parameter_map)
        self.frequency_Hz = get_param(FREQUENCY, self.parameter_map)
        self.sample_rate_Hz = get_param(SAMPLE_RATE, self.parameter_map)
        self.num_samples = int(self.sample_rate_Hz * self.duration_ms * 1e-3)
        try:
            self.apd_bin_size = get_param(APD_BIN_SIZE, self.parameter_map)
        except ParameterException:
            logger.warning("APD bin size not configured. Using no downsampling.")
            self.apd_bin_size = None
        if self.apd_bin_size <= 0:
            logger.warning("APD bin size set to zero or negative. Using no downsampling.")
            self.apd_bin_size = None

    def execute(self, schedule_entry, task_id) -> dict:
        # Acquire IQ data and generate APD result
        start_time = utils.get_datetime_str_now()
        measurement_result = self.acquire_data(self.num_samples, self.nskip)
        apd_result = self.get_power_apd(measurement_result)
        # apd_result is (p, a) concatenated into a single array

        # Save measurement results
        measurement_result["data"] = apd_result
        measurement_result.update(self.parameter_map)
        measurement_result['start_time'] = start_time
        measurement_result['end_time'] = utils.get_datetime_str_now()
        measurement_result["domain"] = Domain.TIME.value
        measurement_result['measurement_type'] = MeasurementType.SINGLE_FREQUENCY.value
        measurement_result['description'] = self.description
        measurement_result['calibration_datetime'] = self.sigan.sensor_calibration_data['calibration_datetime']
        measurement_result['task_id'] = task_id
        measurement_result['sigan_cal'] = self.sigan.sigan_calibration_data
        measurement_result['sensor_cal'] = self.sigan.sensor_calibration_data
        return measurement_result

    def get_power_apd(self, measurement_result: dict) -> np.ndarray:
        # Calibration gain scaling already applied to IQ samples
        p, a = get_apd(measurement_result["data"], self.apd_bin_size)
        # Scaling applied:
        #  a * 2 : dBV --> dB(V^2)
        #  a - impedance_dB : dB(V^2) --> dBW (hard-coded for 50 Ohm systems)
        #  a + 27 : dBW --> dBm (+30), RF/baseband conversion (-3)
        scale_factor = 27 - convert_linear_to_dB(50.)
        ne.evaluate("(2*a)+scale_factor", out=a)
        # For now: concatenate axes to store as a single array
        return np.concatenate((p, a))

    @property
    def description(self):
        """Parameterize and return the module-level docstring."""
        frequency_MHz = self.frequency_Hz / 1e6
        used_keys = [FREQUENCY, APD_BIN_SIZE, "name"]
        acq_plan = f"The signal analyzer is tuned to {frequency_MHz:.2f} MHz and the following parameters are set:\n"
        for name, value in self.parameters.items():
            if name not in used_keys:
                acq_plan += f"{name} = {value}\n"

        definitions = {
            "name": self.name,
            "num_samps": self.num_samples,
            "center_frequency": frequency_MHz,
            "acquisition_plan": acq_plan,
            "apd_bin_size_dB": self.apd_bin_size_dB  # Currently unused in docstring
        }

        # __doc__ refers to the module docstring at the top of the file
        return __doc__.format(**definitions)


    def get_sigmf_builder(self, measurement_result: dict) -> SigMFBuilder:
        # TO DO
        return super().get_sigmf_builder(measurement_result)


    def is_complex(self) -> bool:
        return False