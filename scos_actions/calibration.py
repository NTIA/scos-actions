import json
import logging


logger = logging.getLogger(__name__)


class Calibration(object):
    def __init__(
            self,
            calibration_datetime,
            calibration_params,
            calibration_data,
            clock_rate_lookup_by_sample_rate
    ):
        self.calibration_datetime = calibration_datetime
        self.calibration_parameters = calibration_params
        self.calibration_data = calibration_data
        self.clock_rate_lookup_by_sample_rate = clock_rate_lookup_by_sample_rate

    def get_clock_rate(self, sample_rate):
        """Find the clock rate (Hz) using the given sample_rate (samples per second)"""
        for mapping in self.clock_rate_lookup_by_sample_rate:
            mapped = get_comparable_value(mapping["sample_rate"])
            actual = get_comparable_value(sample_rate)
            if mapped == actual:
                return mapping["clock_frequency"]
        return sample_rate

    def get_values(self, sample_rate, args):
        values = self.calibration_data[sample_rate]
        for i in len(args):
            values = values[args[i]]
        return values

    def get_calibration_dict(self,args):
        """Find the calibration points closest to the specified args (gain, attenuation,ref_level...)."""

        # Check if the sample rate was calibrated
        cal_data = self.calibration_data
        settings_used =[]
        next_setting_levels=[]
        for i in range(len(args)):
            setting_value = args[i]
            setting = self.calibration_parameters[i]
            logger.debug('looking up calibration for {} at {}'.format(setting, setting_value))
            cal_data = filter_by_parameter(cal_data, setting, setting_value)

        return cal_data

    def update(self, params, calibration_datetime, gain, noise_figure, file_path):
        cal_data = self.calibration_data
        self.calibration_datetime = calibration_datetime
        for parameter in self.calibration_parameters:
            if parameter in params:
                value = params[parameter]
                logger.debug('Updating calibration at {} = {}'.format(parameter, value))
                cal_data = cal_data[value]
        if 'gain_sensor' in cal_data:
            cal_data['gain_sensor'] = gain
        else:
            raise Exception('Not enough parameters specified to update sensor gain')
        if 'noise_figure_sensor' in cal_data:
            cal_data['noise_figure_sensor'] = noise_figure
        else:
            raise Exception('Not enough parameters specified to update sensor noise figure')
        dict = {'calibration_datetime': str(self.calibration_datetime),
                'calibration_parameters': self.calibration_parameters,
                'clock_rate_lookup_by_sample_rate':  self.clock_rate_lookup_by_sample_rate,
                'calibration_data': self.calibration_data}
        with open(file_path, 'w') as outfile:
            outfile.write(json.dumps(dict))

def get_comparable_value(f):
    """Allow a frequency of type [float] to be compared with =="""
    f = int(round(f))
    return f


def load_from_json(fname):
    with open(fname) as file:
        calibration = json.load(file)

    # Check that the required fields are in the dict
    assert "calibration_datetime" in calibration
    assert "calibration_data" in calibration
    assert "clock_rate_lookup_by_sample_rate" in calibration
    calibration_data = convert_keys(calibration['calibration_data'])
    # Create and return the Calibration object
    return Calibration(
        calibration["calibration_datetime"],
        calibration['calibration_parameters'],
        calibration_data,
        calibration["clock_rate_lookup_by_sample_rate"],
    )



def convert_keys(dictionary):

    if  isinstance(dictionary, dict):
        keys = list(dictionary.keys())
        for key in keys:
            vals = dictionary[key]
            try:
                new_key = float(key)
                vals = convert_keys(vals)
                dictionary.pop(key)
                dictionary[new_key] = vals
            except Exception:
                vals = convert_keys(vals)
                dictionary.pop(key)
                dictionary[key] = vals
        return dictionary
    else:
        return dictionary

def filter_by_parameter( calibrations, parameter, value):
        if not value in calibrations:
            raise Exception('No calibration was performed with {} at {}'.format(parameter, value))
        else:
            return calibrations[value]



