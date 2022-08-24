import copy
import json
import os
import sys


def interpolate_1d(x, x1, x2, y1, y2):
    """Interpolate between points in one dimension."""
    return y1 * (x2 - x) / (x2 - x1) + y2 * (x - x1) / (x2 - x1)

def interpolate_2d(x, y, x1, x2, y1, y2, z11, z21, z12, z22):
    """Interpolate between points in two dimensions."""
    z_y1 = interpolate_1d(x, x1, x2, z11, z21)
    z_y2 = interpolate_1d(x, x1, x2, z12, z22)
    return interpolate_1d(y, y1, y2, z_y1, z_y2)

def convert_to_int(f):
    """Allow a frequency of type [float] to be compared with =="""
    f = int(round(f))
    return f
    
def get_calibration_dict(calibration_data, cal_args, calibration_frequency_divisions=[]):
    """Find the calibration points closest to the current settings."""
    setting0 = cal_args[0] # originally sample rate
    setting0_name = calibration_data["calibration_parameters"][0]
    setting1 = cal_args[1] # originally frequency
    setting1_name = calibration_data["calibration_parameters"][1]
    setting2 = cal_args[2] # originally gain
    setting2_name = calibration_data["calibration_parameters"][2]
    cal_data = calibration_data["calibration_data"]
    setting0_keys = list(calibration_data["calibration_data"].keys())
    if setting0 not in setting0_keys:
        print(f"Requested {setting0_name} was not calibrated!")
        print(f"Assuming default {setting0_name}:")
        print(f"    Requested {setting0_name}: {setting0}")
        print(f"    Assumed {setting0_name}:   {setting0_keys[0]}")
        setting0 = setting0_keys[0]

    # Get the nearest calibrated frequency and its index
    setting1_i = -1
    setting1_keys = sorted(calibration_data["calibration_data"][setting0].keys())
    bypass_setting1_interpolation = True
    if float(setting1) < float(setting1_keys[0]):  # Frequency below calibrated range
        setting1_i = 0
        print(f"Tuned {setting1_name} is below calibrated range!")
        print(f"Assuming lowest {setting1_name}")
        print(f"    Tuned {setting1_name}:   {setting1}")
        print(f"    Assumed {setting1_name}: {setting1_keys[setting1_i]}")
    elif float(setting1) > float(setting1_keys[-1]):  # Frequency above calibrated range
        setting1_i = len(setting1_keys) - 1
        print(f"Tuned {setting1_name} is above calibrated range!")
        print(f"Assuming highest {setting1_name}:")
        print(f"    Tuned {setting1_name} :   {setting1}".format(setting1))
        print(f"    Assumed {setting1_name} : {setting1_keys[setting1_i]}")
    else:
        # Ensure we use frequency interpolation
        bypass_setting1_interpolation = False
        # Check if we are within a frequency division
        # for div in calibration_frequency_divisions:
        #     if setting1 > div["lower_bound"] and setting1 < div["upper_bound"]:
        #         print("Tuned frequency within a division!")
        #         print("Assuming frequency at lower bound:")
        #         print("    Tuned frequency:   {}".format(setting1))
        #         print(
        #             "    Lower bound:       {}".format(div["lower_bound"])
        #         )
        #         print(
        #             "    Upper bound:       {}".format(div["upper_bound"])
        #         )
        #         print(
        #             "    Assumed frequency: {}".format(div["lower_bound"])
        #         )
        #         setting1 = div[
        #             "lower_bound"
        #         ]  # Interpolation will force this point; no interpolation error
        # Determine the index associated with the closest setting1 key less than or equal to setting1
        for i in range(len(setting1_keys) - 1):
            setting1_i = i
            # If the next setting1 is larger, we're done
            if float(setting1_keys[i + 1]) > float(setting1):
                break

    setting2_i = -1
    setting2_fudge = 0
    setting2_keys = sorted(cal_data[setting0][setting1_keys[setting1_i]].keys())
    bypass_setting2_interpolation = True
    if float(setting2) < float(setting2_keys[0]): # below calibration range
        setting2_i = 0
        setting2_fudge = setting2 - setting2_keys[0]
        print(f"Current {setting2_name} is below calibrated range!")
        print(f"Assuming lowest {setting2_name} and extending:")
        print(f"    Current {setting2_name}: {setting2}")
        print(f"    Assumed {setting2_name}: {setting2_keys[0]}")
        print(f"    Fudge factor: {setting2_fudge}")
    elif float(setting2) > float(setting2_keys[-1]): # above calibrated range
        setting2_i = len(setting2_keys) - 1
        setting2_fudge = setting2 - setting2_keys[-1]
        print(f"Current {setting2_name}is above calibrated range!")
        print(f"Assuming lowest {setting2_name} and extending:")
        print(f"    Current {setting2_name}: {setting2}")
        print(f"    Assumed {setting2_name}: {setting2_keys[-1]}")
        print(f"    Fudge factor: {setting2_fudge}")
    else:
        bypass_setting2_interpolation = False
        # Determine the index associated with the closest setting2 key less than or equal to setting2
        for i in range(len(setting2_keys) - 1):
            setting2_i = i
            # If the next setting2 is larger, we're done
            if float(setting2_keys[i + 1]) > float(setting2):
                break

    # Get the list of calibration factors
    calibration_factors = cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i]].keys()

    # Interpolate as needed for each calibration point
    interpolated_calibration = {}
    for cal_factor in calibration_factors:
        if bypass_setting2_interpolation and bypass_setting1_interpolation:
            factor = float(cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i]][cal_factor])
        elif bypass_setting1_interpolation:
            factor = interpolate_1d(
                float(setting2),
                float(setting2_keys[setting2_i]),
                float(setting2_keys[setting2_i + 1]),
                float(cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i]][cal_factor]),
                float(cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i + 1]][cal_factor]),
            )
        elif bypass_setting2_interpolation:
            factor = interpolate_1d(
                float(setting1),
                float(setting1_keys[setting1_i]),
                float(setting1_keys[setting1_i + 1]),
                float(cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i]][cal_factor]),
                float(cal_data[setting0][setting1_keys[setting1_i + 1]][setting2_keys[setting2_i]][cal_factor]),
            )
        else:
            factor = interpolate_2d(
                float(setting1),
                float(setting2),
                float(setting1_keys[setting1_i]),
                float(setting1_keys[setting1_i + 1]),
                float(setting2_keys[setting2_i]),
                float(setting2_keys[setting2_i + 1]),
                float(cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i]][cal_factor]),
                float(cal_data[setting0][setting1_keys[setting1_i + 1]][setting2_keys[setting2_i]][cal_factor]),
                float(cal_data[setting0][setting1_keys[setting1_i]][setting2_keys[setting2_i + 1]][cal_factor]),
                float(cal_data[setting0][setting1_keys[setting1_i + 1]][setting2_keys[setting2_i + 1]][cal_factor]),
            )

        # Apply the fudge factor based off the calibration type
        if "gain" in cal_factor:
            factor += setting2_fudge
        elif "noise_figure" in cal_factor:
            factor -= setting2_fudge
        elif "compression" in cal_factor:
            factor -= setting2_fudge
        else:
            "Unable to apply fudge factor. Unknown calibration type"

        # Add the calibration factor to the interpolated list
        interpolated_calibration[cal_factor] = factor

    # Return the interpolated calibration factors
    return interpolated_calibration

def add_interpolated_cal():
    if len(sys.argv) < 2:
        print("Path to cal file must be first argument.")
        print("Usage: python add_interpolated_cal cal_file [param1] [param2] [param3]")
        sys.exit(1)
    if len(sys.argv) > 5:
        print("Currently only 3 settings supported.")
        print("Usage: python add_interpolated_cal cal_file [param1] [param2] [param3]")


    cal_file = sys.argv[1]
    calibration_data = {}
    with open(cal_file) as in_f:
        calibration_data = json.load(in_f)
    print(calibration_data.keys())
    calibration_data_copy = copy.deepcopy(calibration_data)


    cal_args = [sys.argv[2], sys.argv[3], sys.argv[4]]
    cal_args_str = ", ".join([str(x) for x in cal_args])
    print(f"cal_args = {cal_args_str}")

    interpolated_calibration = get_calibration_dict(calibration_data, cal_args)
    print(interpolated_calibration)
    #calibration_data_copy[]
    cal_data = calibration_data_copy["calibration_data"]
    if str(cal_args[0]) not in cal_data:
        cal_data[str(cal_args[0])] = {}
    if str(cal_args[1]) not in cal_data[str(cal_args[0])]:
        cal_data[str(cal_args[0])][str(cal_args[1])] = {}
    if str(cal_args[2]) not in cal_data[str(cal_args[0])][str(cal_args[1])]:
        cal_data[str(cal_args[0])][str(cal_args[1])][str(cal_args[2])] = {}
    cal_data[str(cal_args[0])][str(cal_args[1])][str(cal_args[2])].update(interpolated_calibration)
    file_path, ext = os.path.splitext(cal_file)
    new_file = file_path + "_updated" + ext
    with open(new_file, 'w') as out_f:
        json.dump(calibration_data_copy, out_f, indent=4, sort_keys=True)


if __name__ == "__main__":
    add_interpolated_cal()