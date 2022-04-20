def dbm_to_watts(input):
    # Convert an array of values from dBm to Watts
    #return 10 ** ((np.array(input) - 30) / 10)
    return (10. ** (np.array(input) / 10.)) / 1000.

def dBw_to_watts(val):
    return 10 ** (val / 10)

def get_enbw(window, Fs):
    # Return the equivalent noise bandwidth for a given window at a given sampling rate
    return Fs * np.sum(window ** 2) / np.sum(window) ** 2