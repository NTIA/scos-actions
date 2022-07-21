
def get_num_samples_and_fft_size( params):
    if not "nffts" in params:
        raise Exception("nffts missing from measurement parameters")
    num_ffts = params["nffts"]
    if not "fft_size" in  params:
        raise Exception("fft_size missing from measurement parameters")
    fft_size = params["fft_size"]
    num_samples = num_ffts * fft_size
    return num_samples, fft_size


def get_num_skip( params):
    nskip = None
    if "nskip" in params:
        nskip = params["nskip"]
    else:
        raise Exception("nskip missing from measurement parameters")
    return nskip