def get_num_samples_and_fft_size(params: dict) -> tuple:
    num_ffts, fft_size, num_samples = None, None, None
    if "nffts" not in params:
        raise Exception("nffts missing from measurement parameters")
    num_ffts = params["nffts"]
    if "fft_size" not in params:
        raise Exception("fft_size missing from measurement parameters")
    fft_size = params["fft_size"]
    num_samples = num_ffts * fft_size
    return num_samples, fft_size


def get_num_skip(params):
    nskip = None
    if "nskip" in params:
        nskip = params["nskip"]
    else:
        raise Exception("nskip missing from measurement parameters")
    return nskip
