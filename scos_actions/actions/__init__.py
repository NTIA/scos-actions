from .acquire_single_freq_fft import SingleFrequencyFftAcquisition
from .acquire_stepped_freq_tdomain_iq import SteppedFrequencyTimeDomainIqAcquisition
from .acquire_single_freq_gps import SingleFrequencyGPSAcquisition
from .acquire_tdomain_iq_subdev import TimeDomainSubdev
from .transmit_pn import TransmitPN
from .transmit_cw import TransmitCW
from .reconnect import Reconnect
from .disconnect import Disconnect

# Map a class name to an action class
# The YAML loader can key an object with parameters on these class names
action_classes = {
    "single_frequency_fft": SingleFrequencyFftAcquisition,
    "stepped_frequency_time_domain_iq": SteppedFrequencyTimeDomainIqAcquisition,
    "single_frequency_gps": SingleFrequencyGPSAcquisition,
    "stepped_frequency_time_domain_subdev": TimeDomainSubdev,
    "transmit_pn": TransmitPN,
    "transmit_cw": TransmitCW,
    "reconnect": Reconnect,
    "disconnect": Disconnect,
}
