from .amplitude_probability_distribution import AmplitudeProbabilityDistribution
from .periodic_frame_power import PeriodicFramePower
from .power_spectral_density import PowerSpectralDensity
from .power_vs_time import PowerVsTime

product_classes = {
    "psd": PowerSpectralDensity,
    "pvt": PowerVsTime,
    "pfp": PeriodicFramePower,
    "apd": AmplitudeProbabilityDistribution,
}