from .amplitude_probability_distribution import AmplitudeProbabilityDistribution
from .periodic_frame_power import PeriodicFramePower, PeriodicFramePowerRayActor
from .power_spectral_density import PowerSpectralDensityRayActor
from .power_vs_time import PowerVsTime

product_classes = {
    "psd": PowerSpectralDensityRayActor,
    "pvt": PowerVsTime,
    "pfp": PeriodicFramePowerRayActor,
    "apd": AmplitudeProbabilityDistribution,
}