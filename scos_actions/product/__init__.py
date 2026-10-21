from .amplitude_probability_distribution import (
    AmplitudeProbabilityDistribution,
    AmplitudeProbabilityDistributionRayActor,
)
from .periodic_frame_power import PeriodicFramePower, PeriodicFramePowerRayActor
from .power_spectral_density import PowerSpectralDensityRayActor
from .power_vs_time import PowerVsTime, PowerVsTimeRayActor

product_classes = {
    "psd": PowerSpectralDensityRayActor,
    "pvt": PowerVsTimeRayActor,
    "pfp": PeriodicFramePowerRayActor,
    "apd": AmplitudeProbabilityDistributionRayActor,
}