from .acquire_sea_data_product import NasctnSeaDataProduct
from .acquire_sea_data_product_hybrid_sampling import HybridSeaDataProduct
from .acquire_single_freq_fft import SingleFrequencyFftAcquisition
from .acquire_single_freq_tdomain_iq import SingleFrequencyTimeDomainIqAcquisition
from .acquire_stepped_freq_tdomain_iq import SteppedFrequencyTimeDomainIqAcquisition
from .calibrate_y_factor import YFactorCalibration
from .calibrate_y_factor_hybrid_sampling import HybridSamplingYFactorCalibration

# Map a class name to an action class
# The YAML loader can key an object with parameters on these class names
action_classes = {
    "single_frequency_fft": SingleFrequencyFftAcquisition,
    "stepped_frequency_time_domain_iq": SteppedFrequencyTimeDomainIqAcquisition,
    "single_frequency_time_domain_iq": SingleFrequencyTimeDomainIqAcquisition,
    "y_factor_cal": YFactorCalibration,
    "nasctn_sea_data_product": NasctnSeaDataProduct,
    "hybrid_sea_data_product": HybridSeaDataProduct,
    "hybrid_y_factor_cal": HybridSamplingYFactorCalibration,
}
