from scos_actions.metadata.metadata import Metadata
from scos_actions.metadata.sigmf_builder import SigMFBuilder

# This annotation is used to annotate the APD result
# in the absence of full support for probability distribution
# annotations.


class SEADataProductAnnotation(Metadata):
    def __init__(self, start, count, apd_bin_size):
        super().__init__(start, count)
        self.apd_bin_size = apd_bin_size

    def create_metadata(self, sigmf_builder: SigMFBuilder, measurement_result: dict):
        return super().create_metadata(sigmf_builder, measurement_result)
