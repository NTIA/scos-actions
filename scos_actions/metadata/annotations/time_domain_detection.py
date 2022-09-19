from dataclasses import dataclass
from typing import Optional

from scos_actions.metadata.annotation_segment import AnnotationSegment


@dataclass
class TimeDomainDetection(AnnotationSegment):
    """
    Interface for generating TimeDomainDetection annotation segments.

    Refer to the documentation of the ``ntia-algorithm`` extension of
    SigMF for more information.

    The parameters ``detector``, ``number_of_samples``, and ``units`` are
    required.

    :param detector: Detector type, e.g. ``"sample_power"``, ``"mean_power"``,
        ``"max_power"``, etc.
    :param number_of_samples: Number of samples integrated over by the detector.
    :param units: Data units, e.g. ``"dBm"``, ``"watts"``, etc.
    :param reference: Data reference point, e.g. ``"signal analyzer input"``,
        ``"preselector input"``, ``"antenna terminal"``, etc.
    """

    detector: Optional[str] = None
    number_of_samples: Optional[int] = None
    units: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.detector, "detector")
        self.check_required(self.number_of_samples, "number_of_samples")
        self.check_required(self.units, "units")
        # Define SigMF key names
        self.sigmf_keys.update(
            {
                "detector": "ntia-algorithm:detector",
                "number_of_samples": "ntia-algorithm:number_of_samples",
                "units": "ntia-algorithm:units",
                "reference": "ntia-algorithm:reference",
            }
        )
        # Create annotation segment
        super().create_annotation_segment()
