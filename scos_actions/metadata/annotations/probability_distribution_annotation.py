from dataclasses import dataclass
from typing import List, Optional

from scos_actions.metadata.annotation_segment import AnnotationSegment
from scos_actions.metadata.sigmf_builder import SigMFBuilder


@dataclass
class ProbabilityDistributionAnnotation(AnnotationSegment):
    """
    Interface for generating ProbabilityDistributionAnnotation segments.

    Refer to the documentation of the ``ntia-algorithm`` extension of
    SigMF for more information.

    The parameters ``function``, ``units``, and ``probability_units``
    are required.

    :param function: The estimated probability distribution function, e.g.
        ``"cumulative distribution"``, ``"probability density"``,
        ``"amplitude probability distribution"``.
    :param units: Data units, e.g. ``"dBm"``, ``"volts"``, ``"watts"``.
    :param probability_units: Unit of the probability values, generally
        either ``"dimensionless"`` or ``"percent"``.
    :param number_of_samples: Number of samples used to estimate the
        probability distribution function. In the case of a downsampled
        result, this number may be larger than the length of the annotated
        data.
    :param reference: Data reference point, e.g. ``"signal analyzer input"``,
        ``"preselector input"``, ``"antenna terminal"``.
    :param probability_start: Probability of the first data point, in units
        specified by ``probability_units``.
    :param probability_stop: Probability of the last data point, in units
        specified by ``probability_units``.
    :param probability_step: Step size, in ``probability_units``, between
        data points. This should only be used if the step size is constant
        across all data points.
    :param probabilities: A list of the probabilities for all data points.
        This must be used if the probability step size is not constant.
    :param downsampled: Whether or not the probability distribution data
        has been downsampled.
    :param downsampling_method: The method used for downsampling, e.g.
        ``"uniform downsampling by a factor of 2"``, etc.
    """

    function: Optional[str] = None
    units: Optional[str] = None
    probability_units: Optional[str] = None
    number_of_samples: Optional[int] = None
    reference: Optional[str] = None
    probability_start: Optional[float] = None
    probability_stop: Optional[float] = None
    probability_step: Optional[float] = None
    probabilities: Optional[List[float]] = None
    downsampled: Optional[bool] = None
    downsampling_method: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.function, "function")
        self.check_required(self.units, "units")
        self.check_required(self.probability_units, "probability_units")
        # Define SigMF key names
        self.sigmf_keys.update(
            {
                "function": "ntia-algorithm:function",
                "units": "ntia-algorithm:units",
                "probability_units": "ntia-algorithm:probability_units",
                "number_of_samples": "ntia-algorithm:number_of_samples",
                "reference": "ntia-algorithm:reference",
                "probability_start": "ntia-algorithm:probability_start",
                "probability_stop": "ntia-algorithm:probability_stop",
                "probability_step": "ntia-algorithm:probability_step",
                "probabilities": "ntia-algorithm:probabilities",
                "downsampled": "ntia-algorithm:downsampled",
                "downsampling_method": "ntia-algorithm:downsampling_method",
            }
        )
        # Create annotation segment
        super().create_annotation_segment()
