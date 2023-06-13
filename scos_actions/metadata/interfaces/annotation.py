from dataclasses import dataclass
from typing import Optional

from scos_actions.metadata.interfaces.sigmf_object import SigMFObject


@dataclass
class AnnotationSegment(SigMFObject):
    """
    Interface for generating SigMF Annotation Segment objects.

    :param sample_start: The sample index at which this Segment takes effect.
    :param sample_count: The number of samples that this Segment applies to.
    :param generator: Human-readable name of the entity that created this
        annotation. Defaults to `"SCOS"` when using this interface.
    :param label: A short form human/machine-readable label for the annotation.
    :param comment: A human-readable comment.
    :param freq_lower_edge: The frequency, in Hz, of the lower edge of the feature
        described by this annotation.
    :param freq_upper_edge: The frequency, in Hz, of the upper edge of the feature
        described by this annotation.
    :param uuid: A RFC-4122 compliant UUID string of the form `xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx`.
    """

    sample_start: Optional[int] = None
    sample_count: Optional[int] = None
    generator: Optional[str] = "SCOS"
    label: Optional[str] = None
    comment: Optional[str] = None
    freq_lower_edge: Optional[float] = None
    freq_upper_edge: Optional[float] = None
    uuid: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.sample_start, "sample_start")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "sample_start": "core:sample_start",
                "sample_count": "core:sample_count",
                "generator": "core:generator",
                "label": "core:label",
                "comment": "core:comment",
                "freq_lower_edge": "core:freq_lower_edge",
                "freq_upper_edge": "core:freq_upper_edge",
            }
        )
        # Create metadata object
        super().create_json_object()
