from typing import Optional

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS, prepend_core_namespace


class AnnotationSegment(
    msgspec.Struct, rename=prepend_core_namespace, **SIGMF_OBJECT_KWARGS
):
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

    sample_start: int
    sample_count: Optional[int] = None
    generator: Optional[str] = "SCOS"
    label: Optional[str] = None
    comment: Optional[str] = None
    freq_lower_edge: Optional[float] = None
    freq_upper_edge: Optional[float] = None
    uuid: Optional[str] = None
