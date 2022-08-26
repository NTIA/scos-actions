from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional

from scos_actions.metadata.sigmf_builder import SigMFBuilder


@dataclass
class AnnotationSegment(ABC):
    """
    Interface for generating SigMF annotation segments.

    The only required parameter is ``sample_start``. Refer to the SigMF
    documentation for more information.

    :param sample_start: The sample index at which the segment takes effect.
    :param sample_count: The number of samples to which the segment applies.
    :param generator: Human-readable name of the entity that created this annotation.
    :param label: A short form human/machine-readable label for the annotation.
    :param comment: A human-readable comment.
    :param freq_lower_edge: The frequency (Hz) of the lower edge of the feature
        described by this annotation.
    :param freq_upper_edge: The frequency (Hz) of the upper edge of the feature
        described by this annotation.
    """

    sample_start: Optional[int] = None
    sample_count: Optional[int] = None
    generator: Optional[str] = None
    label: Optional[str] = None
    comment: Optional[str] = None
    freq_lower_edge: Optional[float] = None
    freq_upper_edge: Optional[float] = None

    def __post_init__(self):
        # Initialization
        self.annotation_type = self.__class__.__name__
        self.segment = {"ntia-core:annotation_type": self.annotation_type}
        self.required_err_msg = (
            f"{self.annotation_type} segments require a value to be specified for "
        )
        # Ensure required keys have been set
        self.check_required(self.sample_start, "sample_start")
        if self.freq_lower_edge is not None or self.freq_upper_edge is not None:
            err_msg = "Both freq_lower_edge and freq_upper_edge must be provided if one is provided."
            assert (
                self.freq_lower_edge is not None and self.freq_upper_edge is not None
            ), err_msg
        # Define SigMF key names
        self.sigmf_keys = {
            "sample_start": "core:sample_start",
            "sample_count": "core:sample_count",
            "generator": "core:generator",
            "label": "core:label",
            "comment": "core:comment",
            "freq_lower_edge": "core:freq_lower_edge",
            "freq_upper_edge": "core:freq_upper_edge",
            "recording": "core:recording",
        }

    def check_required(self, value: Any, keyname: str) -> None:
        assert value is not None, self.required_err_msg + keyname

    def create_annotation_segment(self) -> None:
        meta_vars = vars(self)
        for varname, value in meta_vars.items():
            if value is not None:
                try:
                    sigmf_key = meta_vars["sigmf_keys"][varname]
                    self.segment[sigmf_key] = value
                except KeyError:
                    pass
        return

    def create_metadata(self, sigmf_builder: SigMFBuilder) -> None:
        sigmf_builder.add_annotation(self.sample_start, self.sample_count, self.segment)
