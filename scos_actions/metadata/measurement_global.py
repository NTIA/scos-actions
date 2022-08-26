from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from scos_actions.metadata.sigmf_builder import SigMFBuilder


@dataclass
class MeasurementMetadata:
    """
    Interface for generating SigMF ntia-core Measurement objects.

    The parameters ``domain``, ``measurement_type``, ``time_start``,
    ``time_stop``, ``frequency_tuned_low``, ``frequency_tuned_high``, and
    ``classification`` are required. Refer to the documentation for the
    ``ntia-core`` extension of SigMF for more information.

    :param domain: Measurement domain, generally ``"time"`` or ``"frequency"``.
    :param measurement_type: Method by which the signal analyzer acquires data:
        ``"single-frequency"`` or ``"scan"``.
    :param time_start: When the action began execution.
    :param time_stop: When the action finished execution.
    :param frequency_tuned_low: Lowest tuned frequency (Hz).
    :param frequency_tuned_high: Highest tuned frequency (Hz).
    :param frequency_tuned_step: Step between tuned frequencies of a ``"scan"``
        measurement. Either ``frequency_tuned_step`` or ``frequencies_tuned``
        SHOULD be included for ``"scan"`` measurements.
    :param classification: The classification markings for the acquisition, e.g.
        ``"UNCLASSIFIED"``, ``"CONTROLLED//FEDCON"``, ``"SECRET"``, etc.
    """

    domain: Optional[str]
    measurement_type: Optional[str]
    time_start: Optional[datetime]
    time_stop: Optional[datetime]
    frequency_tuned_low: Optional[float]
    frequency_tuned_high: Optional[float]
    frequency_tuned_step: Optional[float] = None
    frequencies_tuned: Optional[List[float]] = None
    classification: Optional[str] = None

    def __post_init__(self):
        # Ensure required keys have been set
        self.check_required(self.domain, "domain")
        self.check_required(self.measurement_type, "measurement_type")
        self.check_required(self.time_start, "time_start")
        self.check_required(self.time_stop, "time_stop")
        self.check_required(self.frequency_tuned_low, "frequency_tuned_low")
        self.check_required(self.frequency_tuned_high, "frequency_tuned_high")
        self.check_required(self.classification, "classification")
        # Define SigMF key names
        self.sigmf_keys = {
            "domain": "domain",
            "measurement_type": "measurement_type",
            "time_start": "time_start",
            "time_stop": "time_stop",
            "frequency_tuned_low": "frequency_tuned_low",
            "frequency_tuned_high": "frequency_tuned_high",
            "frequencies_tuned": "frequencies_tuned",
            "classification": "classification",
        }

    def check_required(self, value: Any, keyname: str) -> None:
        assert value is not None, (
            "Measurement metadata requires a value to be specified for " + keyname
        )

    def create_metadata(self, sigmf_builder: SigMFBuilder):
        segment = {}
        meta_vars = vars(self)
        for varname, value in meta_vars.items():
            if value is not None:
                try:
                    sigmf_key = meta_vars["sigmf_keys"][varname]
                    segment[sigmf_key] = value
                except KeyError:
                    pass
        sigmf_builder.add_to_global("ntia-core:measurement", segment)
