"""
Dataclass implementation for "differential calibration" handling.

A differential calibration provides loss values at different frequencies
which represent excess loss between the calibration terminal and the antenna
port. At present, this is measured manually using a calibration probe consisting
of a calibrated noise source and a programmable attenuator.

The ``reference_point`` top-level key defines the point to which measurements
are referenced after using the correction factors included in the file.

The ``calibration_data`` entries are expected to include these correction factors,
with the key name ``"differential_loss"`` and values in decibels (dB). These correction
factors represent the differential loss between the calibration terminal used by onboard
``SensorCalibration`` results and the reference point defined by ``reference_point``.
"""

from dataclasses import dataclass

from scos_actions.calibration.interfaces.calibration import Calibration


@dataclass
class DifferentialCalibration(Calibration):
    reference_point: str

    def update(self):
        """
        SCOS Sensor should not update differential calibration files.

        Instead, these should be generated through an external calibration
        process. This class should only be used to read JSON files, and never
        to update their entries. Therefore, no ``update`` method is implemented.

        If, at some point in the future, functionality is added to automate these
        calibrations, this function may be implemented.
        """
        raise NotImplementedError
