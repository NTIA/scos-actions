"""
Dataclass implementation for "differential calibration" handling.

A differential calibration provides loss values which represent excess loss
between the `SensorCalibration` reference point and another reference point.
A typical usage would be for calibrating out measured cable losses which exist
between the antenna and the Y-factor calibration terminal. At present, this is
measured manually using a calibration probe consisting of a calibrated noise
source and a programmable attenuator.

The ``reference_point`` top-level key defines the point to which measurements
are referenced after using the correction factors included in the file.

The ``calibration_data`` entries are expected to include these correction factors,
with the key name ``"loss"`` and values in decibels (dB). A positive value of
``"loss"`` indicates a LOSS going FROM ``reference_point`` TO the calibration
terminal used by the ``SensorCalibration``.
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
