from dataclasses import dataclass
from typing import List, Optional

from scos_actions.metadata.interfaces.sigmf_object import SigMFObject


@dataclass
class HardwareSpec(SigMFObject):
    """
    Interface for generating `ntia-core` `HardwareSpec` objects.

    The `id` parameter is required.

    :param id: Unique ID of hardware, e.g., serial number.
    :param model: Hardware make and model.
    :param version: Hardware version.
    :param description: Description of the hardware.
    :param supplemental_information: Information about hardware, e.g.,
        URL to online datasheets.
    """

    id: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    supplemental_information: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.id, "id")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "id": "id",
                "model": "model",
                "version": "version",
                "description": "description",
                "supplemental_information": "supplemental_information",
            }
        )
        # Create metadata object
        super().create_json_object()


@dataclass
class Antenna(SigMFObject):
    """
    Interface for generating `ntia-core` `Antenna` objects.

    The `antenna_spec` parameter is required.

    :param antenna_spec: Metadata to describe the antenna.
    :param antenna_type: Antenna type, e.g. `"dipole"`, `"biconical"`, `"monopole"`,
        `"conical monopole"`
    :param frequency_low: Low frequency of operational range, in Hz.
    :param frequency_high: High frequency of operational range, in Hz.
    :param polarization: Antenna polarization, e.g., `"vertical"`, `"horizontal"`,
        `"slant-45"`, `"left-hand circular"`, `"right-hand circular"`
    :param cross_polar_discrimination: Cross-polar discrimination.
    :param gain: Antenna gain in direction of maximum radiation or reception, in dBi.
    :param horizontal_gain_pattern: Antenna gain pattern in horizontal plane from 0
        to 359 degrees in 1 degree steps, in dBi.
    :param vertical_gain_pattern: Antenna gain pattern in vertical plane from -90 to
        +90 degrees in 1 degree steps, in dBi.
    :param horizontal_beamwidth: Horizontal 3 dB beamwidth, in degrees.
    :param vertical_beamwidth: Vertical 3 dB beamwidth, in degrees.
    :param voltage_standing_wave_ratio: Voltage standing wave ratio.
    :param cable_loss: Cable loss for cable connecting antenna and preselector, in dB.
    :param steerable: Defines whether the antenna is steerable.
    :param azimuth_angle: Angle of main beam in azimuthal plane from North, in degrees.
    :param elevation_angle: Angle of main beam in elevation plane from horizontal, in degrees.
    """

    antenna_spec: Optional[HardwareSpec] = None
    antenna_type: Optional[str] = None
    frequency_low: Optional[float] = None
    frequency_high: Optional[float] = None
    polarization: Optional[float] = None
    cross_polar_discrimination: Optional[float] = None
    gain: Optional[float] = None
    horizontal_gain_pattern: Optional[List[float]] = None
    vertical_gain_pattern: Optional[List[float]] = None
    horizontal_beamwidth: Optional[float] = None
    vertical_beamwidth: Optional[float] = None
    voltage_standing_wave_ratio: Optional[float] = None
    cable_loss: Optional[float] = None
    steerable: Optional[bool] = None
    azimuth_angle: Optional[float] = None
    elevation_angle: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        # Ensure required keys have been set
        self.check_required(self.antenna_spec, "antenna_spec")
        # Define SigMF key names
        self.obj_keys.update(
            {
                "antenna_spec": "antenna_spec",
                "antenna_type": "type",
                "frequency_low": "frequency_low",
                "frequency_high": "frequency_high",
                "polarization": "polarization",
                "cross_polar_discrimination": "cross_polar_discrimination",
                "gain": "gain",
                "horizontal_gain_pattern": "horizontal_gain_pattern",
                "vertical_gain_pattern": "vertical_gain_pattern",
                "horizontal_beamwidth": "horizontal_beamwidth",
                "vertical_beamwidth": "vertical_beamwidth",
                "voltage_standing_wave_ratio": "voltage_standing_wave_ratio",
                "cable_loss": "cable_loss",
                "steerable": "steerable",
                "azimuth_angle": "azimuth_angle",
                "elevation_angle": "elevation_angle",
            }
        )
        # Create metadata object
        super().create_json_object()
