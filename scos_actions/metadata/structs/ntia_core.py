from typing import List, Optional

import msgspec

from scos_actions.metadata.utils import SIGMF_OBJECT_KWARGS


class HardwareSpec(msgspec.Struct, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-core` `HardwareSpec` objects.

    :param id: Unique ID of hardware, e.g., serial number.
    :param model: Hardware make and model.
    :param version: Hardware version.
    :param description: Description of the hardware.
    :param supplemental_information: Information about hardware, e.g.,
        URL to online datasheets.
    """

    id: str
    model: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    supplemental_information: Optional[str] = None


class Antenna(msgspec.Struct, rename={"antenna_type": "type"}, **SIGMF_OBJECT_KWARGS):
    """
    Interface for generating `ntia-core` `Antenna` objects.

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

    antenna_spec: HardwareSpec
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
