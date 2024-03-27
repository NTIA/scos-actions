import pytest

from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer


def test_mock_sigan():
    sigan = MockSignalAnalyzer()
    # Test default values are available as properties
    assert sigan.model == sigan._model
    assert sigan.frequency == sigan._frequency
    assert sigan.sample_rate == sigan._sample_rate
    assert sigan.gain == sigan._gain
    assert sigan.attenuation == sigan._attenuation
    assert sigan.preamp_enable == sigan._preamp_enable
    assert sigan.reference_level == sigan._reference_level
    assert sigan.is_available == sigan._is_available
    assert sigan.plugin_version == sigan._plugin_version
    assert sigan.firmware_version == sigan._firmware_version
    assert sigan.api_version == sigan._api_version
