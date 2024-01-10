from scos_actions.core import status_registrar
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer
from scos_actions.signals import register_component_with_status


def test_status_handler():
    mock_sigan = MockSignalAnalyzer()
    register_component_with_status.send(__name__, component=mock_sigan)
    status_registrar.status_components[0] == mock_sigan
