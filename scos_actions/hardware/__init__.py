from scos_actions.hardware.mocks.mock_gps import MockGPS
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer

radio = MockSignalAnalyzer(randomize_values=True)
gps = MockGPS()
