import pytest
from scos_actions.core import signal_analyzer_monitor
from scos_actions.signals import register_signal_analyzer
from scos_actions.hardware.mocks.mock_sigan import MockSignalAnalyzer


def test_sigan_registration():
    sigan = MockSignalAnalyzer()
    register_signal_analyzer.send(__name__, signal_analyzer=sigan)
    assert signal_analyzer_monitor.signal_analyzer == sigan
