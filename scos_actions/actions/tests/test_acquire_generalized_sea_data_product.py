"""Tests for generalized SEA configuration, capture flow, and metadata output."""

import lzma
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scos_actions.actions import acquire_generalized_sea_data_product as generalized
from scos_actions.hardware.mocks.mock_sensor import MockSensor
from scos_actions.data_product.interfaces import DataProductResult, DataProductGraphInfo


def _parameters(**overrides):
    params = {
        "name": "generalized_test",
        "rf_path": "default",
        "iir_gpass_dB": 1.0,
        "iir_gstop_dB": 40.0,
        "iir_pb_edge_Hz": 4.5e6,
        "iir_sb_edge_Hz": 5.0e6,
        "frequency": 100e6,
        "sample_rate": 1000,
        "duration_ms": 4,
        "td_bin_size_ms": 1,
        "pfp_frame_period_ms": 2,
        "apd_bin_size_dB": 1.0,
        "apd_min_bin_dBm": -100.0,
        "apd_max_bin_dBm": 0.0,
        "nskip": 0,
    }
    params.update(overrides)
    return params


@pytest.fixture
def make_action(monkeypatch):
    # Fixed filter coefficients keep these tests focused on action behavior.
    sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    numerators = np.array([1.0, 0.5])
    denominators = np.array([1.0, -0.5])
    generate_filter = Mock(return_value=sos)
    convert_filter = Mock(return_value=(numerators, denominators))
    monkeypatch.setattr(
        generalized, "generate_elliptic_iir_low_pass_filter", generate_filter
    )
    monkeypatch.setattr(generalized, "sos2tf", convert_filter)

    def build(parameters=None):
        return generalized.GeneralizedSeaDataProduct(parameters or _parameters())

    return build, generate_filter, convert_filter, sos, numerators, denominators


def test_init_sets_defaults_and_preserves_input(make_action):
    """Verify constructor defaults, IIR data, iteration order, and input isolation."""
    build, generate_filter, _, sos, numerators, denominators = make_action
    params = _parameters(frequency=[200e6, 100e6])

    action = build(params)

    assert action.parameters[generalized.FFT_SIZE] == generalized.DEFAULT_FFT_SIZE
    assert action.parameters[generalized.DATA_PRODUCTS] == tuple(
        generalized.ALL_DATA_PRODUCTS
    )
    assert generalized.IIR_GPASS not in action.parameters
    assert [item[generalized.FREQUENCY] for item in action.iteration_params] == [
        200e6,
        100e6,
    ]
    assert all(item[generalized.IIR_SOS] is sos for item in action.iteration_params)
    np.testing.assert_array_equal(
        action.iteration_params[0][generalized.IIR_NUMERATORS], numerators
    )
    np.testing.assert_array_equal(
        action.iteration_params[0][generalized.IIR_DENOMINATORS], denominators
    )
    assert generate_filter.call_count == 1
    assert params[generalized.IIR_GPASS] == 1.0
    assert generalized.FFT_SIZE not in params


def test_init_applies_per_capture_overrides_in_input_order(make_action):
    """Verify grouped captures retain their order and override shared parameters."""
    build, generate_filter, _, _, _, _ = make_action
    params = _parameters(
        captures=[
            {
                "frequency": 200e6,
                "sample_rate": 2000,
                "data_products": ["pvt"],
            },
            {
                "frequency": 100e6,
                "fft_size": 256,
                "duration_ms": 8,
                "data_products": ["psd", "apd"],
            },
        ]
    )

    action = build(params)

    assert [item[generalized.FREQUENCY] for item in action.iteration_params] == [
        200e6,
        100e6,
    ]
    assert [item[generalized.SAMPLE_RATE] for item in action.iteration_params] == [
        2000,
        1000,
    ]
    assert [item[generalized.FFT_SIZE] for item in action.iteration_params] == [
        generalized.DEFAULT_FFT_SIZE,
        256,
    ]
    assert [item[generalized.DATA_PRODUCTS] for item in action.iteration_params] == [
        ("pvt",),
        ("psd", "apd"),
    ]
    assert generate_filter.call_count == 2


def test_update_iir_parameters_sets_sections_and_transfer_coefficients(make_action):
    """Verify filter sections are converted and attached to iteration parameters."""
    build, generate_filter, convert_filter, sos, numerators, denominators = make_action
    action = build()
    generate_filter.reset_mock()
    convert_filter.reset_mock()
    params = {generalized.SAMPLE_RATE: 8000}

    action.update_iir_parameters(params)

    generate_filter.assert_called_once_with(1.0, 40.0, 4.5e6, 5.0e6, 8000)
    convert_filter.assert_called_once_with(sos=sos)
    assert params[generalized.IIR_SOS] is sos
    np.testing.assert_array_equal(params[generalized.IIR_NUMERATORS], numerators)
    np.testing.assert_array_equal(params[generalized.IIR_DENOMINATORS], denominators)


def test_build_constants_exposes_product_configuration(make_action):
    """Verify the action supplies all data_product detector and scaling constants."""
    action = make_action[0]()

    constants = action.build_constants()

    assert constants["FFT_PERCENTILES"] is generalized.FFT_PERCENTILES
    assert constants["FFT_WINDOW_TYPE"] == generalized.FFT_WINDOW_TYPE
    assert constants["FFT_WINDOW_CORRECTION_TYPE"] == "energy"
    assert constants["FFT_M3_DETECTOR"] is generalized.FFT_M3_DETECTOR
    assert constants["TD_DETECTOR"] is generalized.TD_DETECTOR
    assert constants["PFP_M3_DETECTOR"] is generalized.PFP_M3_DETECTOR
    assert constants["PFP_FRAME_RESOLUTION_S"] == generalized.PFP_FRAME_RESOLUTION_S
    assert constants["IMPEDANCE_OHMS"] == 50.0


def test_capture_iq_configures_sigan_and_merges_acquisition_parameters(make_action):
    """Verify IQ sample counts, skip counts, analyzer setup, and returned parameters."""
    action = make_action[0]()
    action.configure_sigan = Mock()
    measurement = {"data": np.array([1.0 + 1.0j]), "overload": False}
    sensor = Mock()
    sensor.acquire_time_domain_samples.return_value = measurement
    action.sensor = sensor
    params = {
        generalized.SAMPLE_RATE: 2000,
        generalized.DURATION_MS: 3,
        generalized.FREQUENCY: 100e6,
        generalized.NUM_SKIP: 5,
    }

    result = action.capture_iq(params)

    action.configure_sigan.assert_called_once_with(params)
    sensor.acquire_time_domain_samples.assert_called_once_with(
        6, 5, cal_params=params
    )
    assert result is measurement
    assert result[generalized.FREQUENCY] == 100e6
    assert result[generalized.SAMPLE_RATE] == 2000
    assert result[generalized.DURATION_MS] == 3


def test_required_components_restarts_api_for_unavailable_or_unhealthy_sigan(
    make_action, monkeypatch
):
    """Verify unavailable and unhealthy analyzers trigger the expected restart."""
    action = make_action[0]()
    send_restart = Mock()
    monkeypatch.setattr(generalized.trigger_api_restart, "send", send_restart)
    unavailable_sigan = SimpleNamespace(is_available=False, healthy=Mock())
    action.sensor = SimpleNamespace(signal_analyzer=unavailable_sigan)

    with pytest.raises(RuntimeError, match="signal analyzer is not available"):
        action.test_required_components()

    unavailable_sigan.healthy.assert_not_called()
    send_restart.assert_called_once_with(sender=action.__class__)

    send_restart.reset_mock()
    unhealthy_sigan = SimpleNamespace(
        is_available=True, healthy=Mock(return_value=False)
    )
    action.sensor = SimpleNamespace(signal_analyzer=unhealthy_sigan)

    assert action.test_required_components() is None
    unhealthy_sigan.healthy.assert_called_once_with()
    send_restart.assert_called_once_with(sender=action.__class__)


def test_create_processing_metadata_records_filter_and_optional_dft(make_action):
    """Verify processing metadata includes the IIR filter and optional DFT."""
    action = make_action[0]()
    builder = Mock()
    params = {
        generalized.IIR_NUMERATORS: np.array([1.0, 0.5]),
        generalized.IIR_DENOMINATORS: np.array([1.0, -0.5]),
    }
    dft = object()

    action.create_processing_metadata(builder, params, dft)

    builder.set_processing.assert_called_once_with(["iir_1"])
    processing_info = builder.set_processing_info.call_args.args[0]
    assert len(processing_info) == 2
    assert processing_info[0].id == "iir_1"
    assert processing_info[0].feedforward_coefficients == [1.0, 0.5]
    assert processing_info[0].feedback_coefficients == [1.0, -0.5]
    assert processing_info[1] is dft

    builder.reset_mock()
    action.create_processing_metadata(builder, params, None)
    processing_info = builder.set_processing_info.call_args.args[0]
    assert len(processing_info) == 1
    assert processing_info[0].id == "iir_1"


def test_create_capture_segment_records_calibration_and_sigan_settings(
    make_action,
):
    """Verify capture metadata records calibrated power and analyzer settings."""
    action = make_action[0]()
    action.sensor = MockSensor()
    action.total_channel_data_length = 128
    builder = Mock()
    measurement = {
        "capture_time": "2026-01-02T03:04:05.000Z",
        generalized.DURATION_MS: 4,
        "overload": True,
        "reference": "signal analyzer input",
        "applied_calibration": {"gain": 1.2345, "noise_figure": 2.3456},
    }

    action.create_capture_segment(builder, 2, measurement)

    capture = builder.add_capture.call_args.args[0]
    assert capture.sample_start == 256
    assert capture.frequency == action.sensor.signal_analyzer.frequency
    assert capture.datetime == measurement["capture_time"]
    assert capture.duration == 4
    assert capture.overload is True
    assert capture.sensor_calibration.gain == 1.234
    assert capture.sensor_calibration.noise_figure == 2.346
    assert capture.sensor_calibration.reference == "signal analyzer input"
    assert capture.sigan_settings.reference_level == -30


def test_create_sigmf_builder_filters_schedule_and_sets_recording(
    make_action, monkeypatch
):
    """Verify SigMF setup filters schedule fields and records capture metadata."""
    action = make_action[0]()
    builder = Mock()
    builder.metadata = {"global": {}}
    monkeypatch.setattr(generalized, "SigMFBuilder", Mock(return_value=builder))
    action.sensor = SimpleNamespace(location={"type": "Point", "coordinates": [1, 2]})

    result = action.create_sigmf_builder(
        task_id=9,
        schedule_entry={
            "name": "entry",
            "priority": 2,
            "ignored": "value",
        },
        sample_rate_Hz=2000,
        num_channels=1,
        recording=3,
    )

    assert result is builder
    schedule = builder.set_schedule.call_args.args[0]
    assert schedule.id == "entry"
    assert schedule.name == "entry"
    assert schedule.priority == 2
    action_metadata = builder.set_action.call_args.args[0]
    assert action_metadata.name == "generalized_test"
    builder.set_geolocation.assert_called_once_with(action.sensor.location)
    builder.set_data_type.assert_called_once_with(
        False, bit_width=16, endianness=""
    )
    builder.set_task.assert_called_once_with(9)
    builder.set_classification.assert_called_once_with("Unclassified")
    builder.set_sample_rate.assert_called_once_with(2000)
    builder.set_num_channels.assert_called_once_with(1)
    builder.set_recording.assert_called_once_with(3)


def test_capture_switch_diagnostic_helpers_map_status_values(make_action):
    """Verify switch values map to UPS, temperature, power, and state diagnostics."""
    action = make_action[0]()
    action.sensor = SimpleNamespace(
        signal_analyzer=SimpleNamespace(temperature=42.5)
    )
    switch_status = {
        "ups_power_state": True,
        "ups_battery_level": True,
        "ups_state": False,
        "ups_battery_state": True,
        "internal_temp": 20.0,
        "tec_intake_temp": 21.0,
        "tec_exhaust_temp": 22.0,
        "internal_humidity": 35.0,
        "power_monitor5V": 5.0,
        "power_monitor15V": 15.0,
        "power_monitor24V": 24.0,
        "power_monitor28V": 28.0,
        "sigan_powered": True,
        "temperature_control_powered": False,
        "preselector_powered": True,
        "heating": True,
        "cooling": False,
    }
    switch_diag = {}

    action.set_ups_states(switch_status, switch_diag)
    action.add_temperature_and_humidity_sensors(switch_status, switch_diag)
    action.add_power_sensors(switch_status, switch_diag)
    action.add_power_states(switch_status, switch_diag)
    action.add_heating_cooling(switch_status, switch_diag)

    assert switch_diag["battery_backup"] is False
    assert switch_diag["low_battery"] is False
    assert switch_diag["ups_healthy"] is True
    assert switch_diag["replace_battery"] is False
    assert switch_diag["temperature_sensors"] == [
        {"name": "internal_temp", "value": 20.0},
        {"name": "sigan_internal_temp", "value": 42.5},
        {"name": "tec_intake_temp", "value": 21.0},
        {"name": "tec_exhaust_temp", "value": 22.0},
    ]
    assert switch_diag["humidity_sensors"] == [
        {"name": "internal_humidity", "value": 35.0}
    ]
    assert [sensor["expected_value"] for sensor in switch_diag["power_sensors"]] == [
        5.0,
        15.0,
        24.0,
        28.0,
    ]
    assert switch_diag["sigan_powered"] is True
    assert switch_diag["temperature_control_powered"] is False
    assert switch_diag["preselector_powered"] is True
    assert switch_diag["heating"] is True
    assert switch_diag["cooling"] is False


def test_create_global_sensor_metadata_uses_sensor_definition(make_action):
    """Verify the sensor definition and hash are placed in global metadata."""
    action = make_action[0]()
    builder = Mock()
    sensor = SimpleNamespace(
        capabilities={
            "sensor": {
                "sensor_spec": {"id": "test-sensor"},
                "sensor_sha512": "sensor-hash",
            }
        }
    )

    action.create_global_sensor_metadata(builder, sensor)

    sensor_metadata = builder.set_sensor.call_args.args[0]
    assert sensor_metadata.sensor_spec.id == "test-sensor"
    assert sensor_metadata.sensor_sha512 == "sensor-hash"


def test_capture_diagnostics_collects_sensor_and_computer_values(
    make_action, monkeypatch
):
    """Verify diagnostics aggregate preselector, switch, CPU, and software values."""
    action = make_action[0]()
    switch_status = {
        "name": "spu",
        "healthy": True,
        "ups_power_state": True,
        "ups_battery_level": True,
        "ups_state": False,
        "ups_battery_state": True,
        "internal_temp": 20.0,
        "tec_intake_temp": 21.0,
        "tec_exhaust_temp": 22.0,
        "internal_humidity": 35.0,
        "power_monitor5V": 5.0,
        "power_monitor15V": 15.0,
        "power_monitor24V": 24.0,
        "power_monitor28V": 28.0,
        "sigan_powered": True,
        "temperature_control_powered": False,
        "preselector_powered": True,
        "door_state": True,
    }
    sensor = SimpleNamespace(
        switches={"spu": SimpleNamespace(get_status=lambda: switch_status.copy())},
        preselector=SimpleNamespace(
            get_status=lambda: {
                "name": "preselector",
                "healthy": True,
                "lna_temperature": 35.0,
            }
        ),
        signal_analyzer=SimpleNamespace(
            temperature=40.0,
            plugin_version="plugin-version",
            firmware_version="firmware-version",
            api_version="api-version",
        ),
        start_time=datetime(2026, 1, 1),
    )
    builder = Mock()
    schema = lambda **fields: fields
    for model_name in [
        "Preselector",
        "SPU",
        "Computer",
        "Software",
        "Diagnostics",
        "ScosPlugin",
        "SsdSmartData",
    ]:
        monkeypatch.setattr(generalized.ntia_diagnostics, model_name, schema)

    monkeypatch.setattr(generalized, "perf_counter", Mock(side_effect=[10, 12, 13]))
    monkeypatch.setattr(generalized, "get_cpu_uptime_seconds", lambda: 86400)
    monkeypatch.setattr(generalized.psutil, "cpu_percent", lambda **_kwargs: 12.34)
    monkeypatch.setattr(generalized.psutil, "getloadavg", lambda: (1.0, 2.0, 3.0))
    monkeypatch.setattr(generalized.psutil, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        generalized.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=55.5),
    )
    monkeypatch.setattr(generalized, "get_current_cpu_temperature", lambda: 60.0)
    monkeypatch.setattr(generalized, "get_max_cpu_temperature", lambda: 80.0)
    monkeypatch.setattr(
        generalized,
        "convert_datetime_to_millisecond_iso_format",
        lambda _value: "start",
    )
    monkeypatch.setattr(generalized, "get_days_up", lambda _value: 2.0)
    monkeypatch.setattr(
        generalized, "get_disk_smart_data", lambda _device: {"health": "ok"}
    )
    monkeypatch.setattr(generalized, "get_disk_usage", lambda: 33.3)
    monkeypatch.setattr(generalized.utils, "get_datetime_str_now", lambda: "now")

    action.sensor = sensor
    action.capture_diagnostics(builder, sensor, 10.5, [900.0, 1100.0])

    diagnostics = builder.set_diagnostics.call_args.args[0]
    assert diagnostics["datetime"] == "now"
    assert diagnostics["preselector"] == {"lna_temperature": 35.0}
    assert diagnostics["spu"]["door_closed"] is False
    assert diagnostics["computer"]["cpu_max_clock"] == 1100.0
    assert diagnostics["computer"]["cpu_min_clock"] == 900.0
    assert diagnostics["computer"]["cpu_mean_clock"] == 1000.0
    assert diagnostics["computer"]["action_runtime"] == 1.5
    assert diagnostics["computer"]["action_cpu_usage"] == 12.3
    assert diagnostics["computer"]["system_load_5m"] == 100.0
    assert diagnostics["computer"]["memory_usage"] == 55.5
    assert diagnostics["computer"]["cpu_temp"] == 60.0
    assert diagnostics["computer"]["cpu_overheating"] is False
    assert diagnostics["computer"]["ssd_smart_data"] == {"health": "ok"}
    assert diagnostics["computer"]["disk_usage"] == 33.3
    assert diagnostics["software"]["scos_sigan_plugin"] == {
        "name": "scos_tekrsa",
        "version": "plugin-version",
    }


def test_call_processes_product_results_and_emits_compressed_capture(
    make_action, monkeypatch
):
    """Verify data_product results become compressed bytes and a completion event."""
    params = _parameters(data_products=["psd", "pvt"])
    action = make_action[0](params)
    sensor = SimpleNamespace(preselector=object())
    builder = Mock()
    builder.metadata = {"global": {"test": True}}
    action.test_required_components = Mock()
    action.configure_preselector = Mock()
    action.capture_iq = Mock(
        return_value={
            "data": np.array([1.0 + 1.0j]),
            "reference": "signal analyzer input",
        }
    )
    action.create_sigmf_builder = Mock(return_value=builder)
    action.create_capture_segment = Mock()
    action.create_global_sensor_metadata = Mock()
    action.create_processing_metadata = Mock()
    action.capture_diagnostics = Mock()
    monkeypatch.setattr(generalized.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(generalized.ray, "wait", Mock(return_value=([], [])))
    kill = Mock()
    monkeypatch.setattr(generalized.ray, "kill", kill)
    monkeypatch.setattr(generalized.psutil, "cpu_percent", lambda **_kwargs: 0.0)
    monkeypatch.setattr(generalized, "get_current_cpu_clock_speed", lambda: 1000.0)

    class RemoteRun:
        def __init__(self):
            self.options_args = None
            self.remote_args = None

        def options(self, **kwargs):
            self.options_args = kwargs
            return self

        def remote(self, *args):
            self.remote_args = args
            return "outer-ref"

    remote_run = RemoteRun()
    supervisor = SimpleNamespace(run=remote_run)
    monkeypatch.setattr(generalized, "NUM_ACTORS", 1)
    monkeypatch.setattr(
        generalized,
        "IQProcessorProductActors",
        SimpleNamespace(remote=Mock(return_value=supervisor)),
    )

    psd_graph = SimpleNamespace(reference=None)
    pvt_graph = SimpleNamespace(reference=None)
    dft = object()
    product_results = {
        "psd-ref": DataProductResult(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            None,
            DataProductGraphInfo(2, 4, psd_graph, dft),
        ),
        "pvt-ref": DataProductResult(
            np.array([[5.0, 6.0]]),
            np.array([10.0, 20.0, 30.0, 40.0]),
            DataProductGraphInfo(2, 2, pvt_graph, None),
        ),
    }
    get = Mock(
        side_effect=lambda ref: (
            ["psd-ref", "pvt-ref"] if ref == "outer-ref" else product_results[ref]
        )
    )
    monkeypatch.setattr(generalized.ray, "get", get)
    send_complete = Mock()
    monkeypatch.setattr(
        generalized.measurement_action_completed, "send", send_complete
    )

    action(sensor, {"name": "schedule"}, task_id=42)

    assert remote_run.options_args == {"num_returns": 2}
    assert remote_run.remote_args[0].tolist() == [1.0 + 1.0j]
    assert remote_run.remote_args[1] is action.iteration_params[0]
    assert remote_run.remote_args[2] is action.iteration_params[0][generalized.IIR_SOS]
    assert action.create_processing_metadata.call_args.args == (
        builder,
        action.iteration_params[0],
        dft,
    )
    assert builder.set_data_products.call_args.args[0] == [psd_graph, pvt_graph]
    assert psd_graph.reference == "signal analyzer input"
    assert pvt_graph.reference == "signal analyzer input"
    builder.set_max_of_max_channel_powers.assert_called_once()
    builder.set_median_of_mean_channel_powers.assert_called_once()
    builder.set_mean_channel_powers.assert_called_once()
    builder.set_median_channel_powers.assert_called_once()
    diagnostics_args = action.capture_diagnostics.call_args.args
    assert diagnostics_args[:2] == (builder, sensor)
    assert isinstance(diagnostics_args[2], float)
    assert diagnostics_args[3] == [1000.0]
    send_args = send_complete.call_args.kwargs
    assert send_args["task_id"] == 42
    assert send_args["metadata"] == builder.metadata
    expected_data = np.array([1, 2, 3, 4, 5, 6], dtype=np.half).tobytes()
    assert lzma.decompress(send_args["data"]) == expected_data
    kill.assert_called_once_with(supervisor)


def test_result_conversion_and_action_descriptors(make_action):
    """Verify data flattening, compression, and action descriptor properties."""
    action = make_action[0]()
    data = action.transform_data(
        [np.array([1.25, 2.5]), np.array([3.75])]
    )

    assert data.dtype == np.half
    np.testing.assert_array_equal(data, np.array([1.25, 2.5, 3.75], dtype=np.half))
    assert lzma.decompress(action.compress_bytes_data(b"capture")) == b"capture"
    assert action.is_complex() is False
    assert action.description
