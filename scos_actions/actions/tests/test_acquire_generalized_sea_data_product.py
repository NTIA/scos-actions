from scos_actions.actions.acquire_generalized_sea_data_product import (
    DEFAULT_INLINE_CONFIG_PATH,
    GeneralizedSeaDataProduct,
    build_inline_iteration_parameters,
    load_inline_action_parameters,
    load_inline_iteration_parameters,
)


def test_load_inline_action_parameters():
    parameters = load_inline_action_parameters()

    assert parameters["name"] == "test_SEA_CBRS_Measure_Baseline"
    assert len(parameters["captures"]) == 15
    assert parameters["captures"][0]["frequency"] == 3555e6


def test_build_inline_iteration_parameters():
    parameters = load_inline_action_parameters()

    iteration_parameters = build_inline_iteration_parameters(parameters)

    assert len(iteration_parameters) == 15
    assert iteration_parameters[0]["frequency"] == 3555e6
    assert iteration_parameters[0]["sample_rate"] == 14e6
    assert iteration_parameters[0]["duration_ms"] == 4000
    assert iteration_parameters[0]["products"] == ["pfp", "psd", "afp", "tsp"]

    assert iteration_parameters[1]["frequency"] == 3565e6
    assert iteration_parameters[1]["sample_rate"] == 15e6
    assert iteration_parameters[1]["duration_ms"] == 2000
    assert iteration_parameters[1]["products"] == ["psd", "afp"]

    assert iteration_parameters[7]["frequency"] == 3625e6
    assert iteration_parameters[7]["sample_rate"] == 28e6
    assert iteration_parameters[7]["duration_ms"] == 4000
    assert iteration_parameters[7]["products"] == ["pfp", "afp"]

    assert iteration_parameters[-1]["frequency"] == 3695e6
    assert iteration_parameters[-1]["sample_rate"] == 14e6


def test_load_inline_iteration_parameters():
    iteration_parameters = load_inline_iteration_parameters()

    assert len(iteration_parameters) == 15
    assert all("frequency" in capture for capture in iteration_parameters)


def test_inline_action_uses_default_config():
    action = GeneralizedSeaDataProduct.from_default_config()

    assert action.name == "test_SEA_CBRS_Measure_Baseline"
    assert action.config_path == DEFAULT_INLINE_CONFIG_PATH
    assert len(action.iteration_params) == 15
    assert action.uses_uniform_runtime_parameters() is False
