from scos_actions.hardware import load_preselector


def test_load_preselector():
    preselector = load_preselector({"name": "test", "base_url": "http://127.0.0.1"})
    assert preselector is not None
