from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scos_actions.product import distributed
from scos_actions.product.distributed import actors, iq_processor
from scos_actions.product.interfaces import DataProductResult, ProductGraphInfo


def test_distributed_package_exports_public_actors():
    """Ensure the distributed package exposes both actor types."""
    assert distributed.ProductActor is actors.ProductActor
    assert distributed.IQProcessorProductActors is iq_processor.IQProcessorProductActors


def test_product_actor_constructs_and_runs_product(monkeypatch):
    """Ensure a product actor constructs its product once and delegates compute."""
    expected_result = DataProductResult(
        np.array([1.0]), None, ProductGraphInfo(1, 1, object(), None)
    )
    product = Mock()
    product.compute.return_value = expected_result
    build_product = Mock(return_value=product)
    monkeypatch.setitem(actors._product_builders, "apd", build_product)
    actor_class = actors.ProductActor.__ray_metadata__.modified_class

    actor = actor_class("apd", {"IMPEDANCE_OHMS": 50.0})
    iq = np.array([1.0 + 1.0j])
    params = {"recording_id": 1}

    result = actor.run(iq, params)

    build_product.assert_called_once_with({"IMPEDANCE_OHMS": 50.0})
    product.compute.assert_called_once_with(iq, params)
    assert result is expected_result


def test_product_actor_rejects_unknown_product():
    """Ensure unsupported product names fail explicitly."""
    actor_class = actors.ProductActor.__ray_metadata__.modified_class

    with pytest.raises(KeyError, match="unknown"):
        actor_class("unknown", {})


def test_iq_processor_builds_workers_once_and_preserves_order(monkeypatch):
    """Ensure worker reuse, filtering, and configured product order are preserved."""
    created_workers = []
    worker_calls = []
    filtered_iq = np.array([2.0 + 1.0j, 3.0 + 1.0j])

    class FakeWorker:
        def __init__(self, name):
            self.name = name
            self.run = SimpleNamespace(remote=self._run_remote)
            self.__ray_terminate__ = SimpleNamespace(remote=self._terminate_remote)

        def _run_remote(self, iq_ref, params):
            worker_calls.append((self.name, iq_ref, params))
            return f"ref-{self.name}"

        def _terminate_remote(self):
            return f"terminated-{self.name}"

    def create_worker(name, consts):
        created_workers.append((name, consts))
        return FakeWorker(name)

    monkeypatch.setattr(
        iq_processor,
        "ProductActor",
        SimpleNamespace(remote=create_worker),
    )
    monkeypatch.setattr(iq_processor, "sosfilt", lambda *, sos, x: filtered_iq)
    monkeypatch.setattr(iq_processor.ray, "put", lambda value: ("object-ref", value))
    processor_class = (
        iq_processor.IQProcessorProductActors.__ray_metadata__.modified_class
    )
    processor = processor_class(actor_id=5)
    iq = np.array([1.0 + 1.0j, 2.0 + 1.0j])
    params = {"recording_id": 8, "data_products": ["pfp", "psd"]}
    consts = {"shared": True}
    sos = np.array([[1.0, 0.0, 0.0]])

    first_result = processor.run(iq, params, sos, consts)
    second_result = processor.run(iq, params, sos, consts)

    assert first_result == ("ref-pfp", "ref-psd")
    assert second_result == first_result
    assert created_workers == [("pfp", consts), ("psd", consts)]
    assert len(worker_calls) == 4
    assert worker_calls[0][1] == ("object-ref", filtered_iq)
    assert worker_calls[0][2] is params


def test_iq_processor_shutdown_terminates_workers_and_clears_state(monkeypatch):
    """Ensure shutdown waits for worker termination and clears actor state."""
    processor_class = (
        iq_processor.IQProcessorProductActors.__ray_metadata__.modified_class
    )
    processor = processor_class(actor_id=5)
    terminated = []
    worker = SimpleNamespace(
        __ray_terminate__=SimpleNamespace(
            remote=lambda: terminated.append("worker-ref") or "worker-ref"
        )
    )
    processor.workers = {"apd": worker}
    wait = Mock(return_value=([], ["worker-ref"]))
    monkeypatch.setattr(iq_processor.ray, "wait", wait)

    processor.shutdown()

    assert terminated == ["worker-ref"]
    wait.assert_called_once_with(["worker-ref"], num_returns=1)
    assert processor.workers == {}


def test_iq_processor_shutdown_without_workers_skips_wait(monkeypatch):
    """Ensure shutdown without workers does not call Ray wait."""
    processor_class = (
        iq_processor.IQProcessorProductActors.__ray_metadata__.modified_class
    )
    processor = processor_class(actor_id=5)
    wait = Mock()
    monkeypatch.setattr(iq_processor.ray, "wait", wait)

    processor.shutdown()

    wait.assert_not_called()
    assert processor.workers == {}
