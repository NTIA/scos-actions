"""Tests for the public data-product result and interface types."""

import numpy as np
import pytest

from scos_actions.data_product.interfaces import (
    DataProductInterface,
    DataProductResult,
    DataProductGraphInfo,
)


def test_result_types_are_named_and_positional_tuples():
    """Ensure result objects expose named fields and retain tuple unpacking."""
    graph = object()
    graph_info = DataProductGraphInfo(3, 6, graph, None)
    data = np.array([1.0, 2.0, 3.0])
    summary = np.array([2.0])
    result = DataProductResult(data, summary, graph_info)

    assert result.data_result is data
    assert result.summary is summary
    assert result.graph_info is graph_info
    assert tuple(result) == (data, summary, graph_info)
    assert tuple(graph_info) == (3, 6, graph, None)


def test_data_product_interface_is_abstract():
    """Ensure callers cannot instantiate the unimplemented data_product interface."""
    with pytest.raises(TypeError):
        DataProductInterface({})
