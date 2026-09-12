from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from brl.bilateral import _ordered_by_travel
from brl.local_env import RadioEnv, Source
from brl.protocol import quantize_bearing_deg


def test_symmetric_bilateral_probe_uses_declared_order():
    points = [np.array([750.0, 18.0]), np.array([750.0, -18.0])]
    ordered = _ordered_by_travel(points, np.zeros(2))
    assert np.array_equal(ordered[0], points[0])


def test_official_bearing_rounding_is_two_decimals():
    assert quantize_bearing_deg(359.999) == 0.0
    assert quantize_bearing_deg(12.3451) == 12.35


def test_local_environment_can_enable_protocol_rounding():
    source = Source(1, np.array([100.0, 100.0]), 1500.0)
    env = RadioEnv(mode=3, n_sources=10, seed=1, bearing_decimals=2)
    env.reset(seed=1, n_sources=1, sources=[source])
    result = env.measure([0.0, 0.0], 1)
    assert result["measure_result"] == "direction"
    assert result["svd_deg"] == round(result["svd_deg"], 2)


def test_isr_adapter_does_not_read_local_truth():
    source = Path(__file__).parents[1] / "brl" / "independent_candidate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {"sources", "source_by_channel", "error_field", "rng", "seed"}
    reads = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (reads & forbidden)
