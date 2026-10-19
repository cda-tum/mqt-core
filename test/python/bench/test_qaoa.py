# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the QAOA MaxCut benchmark."""

from __future__ import annotations

import json
import math

import pytest

from mqt.core.bench import qaoa

from .utils import assert_generates


def test_qaoa_reference_and_json() -> None:
    """Expose the default angles, cut evaluation, and verification manifest."""
    benchmark = qaoa.QAOA(qaoa.Options(qubits=4))
    assert benchmark.options.qubits == 4
    assert benchmark.options.gammas == pytest.approx([math.pi / 4])
    assert benchmark.options.betas == pytest.approx([math.pi / 8])
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 4
    assert benchmark.cut_size("0101") == 4
    evaluation = benchmark.evaluate({"0101": 3, "0000": 1})
    assert evaluation.mean_cut == pytest.approx(3)
    assert evaluation.approximation_ratio == pytest.approx(0.75)
    assert json.loads(benchmark.manifest_json)["reference"]["kind"] == "verification"
    for restored in (
        qaoa.QAOA.from_instance_specification_json(benchmark.instance_specification_json),
        qaoa.QAOA.from_manifest_json(benchmark.manifest_json),
    ):
        assert restored.case_id == benchmark.case_id


def test_qaoa_generation() -> None:
    """Pass multiple angle pairs through the Python generation boundary."""
    benchmark = qaoa.QAOA(qaoa.Options(qubits=5, gammas=[0.2, 0.7], betas=[0.3, 0.4]))
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {
        "qubits": 5,
        "gammas": [0.2, 0.7],
        "betas": [0.3, 0.4],
    }
    assert_generates(benchmark.generate())


def test_qaoa_errors() -> None:
    """Propagate invalid parameters and outcomes through the binding."""
    with pytest.raises(ValueError, match=r"same.*length"):
        qaoa.QAOA(qaoa.Options(qubits=4, gammas=[0.2, 0.7], betas=[0.3]))
    with pytest.raises(ValueError, match="width"):
        qaoa.QAOA(qaoa.Options(qubits=4)).cut_size("010")
