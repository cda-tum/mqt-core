# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared serialization contracts of the benchmark Python bindings."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mqt.core.bench import (
    bv,
    fan_out,
    ghz,
    grover,
    grover_weak_measurement,
    magic_state_distillation,
    modular_multiplier,
    multiplexer,
    qaoa,
    qft,
    qft_adder,
    qpe,
    repeat_until_success,
    shor,
    teleportation,
    w_state,
)

Benchmark = (
    bv.BV
    | fan_out.FanOut
    | ghz.GHZ
    | grover.Grover
    | grover_weak_measurement.Grover
    | magic_state_distillation.MagicStateDistillation
    | modular_multiplier.ModularMultiplier
    | multiplexer.Multiplexer
    | qaoa.QAOA
    | qft.QFT
    | qft_adder.QFTAdder
    | qpe.QPE
    | repeat_until_success.RepeatUntilSuccess
    | shor.Shor
    | teleportation.Teleportation
    | w_state.WState
)


@pytest.mark.parametrize(
    "benchmark",
    [
        bv.BV(bv.Options(hidden_bitstring="101")),
        fan_out.FanOut(fan_out.Options(controls=3)),
        ghz.GHZ(ghz.Options(qubits=3)),
        grover.Grover(grover.Options(marked_bitstring="10")),
        grover_weak_measurement.Grover(grover_weak_measurement.Options(marked_bitstring="10")),
        magic_state_distillation.MagicStateDistillation(),
        modular_multiplier.ModularMultiplier(
            modular_multiplier.Options(multiplier="011", modulus="101", multiplicand="111")
        ),
        multiplexer.Multiplexer(multiplexer.Options(qubits=3)),
        qaoa.QAOA(qaoa.Options(qubits=4)),
        qft.QFT(qft.Options(qubits=3, period_exponent=1)),
        qft_adder.QFTAdder(qft_adder.Options(addend="110", accumulator="011")),
        qpe.QPE(qpe.Options(precision=3, phase=Fraction(1, 8))),
        repeat_until_success.RepeatUntilSuccess(),
        shor.Shor(shor.Options(number=21)),
        teleportation.Teleportation(),
        w_state.WState(w_state.Options(qubits=3)),
    ],
    ids=lambda benchmark: type(benchmark).__module__,
)
def test_serialization(benchmark: Benchmark) -> None:
    """Preserve the family and canonical metadata through both JSON parsers."""
    benchmark_type = type(benchmark)
    for parse, document in (
        (benchmark_type.from_instance_specification_json, benchmark.instance_specification_json),
        (benchmark_type.from_manifest_json, benchmark.manifest_json),
    ):
        restored = parse(document)
        assert type(restored) is benchmark_type
        assert restored.instance_specification_json == benchmark.instance_specification_json
        assert restored.manifest_json == benchmark.manifest_json
        assert restored.case_id == benchmark.case_id
        with pytest.raises(ValueError, match=r"^invalid\.json:\$/schema_version"):
            parse("{}", source="invalid.json")
