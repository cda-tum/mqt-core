# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check that presentation extraction preserves compiler mapping evidence."""

from __future__ import annotations

import runpy
from math import pi
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from mqt.core.mlir import QCProgram, QIRProfile

SCRIPT = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_programs.py"


def test_compact_syndrome_source_preserves_data_and_ancilla_roles() -> None:
    """Register slices must prepare only data and broadcast both parity CXs."""
    source = runpy.run_path(str(SCRIPT))["PARITY_SOURCE"]
    program = QCProgram.from_openqasm_str(source)
    circuit = program.to_qiskit()
    hadamards = [circuit.find_bit(op.qubits[0]).index for op in circuit.data if op.operation.name == "h"]
    assert hadamards == [0, 1]
    loop = next(op.operation.blocks[0] for op in circuit.data if op.operation.name == "for_loop")
    controls = [[loop.find_bit(q).index for q in op.qubits] for op in loop.data if op.operation.name == "cx"]
    assert controls == [[0, 2], [1, 2]]
    qir = program.to_qir(QIRProfile.ADAPTIVE).llvm_ir
    assert "__quantum__qis__reset__body" in qir
    assert "br i1" in qir
    assert "phi" in qir


def test_rus_source_implements_the_literature_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure branches must restore arbitrary data before retrying."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    helpers = runpy.run_path(str(SCRIPT.with_name("capture_compiler_evidence.py")))
    semantics = helpers["rus_semantics"]()
    assert semantics["success_probability"] == pytest.approx(5 / 8)
    assert semantics["max_residual"] < 1e-12
    source = runpy.run_path(str(SCRIPT))["RUS_SOURCE"]
    qir = QCProgram.from_openqasm_str(source).to_qir(QIRProfile.ADAPTIVE).llvm_ir
    assert "br i1" in qir
    assert "__quantum__qis__reset__body" in qir
    monkeypatch.setitem(
        helpers["rus_semantics"].__globals__,
        "RUS_SOURCE",
        source.replace("if (second) { z q[2]; }", "if (second) { x q[2]; }"),
    )
    with pytest.raises(AssertionError):
        helpers["rus_semantics"]()


def test_grover_size_measures_real_fully_unrolled_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read measured files back and catch a preserved loop on the unrolled path."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    helpers = runpy.run_path(str(SCRIPT.with_name("capture_compiler_evidence.py")))
    result = helpers["grover_size"](tmp_path, (4, 8))
    for point in result["points"]:
        for label in ("preserved", "unrolled"):
            source = (tmp_path / point[f"{label}_file"]).read_text(encoding="utf-8")
            assert len(source.encode()) == point[f"{label}_bytes"]
            assert helpers["digest"](source) == point[f"{label}_sha256"]
            assert ("for " in source) == (label == "preserved")
    assert result["points"][-1]["unrolled_bytes"] > result["points"][-1]["preserved_bytes"]


def test_routing_trace_rejects_invented_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace steps and A* parent edges must compose the recorded permutations."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    validate = runpy.run_path(str(SCRIPT.with_name("capture_compiler_evidence.py")))["validate_routing_trace"]
    event = {
        "before": [0, 2, 1],
        "after": [1, 2, 0],
        "swaps": 1,
        "steps": [{"operation": "qco.swap", "sites": [0, 1], "logical": [0, 2]}],
    }
    validate([event], [[0, 1], [1, 2]])
    event["after"] = [0, 2, 1]
    with pytest.raises(AssertionError):
        validate([event], [[0, 1], [1, 2]])


def test_weyl_capture_checks_recorded_unitaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known CZ coordinates anchor the chamber; invalid matrix traces fail."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    helpers = runpy.run_path(str(SCRIPT.with_name("capture_compiler_evidence.py")))
    matrix = [[[float((1 if i < 3 else -1) if i == j else 0), 0.0] for j in range(4)] for i in range(4)]
    block = {"matrix": matrix, "operations": ["qco.cz"], "two_qubit_before": 2, "basis_uses_after": 1}
    target = {"id": "iqm.test", "compilation": {"synthesis_trace": [block], "source_sha256": "test"}}
    point = helpers["weyl"]([target])["points"][0]
    assert point["coordinates"] == pytest.approx([pi / 4, 0, 0])
    matrix[0][0][0] = 2.0
    with pytest.raises(AssertionError):
        helpers["weyl"]([target])


def test_extract_mapping_composes_permutations_and_keeps_branch_context() -> None:
    """Catch swapped mapping directions and invented executed SWAP ordering."""
    assert SCRIPT.is_file(), "The compiler capture entry point must exist"
    extract = runpy.run_path(str(SCRIPT))["extract_layout"]
    ir = """module attributes {mqt.layout = {
      initial = array<i64: 2, 0, 1>, input_count = 2 : i64,
      routing = array<i64: 1, 2, 0>, sites = array<i64: 7, 19, 42>}} {}
    """
    qasm = """OPENQASM 3.1;
if (result[0]) {
  swap $7, $42;
}
swap $19, $7;
"""
    layout = extract(ir, qasm)
    assert layout["initial"] == [42, 7]
    assert layout["final"] == [7, 19]
    assert layout["swaps"] == [[7, 42], [19, 7]]
    assert layout["swap_contexts"] == [
        {"line": 3, "regions": ["if (result[0]) {"]},
        {"line": 5, "regions": []},
    ]
    with pytest.raises(ValueError, match="permutation"):
        extract(ir.replace("routing = array<i64: 1, 2, 0>", "routing = array<i64: 1, 1, 0>"), qasm)


def test_pass_dumps_keep_exact_module_and_reject_missing_capture() -> None:
    """Catch accidental use of the final module as an intermediate stage."""
    assert SCRIPT.is_file(), "The compiler capture entry point must exist"
    extract = runpy.run_path(str(SCRIPT))["extract_pass_ir"]
    dumps = """// -----// IR Dump After MappingPass: place-and-route{ntrials=1} ('builtin.module' operation) //----- //
module {\n  %q = qco.static 7 : !qco.qubit\n}\n
// -----// IR Dump After TargetNativeSynthesis: target-native-synthesis ('builtin.module' operation) //----- //
module {\n  %q = qco.static 19 : !qco.qubit\n}\n
"""
    assert extract(dumps, "place-and-route") == "module {\n  %q = qco.static 7 : !qco.qubit\n}\n"
    with pytest.raises(ValueError, match="missing"):
        extract(dumps, "legalize-control-flow")


def test_qiskit_diagram_size_limit_preserves_small_circuits() -> None:
    """Prevent a successfully exported large circuit from bloating the offline deck."""
    helpers = runpy.run_path(str(SCRIPT))
    assert "render_qiskit" in helpers, "Capture must bound optional circuit drawings"
    render = helpers["render_qiskit"]
    circuit = QuantumCircuit(1)
    circuit.h(0)
    assert "H" in render(circuit)
    for _ in range(1000):
        circuit.x(0)
    with pytest.raises(ValueError, match=r"export succeeded.*text diagram omitted"):
        render(circuit)


def test_circuit_metadata_keeps_control_flow_and_physical_sites() -> None:
    """Draw nested gates on their actual parent wires and preserve site IDs."""
    capture = runpy.run_path(str(SCRIPT))["capture_circuit"]
    circuit = QuantumCircuit(3, 1)
    circuit.h(0)
    circuit.measure(2, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.cx(2, 1)
    diagram = capture(circuit, [7, 19, 42])
    assert [wire["site"] for wire in diagram["qubits"]] == [7, 19, 42]
    branch = diagram["operations"][-1]
    assert branch["name"] == "if_else"
    assert diagram["operations"][1]["clbits"] == [0]
    assert branch["condition_bits"] == [0]
    assert branch["blocks"][0][0]["qubits"] == [2, 1]
    loop = QuantumCircuit(54)
    body = QuantumCircuit(54)
    body.cx(2, 1)
    loop.for_loop(range(2), None, body, loop.qubits, [], label=None)
    diagram = capture(loop, list(range(54)))
    assert [wire["site"] for wire in diagram["qubits"]] == [1, 2]
    assert diagram["operations"][0]["blocks"][0][0]["qubits"] == [1, 0]


def test_parity_example_retains_loop_feedback_and_even_data_parity() -> None:
    """The small compiler story must be real and have a simple semantic check."""
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    helpers = runpy.run_path(str(SCRIPT))
    source = helpers["PARITY_SOURCE"]
    program = QCProgram.from_openqasm_str(source).to_qco()
    program.cleanup()
    assert "scf.for" in program.ir
    assert "qco.if" in program.ir
    circuit = helpers["capture_circuit"](program.to_qiskit())
    loop = next(op for op in circuit["operations"] if op["name"] == "for_loop")
    branch = next(op for op in loop["blocks"][0] if op["name"] == "if_else")
    assert branch["condition_bits"] == [0]
    counts = program.sample(128, 7)
    assert set(counts) == {"000", "110"}


def test_four_qubit_qpe_resolves_non_exact_phase() -> None:
    """Check the two dominant bins against the analytical QPE distribution."""
    import math  # ruff: ignore[import-outside-top-level]

    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    source = runpy.run_path(str(SCRIPT))["qpe_source"]()
    program = QCProgram.from_openqasm_str(source).to_qco()
    counts = program.sample(2048, 7)
    for bin_value in (85, 86):
        difference = 1 / 3 - bin_value / 256
        expected = (math.sin(256 * math.pi * difference) / (256 * math.sin(math.pi * difference))) ** 2
        assert counts[f"{bin_value:08b}"] / 2048 == pytest.approx(expected, abs=0.05)


def test_repeat_until_success_retains_conditional_loop_and_heralded_state() -> None:
    """The retry loop must survive and prepare the non-Clifford V3 state."""
    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level]

    source = runpy.run_path(str(SCRIPT))["RUS_SOURCE"]
    program = QCProgram.from_openqasm_str(source).to_qco()
    program.cleanup()
    assert "scf.while" in program.ir
    counts = program.sample(2048, 7)
    assert set(counts) == {"0001", "1001"}
    assert counts["1001"] / 2048 == pytest.approx(4 / 5, abs=0.04)


@pytest.mark.parametrize(
    ("language", "code", "expected"),
    [
        ("cpp", "// copyright\n#include <array>\n\niterativeQPE(qc::QCProgramBuilder& builder) {\n}\n", 4),
        (
            "mlir",
            (
                "module {\n  func.func @helper() {\n    scf.for %i = %lb to %ub step %s {\n    }\n  }\n"
                "  func.func @main() attributes {mqt.entry_point} {\n    %c0 = arith.constant 0 : index\n"
                "    %q = qco.alloc : !qco.qubit\n    %result = qco.if %condition args(%arg0 = %q) {\n"
                "    }\n  }\n}\n"
            ),
            9,
        ),
    ],
)
def test_stage_focuses_algorithm_without_changing_captured_code(language: str, code: str, expected: int) -> None:
    """Skip copyrights, metadata, and helper functions when opening a capture."""
    artifact = runpy.run_path(str(SCRIPT))["stage"]("source", "Source", language, code)
    assert artifact.get("focus_line") == expected
    assert artifact["code"] == code
