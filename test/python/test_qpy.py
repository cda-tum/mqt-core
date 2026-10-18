# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QPY interoperability through real Qiskit circuits and Core programs."""

from __future__ import annotations

import subprocess
import sys
from io import BytesIO, StringIO
from typing import TYPE_CHECKING

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit
from qiskit import qpy as qiskit_qpy
from qiskit.circuit import IfElseOp, ParameterVector
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit_support import supports_qiskit_translation

from mqt.core import qpy
from mqt.core.mlir import CompilerTarget, QCProgram

if TYPE_CHECKING:
    from pathlib import Path

requires_translation = pytest.mark.skipif(
    not supports_qiskit_translation(), reason=f"No registered Qiskit adapter for {qiskit.__version__}"
)


@requires_translation
@pytest.mark.parametrize("dialect", ["qc", "qco"])
@pytest.mark.parametrize("version", [17, None])
def test_qpy_paths_preserve_parameters_phase_and_sources(tmp_path: Path, dialect: str, version: int | None) -> None:
    """Qiskit can read exported programs and bind the original parameter identities."""
    angles = ParameterVector("angles", 2)
    circuit = QuantumCircuit(2, global_phase=0.25)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(angles[0] + 2 * angles[1], 1)
    input_path = tmp_path / "input.qpy"
    with input_path.open("wb") as stream:
        qiskit_qpy.dump(circuit, stream)
    program = qpy.load(str(input_path))[0]
    if dialect == "qco":
        program = program.to_qco()
    original_ir = program.ir
    output_path = tmp_path / "output.qpy"
    qpy.dump(program, output_path, version=version)
    assert program.is_valid
    assert program.ir == original_ir
    with output_path.open("rb") as stream:
        assert qiskit_qpy.get_qpy_version(stream) == (qiskit_qpy.QPY_VERSION if version is None else version)
        stream.seek(0)
        restored = qiskit_qpy.load(stream)[0]
    bindings = {angles[0]: 0.125, angles[1]: -0.75}
    assert set(restored.parameters) == set(circuit.parameters)
    np.testing.assert_allclose(
        Operator(restored.assign_parameters(bindings)).data,
        Operator(circuit.assign_parameters(bindings)).data,
        atol=1e-12,
    )


@requires_translation
def test_qpy_streams_preserve_order_control_flow_and_ownership() -> None:
    """Streams use their current position and remain open, preserving circuit order and control flow."""
    first = QuantumCircuit(1)
    first.h(0)
    dynamic = QuantumCircuit(2, 1)
    dynamic.measure(0, 0)
    with dynamic.if_test((dynamic.clbits[0], 1)):
        dynamic.x(1)
    source = BytesIO()
    qiskit_qpy.dump([first, dynamic], source)
    prefixed = BytesIO(b"prefix" + source.getvalue())
    prefixed.seek(6)
    programs = qpy.load(prefixed)
    assert not prefixed.closed
    assert len(programs) == 2
    output = BytesIO(b"prefix")
    output.seek(6)
    qpy.dump(iter(programs), output)
    assert not output.closed
    assert output.getvalue().startswith(b"prefixQISKIT")
    restored = qiskit_qpy.load(BytesIO(output.getvalue()[6:]))
    np.testing.assert_allclose(Operator(restored[0]).data, Operator(first).data, atol=1e-12)
    assert restored[1].count_ops() == {"measure": 1, "if_else": 1}
    branch = restored[1].data[-1].operation
    assert isinstance(branch, IfElseOp)
    assert branch.blocks[0].count_ops() == {"x": 1}
    assert all(program.is_valid for program in programs)


def test_empty_archive() -> None:
    """Empty program collections produce empty QPY archives."""
    stream = BytesIO()
    qpy.dump([], stream)
    stream.seek(0)
    assert qpy.load(stream) == []


def test_unsupported_qpy_version() -> None:
    """QPY decoding errors propagate from Qiskit."""
    with pytest.raises(QiskitError, match="QPY format version"):
        qpy.load(BytesIO(b"QISKIT\xff"))


def test_text_stream_is_rejected_as_caller_error() -> None:
    """The input stream must contain binary data."""
    with pytest.raises(TypeError):
        qpy.load(StringIO("not binary"))  # ty: ignore[invalid-argument-type] Invalid public input.


@requires_translation
def test_unsupported_circuit_in_archive_fails() -> None:
    """A valid archive with a later unsupported circuit must not return a partial list."""
    supported = QuantumCircuit(1)
    supported.h(0)
    unsupported = QuantumCircuit(1)
    unsupported.delay(10, 0)
    stream = BytesIO()
    qiskit_qpy.dump([supported, unsupported], stream)
    stream.seek(0)
    with pytest.raises(RuntimeError, match=r"[Dd]elay|unsupported"):
        qpy.load(stream)
    assert not stream.closed


@requires_translation
def test_qpy_export_maps_target_sites() -> None:
    """Target-aware export maps sparse physical site IDs into dense Qiskit indices."""
    target = CompilerTarget(
        "sparse target",
        [CompilerTarget.Site(10), CompilerTarget.Site(20)],
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    program = QCProgram.from_mlir_str("""module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.static 20 : !qc.qubit
    qc.x %q : !qc.qubit
    return
  }
}""")
    stream = BytesIO()
    qpy.dump(program, stream, target=target)
    stream.seek(0)
    restored = qiskit_qpy.load(stream)[0]
    expected = QuantumCircuit(2)
    expected.x(1)
    np.testing.assert_allclose(Operator(restored).data, Operator(expected).data, atol=1e-12)


def test_invalid_export_type_does_not_write() -> None:
    """Reject objects outside the program API before touching the output."""
    stream = BytesIO(b"keep me")
    with pytest.raises(TypeError, match="QCProgram or QCOProgram"):
        qpy.dump([object()], stream)  # ty: ignore[invalid-argument-type] Invalid public input.
    assert stream.getvalue() == b"keep me"


def test_short_write_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Incomplete output writes raise an I/O error."""
    stream = BytesIO()
    monkeypatch.setattr(stream, "write", lambda data: len(data) - 1)
    with pytest.raises(OSError, match="complete QPY payload"):
        qpy.dump([], stream)


@requires_translation
def test_failed_export_preserves_destination_and_source(tmp_path: Path) -> None:
    """An invalid QPY version must not truncate an existing destination or consume IR."""
    program = QCProgram.from_qiskit(QuantumCircuit(1))
    original_ir = program.ir
    destination = tmp_path / "keep.qpy"
    destination.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="version"):
        qpy.dump(program, destination, version=qiskit_qpy.QPY_VERSION + 1)
    assert destination.read_bytes() == b"keep me"
    assert program.ir == original_ir


@requires_translation
def test_failed_translation_preserves_destination_and_sources() -> None:
    """Failure in a later program leaves earlier programs valid and writes nothing."""
    valid = QCProgram.from_qiskit(QuantumCircuit(1))
    invalid = valid.copy()
    invalid.to_qco()
    stream = BytesIO(b"keep me")
    with pytest.raises(RuntimeError, match=r"[Ii]nvalid|consumed"):
        qpy.dump([valid, invalid], stream)
    assert valid.is_valid
    assert stream.getvalue() == b"keep me"


def test_qpy_import_keeps_qiskit_optional() -> None:
    """Importing the helper module must not load Qiskit or native compiler bindings."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mqt.core.qpy; "
                "assert 'qiskit' not in sys.modules; assert 'mqt.core.mlir' not in sys.modules"
            ),
        ],
        check=True,
    )
