"""IO handling within MQT Core."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from . import QuantumComputation

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def load(input_circuit: QuantumComputation | str | PathLike[str] | QuantumCircuit) -> QuantumComputation:
    """Load a quantum circuit from any supported format as a ``QuantumComputation``.

    Args:
        input_circuit: The input circuit to translate to a ``QuantumComputation``.

    Returns:
        The ``QuantumComputation``.

    Raises:
        ValueError: If the input circuit is a Qiskit `QuantumCircuit` but the `qiskit` extra is not installed.
        FileNotFoundError: If the input circuit is a file and the file does not exist.
    """
    if isinstance(input_circuit, QuantumComputation):
        return input_circuit

    if isinstance(input_circuit, (str, PathLike)):
        if not Path(input_circuit).is_file():
            msg = f"File {input_circuit} does not exist."
            raise FileNotFoundError(msg)

        return QuantumComputation(input_circuit)

    try:
        from mqt.core.plugins.qiskit import qiskit_to_mqt
    except ImportError:
        msg = "Qiskit is not installed. Please install `mqt.core[qiskit]` to use Qiskit circuits as input."
        raise ValueError(msg) from None

    return qiskit_to_mqt(input_circuit)


__all__ = ["load"]
