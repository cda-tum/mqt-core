# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ._version import version as __version__
from ._version import version_tuple as version_info
from .ir import QuantumComputation

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def load(input_circuit: QuantumComputation | str | os.PathLike[str] | QuantumCircuit) -> QuantumComputation:
    """Load a quantum circuit from any supported format as a :class:`~mqt.core.ir.QuantumComputation`.

    Args:
        input_circuit: The input circuit to translate to a :class:`~mqt.core.ir.QuantumComputation`.
                       This can be a :class:`~mqt.core.ir.QuantumComputation` itself,
                       a file name to any of the supported file formats,
                       an OpenQASM (2.0 or 3.0) string, or
                       a Qiskit :class:`~qiskit.circuit.QuantumCircuit`.

    Returns:
        The :class:`~mqt.core.ir.QuantumComputation`.

    Raises:
        ValueError: If the input circuit is a Qiskit :class:`~qiskit.circuit.QuantumCircuit`,
                    but the `qiskit` extra is not installed.
        FileNotFoundError: If the input circuit is a file name and the file does not exist.
    """
    if isinstance(input_circuit, QuantumComputation):
        return input_circuit

    if isinstance(input_circuit, (str, os.PathLike)):
        input_str = str(input_circuit)
        max_filename_length = 255 if os.name == "nt" else os.pathconf("/", "PC_NAME_MAX")
        if len(input_str) > max_filename_length or not Path(input_circuit).is_file():
            if isinstance(input_circuit, os.PathLike) or "OPENQASM" not in input_circuit:
                msg = f"File {input_circuit} does not exist."
                raise FileNotFoundError(msg)
            # otherwise, we assume that this is a QASM string
            return QuantumComputation.from_qasm(input_str)

        return QuantumComputation(input_str)

    try:
        from .plugins.qiskit import qiskit_to_mqt
    except ImportError:
        msg = "Qiskit is not installed. Please install `mqt.core[qiskit]` to use Qiskit circuits as input."
        raise ValueError(msg) from None

    return qiskit_to_mqt(input_circuit)


__all__ = ["__version__", "load", "version_info"]
