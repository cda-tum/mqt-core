# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Qiskit Plugin."""

# ruff: file-ignore[non-empty-init-module]

from __future__ import annotations

__lazy_modules__ = {
    "mqt.core.plugins.qiskit.backend",
    "mqt.core.plugins.qiskit.exceptions",
    "mqt.core.plugins.qiskit.job",
    "mqt.core.plugins.qiskit.provider",
    "mqt.core.plugins.qiskit.serializers",
}

from importlib import import_module
from typing import TYPE_CHECKING

try:
    import_module("qiskit")
except ModuleNotFoundError as error:
    if error.name != "qiskit":
        raise
    HAS_QISKIT = False
else:
    HAS_QISKIT = True

__all__ = [
    "HAS_QISKIT",
]

if TYPE_CHECKING or HAS_QISKIT:
    from mqt.core.plugins.qiskit.backend import QDMIBackend
    from mqt.core.plugins.qiskit.exceptions import (
        CircuitValidationError,
        JobExecutionError,
        JobSubmissionError,
        QDMIQiskitError,
        TranslationError,
        UnsupportedFormatError,
        UnsupportedOperationError,
    )
    from mqt.core.plugins.qiskit.job import QDMIJob
    from mqt.core.plugins.qiskit.provider import QDMIProvider
    from mqt.core.plugins.qiskit.serializers import (
        ProgramSerializer,
        program_serializer,
        register_program_serializer,
        unregister_program_serializer,
    )

    # Load the backend now to register the built-in OpenQASM serializers
    import_module("mqt.core.plugins.qiskit.backend")

    __all__ += [
        "CircuitValidationError",
        "JobExecutionError",
        "JobSubmissionError",
        "ProgramSerializer",
        "QDMIBackend",
        "QDMIJob",
        "QDMIProvider",
        "QDMIQiskitError",
        "TranslationError",
        "UnsupportedFormatError",
        "UnsupportedOperationError",
        "program_serializer",
        "register_program_serializer",
        "unregister_program_serializer",
    ]
