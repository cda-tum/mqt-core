# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: INP001

"""LIT Configuration file for the MQT MLIR test suite.

This file configures the LLVM LIT testing infrastructure for MLIR dialect tests.
"""

from __future__ import annotations

from pathlib import Path

import lit.formats
from lit.llvm import llvm_config

# Use `lit_config` to access `config` from lit.site.cfg.py
config = globals().get("config")
if config is None:
    msg = "LIT config object is missing. Ensure lit.site.cfg.py is loaded first."
    raise RuntimeError(msg)

config.name = "MQT MLIR test suite"
config.test_format = lit.formats.ShTest(execute_external=True)

# Define the file extensions to treat as test files.
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py"]

# Define the root path of where to look for tests.
config.test_source_root = Path(__file__).parent

# Define where to execute tests (and produce the output).
config.test_exec_root = getattr(config, "quantum_test_dir", ".lit")

# Define PATH to include the various tools needed for our tests.
try:
    # From within a build target we have access to cmake variables configured in lit.site.cfg.py.in.
    llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)  # FileCheck
    llvm_config.with_environment("PATH", config.quantum_bin_dir, append_path=True)  # quantum-opt
except AttributeError:
    # The system PATH is available by default.
    pass
