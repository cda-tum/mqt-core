# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Opt-in collection of tests that require a source checkout."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_OPTIONAL_TESTS = {
    "test_docs_links.py": "link-checks",
    "test_qiskit_c_api_adopt.py": "qiskit-c-api",
    "test_slurm_integration.py": "slurm",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register source-only test suites."""
    for suite in _OPTIONAL_TESTS.values():
        parser.addoption(
            f"--run-{suite}", action="store_true", help=f"Include {suite} tests (requires a source checkout)"
        )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Exclude optional suites before importing their repository scripts.

    Returns:
        True for disabled suites; otherwise leave collection to pytest.
    """
    suite = _OPTIONAL_TESTS.get(collection_path.name)
    if suite is not None and not config.getoption(f"--run-{suite}"):
        return True
    return None
