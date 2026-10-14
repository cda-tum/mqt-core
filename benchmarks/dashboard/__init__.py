# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT compiler benchmark dashboard."""

from benchmarks.dashboard._report import METRICS, build_dashboard, read_results

__all__ = ["METRICS", "build_dashboard", "read_results"]
