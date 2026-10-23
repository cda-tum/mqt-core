/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <array>
#include <utility>

namespace mlir::qir {

/// Optional QIR module flags and their payload capability IDs.
inline constexpr std::array OPTIONAL_CAPABILITIES{
    std::pair{"dynamic_qubit_management", "qir.dynamic-qubit-management"},
    std::pair{"dynamic_result_management", "qir.dynamic-result-management"},
    std::pair{"arrays", "qir.arrays"},
    std::pair{"ir_functions", "qir.ir-functions"},
    std::pair{"multiple_return_points", "qir.multiple-return-points"},
    std::pair{"int_computations", "qir.int-computations"},
    std::pair{"float_computations", "qir.float-computations"},
    std::pair{"multiple_target_branching", "multiway-branching"},
};

} // namespace mlir::qir
