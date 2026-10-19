/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QAOA.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstddef>
#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> qaoa(qc::QCProgramBuilder& builder, const QAOA& benchmark) {
  const auto& options = benchmark.options();
  const auto size = static_cast<int64_t>(options.qubits);
  auto qubits = builder.allocQubitRegisterStorage(size, "vertices");
  auto result =
      builder.allocClassicalBitRegister(size, benchmark.output().name);
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(size - 1);

  builder.scfFor(0, size, 1,
                 [&](Value iv) { builder.h(builder.loadQubit(qubits, iv)); });
  for (size_t layer = 0; layer < options.gammas.size(); ++layer) {
    // For C_jk = (1 - Z_j Z_k)/2, RZZ(-gamma) differs only by a global phase.
    const auto costAngle = -options.gammas[layer];
    builder.scfFor(0, size - 1, 1, [&](Value iv) {
      auto next = arith::AddIOp::create(builder, iv, one);
      builder.rzz(costAngle, builder.loadQubit(qubits, iv),
                  builder.loadQubit(qubits, next));
    });
    builder.rzz(costAngle, builder.loadQubit(qubits, last),
                builder.loadQubit(qubits, zero));
    builder.scfFor(0, size, 1, [&](Value iv) {
      auto qubit = builder.loadQubit(qubits, iv);
      builder.rx(2. * options.betas[layer], qubit);
      if (layer + 1 == options.gammas.size()) {
        builder.measure(qubit, result, iv);
      }
    });
  }
  return {result};
}

} // namespace mqt::bench
