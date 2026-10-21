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

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> qaoa(qc::QCProgramBuilder& builder, const QAOA& benchmark) {
  const auto& options = benchmark.options();
  const auto size = static_cast<int64_t>(options.qubits);
  const auto layers = static_cast<int64_t>(options.depth);
  const auto parameterCount = 2 * options.depth;
  auto entry = cast<func::FuncOp>(builder.getInsertionBlock()->getParentOp());
  const SmallVector<Type> types(parameterCount, builder.getF64Type());
  entry.getBody().front().addArguments(
      types, SmallVector<Location>(parameterCount, builder.getLoc()));
  entry.setType(builder.getFunctionType(types, {}));
  const auto inputName =
      mlir::mqt::MQTDialect::InputNameAttrHelper::getNameStr();
  SmallVector<DictionaryAttr> attributes(parameterCount);
  for (size_t layer = 0; layer < options.depth; ++layer) {
    const auto suffix = std::to_string(layer);
    attributes[layer] = builder.getDictionaryAttr({
        builder.getNamedAttr(inputName,
                             builder.getStringAttr("gamma_" + suffix)),
    });
    attributes[options.depth + layer] = builder.getDictionaryAttr({
        builder.getNamedAttr(inputName,
                             builder.getStringAttr("beta_" + suffix)),
    });
  }
  entry.setAllArgAttrs(attributes);
  auto arguments = entry.getArguments();
  auto gammas = tensor::FromElementsOp::create(
      builder, ValueRange(arguments.take_front(options.depth)));
  auto betas = tensor::FromElementsOp::create(
      builder, ValueRange(arguments.drop_front(options.depth)));
  auto qubits = builder.allocQubitRegisterStorage(size, "vertices");
  auto result =
      builder.allocClassicalBitRegister(size, benchmark.output().name);
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(size - 1);
  auto two = builder.floatConstant(2.);

  builder.scfFor(0, size, 1,
                 [&](Value iv) { builder.h(builder.loadQubit(qubits, iv)); });
  builder.scfFor(0, layers, 1, [&](Value layer) {
    auto gamma = tensor::ExtractOp::create(builder, gammas, ValueRange{layer});
    auto beta = tensor::ExtractOp::create(builder, betas, ValueRange{layer});
    // For C_jk = (1 - Z_j Z_k)/2, RZZ(-gamma) differs only by a global phase.
    auto costAngle = arith::NegFOp::create(builder, gamma);
    auto mixerAngle = arith::MulFOp::create(builder, two, beta);
    builder.scfFor(0, size - 1, 1, [&](Value iv) {
      auto next = arith::AddIOp::create(builder, iv, one);
      builder.rzz(costAngle, builder.loadQubit(qubits, iv),
                  builder.loadQubit(qubits, next));
    });
    builder.rzz(costAngle, builder.loadQubit(qubits, last),
                builder.loadQubit(qubits, zero));
    builder.scfFor(0, size, 1, [&](Value iv) {
      builder.rx(mixerAngle, builder.loadQubit(qubits, iv));
    });
  });
  builder.measureQubitRegister(qubits, result, size);
  return {result};
}

} // namespace mqt::bench
