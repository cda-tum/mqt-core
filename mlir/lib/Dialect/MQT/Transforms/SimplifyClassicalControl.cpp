/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitAttributes.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include <cstdint>
#include <utility>

namespace mlir::mqt {

#define GEN_PASS_DEF_SIMPLIFYCLASSICALCONTROL
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {

struct RegisterValues {
  DenseMap<int64_t, Value> constants;
  DenseMap<Value, Value> dynamic;
  bool zeroInitialized = false;
};

struct Condition {
  bool truth;
  Block* block;
  Value constant;
};

/// Forward stored bits and scoped branch conditions.
///
/// Memory facts stay block-local; branch facts are scoped to the region in
/// which the condition is known.
class ClassicalControlSimplifier {
  IRRewriter rewriter;
  llvm::ScopedHashTable<Value, Condition*> conditions;

public:
  explicit ClassicalControlSimplifier(MLIRContext* context)
      : rewriter(context) {}

  void run(Region& region, bool hasConditions = false) {
    for (Block& block : region) {
      // Only scalar loads consume block-local memory facts.
      const bool hasLoads = !block.getOps<cbit::LoadOp>().empty();
      DenseMap<Value, RegisterValues> registers;
      for (Operation& operation : llvm::make_early_inc_range(block)) {
        if (hasConditions) {
          for (OpOperand& operand : operation.getOpOperands()) {
            if (auto* condition = conditions.lookup(operand.get())) {
              if (!condition->constant) {
                rewriter.setInsertionPointToStart(condition->block);
                condition->constant = arith::ConstantIntOp::create(
                    rewriter, operation.getLoc(),
                    static_cast<int64_t>(condition->truth), 1);
              }
              rewriter.modifyOpInPlace(
                  &operation, [&] { operand.set(condition->constant); });
            }
          }
        }
        if (hasLoads) {
          if (auto alloc = dyn_cast<cbit::AllocOp>(operation)) {
            if (alloc.getInitialization() == cbit::Initialization::Zero) {
              registers[alloc.getResult()].zeroInitialized = true;
            }
            continue;
          }
          if (auto store = dyn_cast<cbit::StoreOp>(operation)) {
            // Other register values may alias this storage.
            if (registers.size() > 1 ||
                (!registers.empty() && !registers.contains(store.getReg()))) {
              auto previous = std::move(registers[store.getReg()]);
              registers.clear();
              registers[store.getReg()] = std::move(previous);
            }
            auto& values = registers[store.getReg()];
            values.dynamic.clear();
            if (const auto index = getConstantIntValue(store.getIndex())) {
              values.constants[*index] = store.getValue();
            } else {
              values.constants.clear();
              values.zeroInitialized = false;
              values.dynamic[store.getIndex()] = store.getValue();
            }
            continue;
          }
          if (auto load = dyn_cast<cbit::LoadOp>(operation)) {
            auto& values = registers[load.getReg()];
            const auto index = getConstantIntValue(load.getIndex());
            Value known = index ? values.constants.lookup(*index)
                                : values.dynamic.lookup(load.getIndex());
            rewriter.setInsertionPoint(load);
            if (!known && values.zeroInitialized &&
                (index || values.constants.empty())) {
              known =
                  arith::ConstantIntOp::create(rewriter, load.getLoc(), 0, 1);
            }
            if (index) {
              values.constants[*index] = known ? known : load.getResult();
            } else {
              values.dynamic[load.getIndex()] =
                  known ? known : load.getResult();
            }
            if (known) {
              rewriter.replaceOp(load, known);
            }
            continue;
          }
          if (isa<cbit::ReadOp>(operation)) {
            continue;
          }
          if (!registers.empty()) {
            if (!isMemoryEffectFree(&operation)) {
              registers.clear();
            } else {
              for (Value operand : operation.getOperands()) {
                if (isa<cbit::RegisterType>(operand.getType())) {
                  registers.erase(operand);
                }
              }
            }
          }
        }
        auto branch = dyn_cast<qco::IfOp>(operation);
        for (auto [index, nested] : llvm::enumerate(operation.getRegions())) {
          llvm::ScopedHashTableScope<Value, Condition*> scope(conditions);
          Condition condition{
              .truth = index == 0,
              .block = nested.empty() ? nullptr : &nested.front(),
              .constant = {},
          };
          if (branch && !nested.empty()) {
            conditions.insert(branch.getCondition(), &condition);
          }
          run(nested, hasConditions || branch);
        }
      }
    }
  }
};

struct SimplifyClassicalControl final
    : impl::SimplifyClassicalControlBase<SimplifyClassicalControl> {
  using SimplifyClassicalControlBase::SimplifyClassicalControlBase;

protected:
  void runOnOperation() override { simplifyClassicalControl(getOperation()); }
};

} // namespace

void simplifyClassicalControl(Operation* operation) {
  ClassicalControlSimplifier simplifier(operation->getContext());
  for (Region& region : operation->getRegions()) {
    simplifier.run(region);
  }
}

} // namespace mlir::mqt
