/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitOps.h"

#include "mqt/Dialect/CBit/IR/CBitAttributes.h" // IWYU pragma: associated
#include "mqt/Dialect/CBit/IR/CBitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

using namespace mlir;
using namespace mlir::cbit;

void AllocOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  const auto name =
      getOperation()->getAttrOfType<StringAttr>("mqt.register_name");
  setNameFn(getResult(), name ? name.getValue() : "c");
}

void LoadOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "b");
}

void ReadOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "val");
}

#include "mqt/Dialect/CBit/IR/CBitOpsDialect.cpp.inc"
#include "mqt/Dialect/CBit/IR/CBitOpsEnums.cpp.inc"

void CBitDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mqt/Dialect/CBit/IR/CBitOpsAttributes.cpp.inc"

      >();

  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mqt/Dialect/CBit/IR/CBitOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mqt/Dialect/CBit/IR/CBitOps.cpp.inc"

      >();
}

#define GET_ATTRDEF_CLASSES
#include "mqt/Dialect/CBit/IR/CBitOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mqt/Dialect/CBit/IR/CBitOpsTypes.cpp.inc"

LogicalResult
RegisterType::verify(const function_ref<InFlightDiagnostic()> emitError,
                     const int64_t width) {
  if (width <= 0) {
    return emitError() << "register width must be positive";
  }
  return success();
}

void mlir::cbit::validateStaticRegisterIndex(
    Value reg, const std::variant<int64_t, Value>& index) {
  const auto type = dyn_cast<RegisterType>(reg.getType());
  if (!type) {
    llvm::reportFatalUsageError("Expected a CBit register");
  }

  const auto* constant = std::get_if<int64_t>(&index);
  if (constant == nullptr) {
    return;
  }
  if (*constant < 0) {
    llvm::reportFatalUsageError("Register index must be non-negative");
  }
  if (*constant >= type.getWidth()) {
    llvm::reportFatalUsageError("Register index is out of bounds");
  }
}

static LogicalResult verifyIndex(Operation* operation, Value registerValue,
                                 Value indexValue) {
  const auto index = getConstantIntValue(indexValue);
  if (!index) {
    return success();
  }

  if (*index < 0) {
    return operation->emitOpError("index must be non-negative");
  }

  const auto width = cast<RegisterType>(registerValue.getType()).getWidth();
  if (*index >= width) {
    return operation->emitOpError("index exceeds register width");
  }
  return success();
}

/// Find a known snapshot or zero initialization; a null index denotes a whole
/// read.
static OpFoldResult findKnownValue(Operation* read, Value reg, Value index) {
  const bool singleBit = cast<RegisterType>(reg.getType()).getWidth() == 1;
  const auto matchesIndex = [&](Value candidate) {
    return index ? isEqualConstantIntOrValue(index, candidate)
                 : singleBit && getConstantIntValue(candidate) == 0;
  };
  const bool readsWholeRegister =
      !index || (singleBit && getConstantIntValue(index) == 0);

  // Bound the work per read so long blocks cannot cause quadratic searches.
  unsigned scanned = 0;
  for (auto* previous = read->getPrevNode();
       previous != nullptr && scanned < 16;
       previous = previous->getPrevNode(), ++scanned) {
    if (auto load = dyn_cast<LoadOp>(previous)) {
      if (load.getReg() == reg && matchesIndex(load.getIndex())) {
        return load.getResult();
      }
      continue;
    }
    if (auto snapshot = dyn_cast<ReadOp>(previous)) {
      if (snapshot.getReg() == reg && readsWholeRegister) {
        return snapshot.getResult();
      }
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(previous)) {
      if (store.getReg() == reg) {
        if (matchesIndex(store.getIndex())) {
          return store.getValue();
        }
        const auto storedIndex = getConstantIntValue(store.getIndex());
        const auto loadedIndex =
            index ? getConstantIntValue(index) : std::nullopt;
        if (storedIndex && loadedIndex && storedIndex != loadedIndex) {
          continue;
        }
      }
      // Distinct register arguments may refer to the same storage.
      return {};
    }
    if (auto write = dyn_cast<WriteOp>(previous)) {
      return write.getReg() == reg && readsWholeRegister
                 ? OpFoldResult(write.getValue())
                 : OpFoldResult{};
    }
    if (auto alloc = dyn_cast<AllocOp>(previous);
        alloc && alloc.getResult() == reg) {
      return alloc.getInitialization() == Initialization::Zero
                 ? OpFoldResult(
                       IntegerAttr::get(read->getResult(0).getType(), 0))
                 : OpFoldResult{};
    }
    if (previous->getNumRegions() != 0 || !isMemoryEffectFree(previous)) {
      return {};
    }
  }
  return {};
}

static LogicalResult forwardKnownValue(Operation* read, Value reg, Value index,
                                       PatternRewriter& rewriter) {
  auto known = findKnownValue(read, reg, index);
  if (!known) {
    return failure();
  }
  if (auto value = dyn_cast<Value>(known)) {
    rewriter.replaceOp(read, value);
  } else {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        read, cast<IntegerAttr>(cast<Attribute>(known)));
  }
  return success();
}

namespace {
struct EraseWriteOnlyRegister final : OpRewritePattern<AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp alloc,
                                PatternRewriter& rewriter) const override {
    if (!llvm::all_of(alloc.getResult().getUsers(), [](Operation* user) {
          return isa<StoreOp, WriteOp>(user);
        })) {
      return failure();
    }
    for (auto* user :
         llvm::make_early_inc_range(alloc.getResult().getUsers())) {
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(alloc);
    return success();
  }
};

struct DecomposeRead final : OpRewritePattern<ReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadOp read,
                                PatternRewriter& rewriter) const override {
    auto loc = read.getLoc();
    const auto type = read.getResult().getType();
    const auto width = type.getWidth();
    Value result;
    for (unsigned index = 0; index < width; ++index) {
      auto indexValue = arith::ConstantIndexOp::create(rewriter, loc, index);
      Value bit = LoadOp::create(rewriter, loc, rewriter.getI1Type(),
                                 read.getReg(), indexValue);
      if (width != 1) {
        bit = arith::ExtUIOp::create(rewriter, loc, type, bit);
      }
      if (index == 0) {
        result = bit;
      } else {
        auto shift = arith::ConstantIntOp::create(rewriter, loc, type, index);
        bit = arith::ShLIOp::create(rewriter, loc, bit, shift);
        result = arith::OrIOp::create(rewriter, loc, result, bit);
      }
    }
    rewriter.replaceOp(read, result);
    return success();
  }
};

struct DecomposeWrite final : OpRewritePattern<WriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WriteOp write,
                                PatternRewriter& rewriter) const override {
    auto loc = write.getLoc();
    const auto type = write.getValue().getType();
    const auto width = type.getWidth();
    for (unsigned index = 0; index < width; ++index) {
      Value bit = write.getValue();
      if (index != 0) {
        auto shift = arith::ConstantIntOp::create(rewriter, loc, type, index);
        bit = arith::ShRUIOp::create(rewriter, loc, bit, shift);
      }
      if (width != 1) {
        bit = arith::TruncIOp::create(rewriter, loc, rewriter.getI1Type(), bit);
      }
      auto indexValue = arith::ConstantIndexOp::create(rewriter, loc, index);
      StoreOp::create(rewriter, loc, bit, write.getReg(), indexValue);
    }
    rewriter.eraseOp(write);
    return success();
  }
};

} // namespace

LogicalResult LoadOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

LogicalResult ReadOp::verify() {
  if (std::cmp_not_equal(getResult().getType().getWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("result width must match register width");
  }
  return success();
}

LogicalResult WriteOp::verify() {
  if (std::cmp_not_equal(getValue().getType().getWidth(),
                         getReg().getType().getWidth())) {
    return emitOpError("value width must match register width");
  }
  return success();
}

void mlir::cbit::populateCBitDecompositionPatterns(
    RewritePatternSet& patterns) {
  patterns.add<DecomposeRead, DecomposeWrite>(patterns.getContext());
}

void AllocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<EraseWriteOnlyRegister>(context);
}

LogicalResult LoadOp::canonicalize(LoadOp load, PatternRewriter& rewriter) {
  return forwardKnownValue(load, load.getReg(), load.getIndex(), rewriter);
}

LogicalResult ReadOp::canonicalize(ReadOp read, PatternRewriter& rewriter) {
  return forwardKnownValue(read, read.getReg(), {}, rewriter);
}

LogicalResult StoreOp::verify() {
  return verifyIndex(getOperation(), getReg(), getIndex());
}

#define GET_OP_CLASSES
#include "mqt/Dialect/CBit/IR/CBitOps.cpp.inc"
