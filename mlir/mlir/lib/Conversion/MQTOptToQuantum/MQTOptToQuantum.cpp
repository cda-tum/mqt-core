/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTOptToQuantum/MQTOptToQuantum.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DEF_MQTOPTTOQUANTUM
#include "mlir/Conversion/MQTOptToQuantum/MQTOptToQuantum.h.inc"

using namespace mlir;

class MQTOptToQuantumTypeConverter : public TypeConverter {
public:
  explicit MQTOptToQuantumTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if
    // needed.
    addConversion([](Type type) { return type; });

    // Convert QubitRegisterType to QuregType
    addConversion([ctx](::mqt::ir::opt::QubitRegisterType /*type*/) -> Type {
      return catalyst::quantum::QuregType::get(ctx);
    });

    // Convert QubitType in the new dialect to Catalyst's QubitType
    addConversion([ctx](::mqt::ir::opt::QubitType /*type*/) -> Type {
      return catalyst::quantum::QubitType::get(ctx);
    });
  }
};

struct ConvertAlloc : public OpConversionPattern<::mqt::ir::opt::AllocOp> {

  ConvertAlloc(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::AllocOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto const nQubitsValue = adaptor.getSize();
    auto const nQubitsIntegerAttr = adaptor.getSizeAttrAttr();

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the original Catalyst operation
    auto catalystOp = rewriter.create<catalyst::quantum::AllocOp>(
        op.getLoc(), resultType, nQubitsValue, nQubitsIntegerAttr);

    // Get the result of the new operation, which represents the qubit register
    auto trgtQreg = catalystOp->getResult(0);

    // Collect the users of the original operation to update their operands
    std::vector<mlir::Operation*> users(op->getUsers().begin(),
                                        op->getUsers().end());

    // Iterate over the users in (TODO: reverse?) order
    for (auto* user : llvm::reverse(users)) {
      // Registers should only be used in Extract, Insert or Dealloc operations
      if (mlir::isa<::mqt::ir::opt::ExtractOp>(user) ||
          mlir::isa<::mqt::ir::opt::InsertOp>(user) ||
          mlir::isa<::mqt::ir::opt::DeallocOp>(user)) {
        // Update the operand of the user operation to the new qubit register
        user->setOperand(0, trgtQreg);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertDealloc : public OpConversionPattern<::mqt::ir::opt::DeallocOp> {

  ConvertDealloc(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::DeallocOp>(typeConverter, context) {
  }

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto qregValue = adaptor.getQureg();

    // Prepare the result type(s)
    auto resultTypes = ::mlir::TypeRange({});

    // Create the original Catalyst operation
    auto catalystOp = rewriter.create<catalyst::quantum::DeallocOp>(
        op.getLoc(), resultTypes, qregValue);

    // Replace the MQT operation with the Catalyst operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<::mqt::ir::opt::ExtractOp> {

  ConvertExtract(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::ExtractOp>(typeConverter, context) {
    this->setHasBoundedRewriteRecursion(true);
  }

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s), and attribute(s)
    auto qregValue = adaptor.getInQureg();
    auto idxValue = adaptor.getIndex();
    auto idxIntegerAttr = adaptor.getIndexAttrAttr();

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::ExtractOp>(
        op.getLoc(), resultType, qregValue, idxValue, idxIntegerAttr);

    auto mqtQreg = op.getOperand(0);
    auto catalystQreg = catalystOp->getResult(0);

    // Collect the users of the original input qubit register to update their
    // operands
    std::vector<mlir::Operation*> users(mqtQreg.getUsers().begin(),
                                        mqtQreg.getUsers().end());

    // Iterate over users in (TODO: reverse?) order to update their operands
    // properly
    for (auto* user : llvm::reverse(users)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {
        // Update operands in the user operation
        if (mlir::isa<::mqt::ir::opt::ExtractOp>(user) ||
            mlir::isa<::mqt::ir::opt::InsertOp>(user) ||
            mlir::isa<::mqt::ir::opt::DeallocOp>(user)) {
          user->setOperand(0, catalystQreg);
        }
      }
    }

    // Collect the users of the original output qubit
    auto oldQubit = op->getResult(1);
    auto newQubit = catalystOp->getResult(0);

    std::vector<mlir::Operation*> qubitUsers(oldQubit.getUsers().begin(),
                                             oldQubit.getUsers().end());

    // Iterate over qubit users in (TODO: reverse?) order
    for (auto* user : llvm::reverse(qubitUsers)) {

      // Only consider operations after the current operation
      if (!user->isBeforeInBlock(catalystOp) && user != catalystOp &&
          user != op) {

        auto operandIdx = 0;
        for (auto operand : user->getOperands()) {
          if (operand == oldQubit) {
            user->setOperand(operandIdx, newQubit);
          }
          operandIdx++;
        }
      }
    }

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertInsert : public OpConversionPattern<::mqt::ir::opt::InsertOp> {

  // Explicit constructor that initializes the reference and passes to the base
  // constructor
  ConvertInsert(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::InsertOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto inQregValue = adaptor.getInQureg();
    auto qubitValue = adaptor.getInQubit();
    auto idxValue = adaptor.getIndex();
    auto idxIntegerAttr = adaptor.getIndexAttrAttr();

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the original Catalyst operation
    auto catalystOp = rewriter.create<catalyst::quantum::InsertOp>(
        op.getLoc(), resultType, inQregValue, idxValue, idxIntegerAttr,
        qubitValue);

    // Replace the MQT operation with the Catalyst operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertCustom : public OpConversionPattern<catalyst::quantum::CustomOp> {

  // Explicit constructor that initializes the reference and passes to the base
  // constructor
  ConvertCustom(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<catalyst::quantum::CustomOp>(typeConverter,
                                                         context) {}

  LogicalResult
  matchAndRewrite(catalyst::quantum::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto gateName = op.getGateName();

    // Extract operand(s) and attribute(s)
    auto paramsValues = adaptor.getParams();
    auto inQubitsValues = adaptor.getInQubits();
    auto inCtrlQubitsValues = adaptor.getInCtrlQubits();
    // TODO: extract actual values
    auto inNegCtrlQubitsValues = mlir::ValueRange({});
    auto staticParams = ::mlir::DenseF64ArrayAttr();
    auto paramsMask = ::mlir::DenseBoolArrayAttr();

    // Prepare the result type(s)
    mlir::Type qubitType =
        ::mqt::ir::opt::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubitsValues.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Create the new operation
    Operation* catalystOp = nullptr;

    if (gateName.compare("Hadamard") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::HOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliX") == 0 ||
               gateName.compare("CNOT") == 0 ||
               gateName.compare("Toffoli") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::XOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliY") == 0 || gateName.compare("CY") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::YOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PauliZ") == 0 || gateName.compare("CZ") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::ZOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("SWAP") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::SWAPOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("RX") == 0 || gateName.compare("CRX") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::RXOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("RY") == 0 || gateName.compare("CRY") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::RYOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("RZ") == 0 || gateName.compare("CRZ") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::RZOp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else if (gateName.compare("PhaseShift") == 0 ||
               gateName.compare("ControlledPhaseShift") == 0) {
      catalystOp = rewriter.create<::mqt::ir::opt::POp>(
          op.getLoc(), outQubitTypes, staticParams, paramsMask, paramsValues,
          inQubitsValues, inCtrlQubitsValues, inNegCtrlQubitsValues);
    } else {
      llvm::errs() << "Unsupported gate: " << gateName << "\n";
      return failure();
    }

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct MQTOptToQuantum : impl::MQTOptToQuantumBase<MQTOptToQuantum> {
  using MQTOptToQuantumBase::MQTOptToQuantumBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<catalyst::quantum::QuantumDialect>();
    target.addIllegalDialect<::mqt::ir::opt::MQTOptDialect>();

    RewritePatternSet patterns(context);
    MQTOptToQuantumTypeConverter typeConverter(context);

    patterns.add<ConvertAlloc, ConvertDealloc, ConvertExtract, ConvertInsert,
                 ConvertCustom>(typeConverter, context);

    // Boilerplate code to prevent: unresolved materialization
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::mqt::ir::conversions
