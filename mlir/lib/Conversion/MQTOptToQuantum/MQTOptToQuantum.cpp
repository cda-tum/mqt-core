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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
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

struct ConvertMQTOptAlloc
    : public OpConversionPattern<::mqt::ir::opt::AllocOp> {

  ConvertMQTOptAlloc(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::AllocOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto const nQubitsValue = adaptor.getSize();
    auto const nQubitsIntegerAttr = adaptor.getSizeAttrAttr();

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::AllocOp>(
        op.getLoc(), resultType, nQubitsValue, nQubitsIntegerAttr);

    // Get the result of the new operation, which represents the qubit register
    auto trgtQreg = catalystOp->getResult(0);

    // Collect the users of the original operation to update their operands
    std::vector<mlir::Operation*> users(op->getUsers().begin(),
                                        op->getUsers().end());

    // Iterate over the users in reverse order
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

struct ConvertMQTOptDealloc
    : public OpConversionPattern<::mqt::ir::opt::DeallocOp> {

  ConvertMQTOptDealloc(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::DeallocOp>(typeConverter, context) {
  }

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto qregValue = adaptor.getQureg();

    // Prepare the result type(s)
    auto resultTypes = ::mlir::TypeRange({});

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::DeallocOp>(
        op.getLoc(), resultTypes, qregValue);

    // Replace the MQT operation with the Catalyst operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertMQTOptExtract
    : public OpConversionPattern<::mqt::ir::opt::ExtractOp> {

  ConvertMQTOptExtract(const TypeConverter& typeConverter, MLIRContext* context)
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

    auto mqtQreg = op->getResult(0);
    auto catalystQreg = catalystOp.getOperand(0);

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

struct ConvertMQTOptInsert
    : public OpConversionPattern<::mqt::ir::opt::InsertOp> {

  // Explicit constructor that initializes the reference and passes to the base
  // constructor
  ConvertMQTOptInsert(const TypeConverter& typeConverter, MLIRContext* context)
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

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::InsertOp>(
        op.getLoc(), resultType, inQregValue, idxValue, idxIntegerAttr,
        qubitValue);

    // Replace the MQT operation with the Catalyst operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

template <typename MQTGateOp>
struct ConvertMQTOptSimpleGate : public OpConversionPattern<MQTGateOp> {
  using OpConversionPattern<MQTGateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOp op, typename MQTGateOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto paramsValues = adaptor.getParams();
    auto inQubitsValues = adaptor.getInQubits(); // excl. controls
    auto posCtrlQubitsValues = adaptor.getPosCtrlQubits();
    auto negCtrlQubitsValues = adaptor.getNegCtrlQubits();

    llvm::SmallVector<mlir::Value> inCtrlQubits;
    inCtrlQubits.append(posCtrlQubitsValues.begin(), posCtrlQubitsValues.end());
    inCtrlQubits.append(negCtrlQubitsValues.begin(), negCtrlQubitsValues.end());

    // Output type
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubits.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Determine gate name depending on control count
    llvm::StringRef gateName = getGateName(inCtrlQubits.size());
    if (gateName.empty()) {
      llvm::errs() << "Unsupported controlled gate for op: " << op->getName()
                   << "\n";
      return failure();
    }

    // TODO: This is not nice (HACK) but controlled gates expect runtime values.
    llvm::SmallVector<mlir::Value> i1Values;
    if (!inCtrlQubits.empty()) {
      mlir::Type i1Type = rewriter.getI1Type();
      auto constTrue = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), i1Type, rewriter.getBoolAttr(true));
      for (size_t i = 0; i < inCtrlQubits.size(); ++i) {
        i1Values.push_back(constTrue.getResult());
      }
    }
    mlir::ValueRange inCtrlQubitsValues(i1Values);

    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, paramsValues, inQubitsValues, gateName,
        nullptr, inCtrlQubits, inCtrlQubitsValues);

    catalystOp.getProperties().setResultSegmentSizes(
        {static_cast<int>(inQubitsValues.size()),
         static_cast<int>(inCtrlQubits.size())});

    rewriter.replaceOp(op, catalystOp);
    return success();
  }

private:
  // Specialize this for each gate type
  static llvm::StringRef getGateName(std::size_t numControls);
};

// -- XOp (PauliX, CNOT, Toffoli)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::XOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PauliX";
  if (numControls == 1)
    return "CNOT";
  if (numControls == 2)
    return "Toffoli";
  return "";
}

// -- YOp (PauliY, CY)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::YOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PauliY";
  if (numControls == 1)
    return "CY";
  return "";
}

// -- ZOp (PauliZ, CZ)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::ZOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PauliZ";
  if (numControls == 1)
    return "CZ";
  return "";
}

// -- HOp (Hadamard)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::HOp>::getGateName(
    std::size_t numControls) {
  return "Hadamard";
}

// -- SWAPOp (SWAP)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::SWAPOp>::getGateName(
    std::size_t numControls) {
  return "SWAP";
}

// -- RXOp (RX, CRX)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RXOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "RX";
  if (numControls == 1)
    return "CRX";
  return "";
}

// -- RYOp (RY, CRY)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RYOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "RY";
  if (numControls == 1)
    return "CRY";
  return "";
}

// -- RZOp (RZ, CRZ)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::RZOp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "RZ";
  if (numControls == 1)
    return "CRZ";
  return "";
}

// -- POp (PhaseShift, ControlledPhaseShift)
template <>
llvm::StringRef ConvertMQTOptSimpleGate<::mqt::ir::opt::POp>::getGateName(
    std::size_t numControls) {
  if (numControls == 0)
    return "PhaseShift";
  if (numControls == 1)
    return "ControlledPhaseShift";
  return "";
}

struct ConvertMQTOptH : public OpConversionPattern<::mqt::ir::opt::HOp> {

  // Explicit constructor that initializes the reference and passes to the base
  // constructor
  ConvertMQTOptH(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::HOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::HOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    auto paramsValues = adaptor.getParams();
    auto inQubitsValues = adaptor.getInQubits();
    auto inCtrlQubitsValues = adaptor.getPosCtrlQubits();
    // auto inNegCtrlQubitsValues = adaptor.getNegCtrlQubits();

    // Prepare the result type(s)
    mlir::Type qubitType =
        catalyst::quantum::QubitType::get(rewriter.getContext());
    std::vector<mlir::Type> qubitTypes(
        inQubitsValues.size() + inCtrlQubitsValues.size(), qubitType);
    auto outQubitTypes = mlir::TypeRange(qubitTypes);

    // Create the new operation
    llvm::StringRef gateName = "Hadamard";

    auto ctrlValues = ValueRange(inCtrlQubitsValues);

    // TODO: check arguments
    auto catalystOp = rewriter.create<catalyst::quantum::CustomOp>(
        op.getLoc(), outQubitTypes, paramsValues, inQubitsValues, gateName,
        nullptr, inCtrlQubitsValues, ctrlValues);

    catalystOp->emitRemark() << "In count: " << inQubitsValues.size()
                             << " Ctrl count: " << inCtrlQubitsValues.size();

    catalystOp.getProperties().setResultSegmentSizes(
        {static_cast<int>(inQubitsValues.size()),
         static_cast<int>(inCtrlQubitsValues.size())}); // TODO: necessary?

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertMQTOptMeasure
    : public OpConversionPattern<::mqt::ir::opt::MeasureOp> {

  // Explicit constructor that initializes the reference and passes to the base
  // constructor
  ConvertMQTOptMeasure(const TypeConverter& typeConverter, MLIRContext* context)
      : OpConversionPattern<::mqt::ir::opt::MeasureOp>(typeConverter, context) {
  }

  LogicalResult
  matchAndRewrite(::mqt::ir::opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s) and attribute(s)
    auto qubitValues = adaptor.getInQubits();

    // Prepare the result type(s)
    auto bitType = rewriter.getI1Type();
    auto qubitType = catalyst::quantum::QubitType::get(rewriter.getContext());

    std::vector<mlir::Type> bitTypes(qubitValues.size(), bitType);
    std::vector<mlir::Type> qubitTypes(qubitValues.size(), qubitType);

    // Fill the resultTypes: qubitTypes + bitTypes
    auto resultTypesVec = std::vector<mlir::Type>();
    for (auto qubitType : qubitTypes) {
      resultTypesVec.push_back(qubitType);
    }
    for (auto qubitType : qubitTypes) {
      resultTypesVec.push_back(qubitType);
    }
    auto resultTypesRange = mlir::TypeRange(resultTypesVec);

    llvm::outs() << "Reached MeasureOp\n"; // DEBUGGING

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::MeasureOp>(
        op.getLoc(), resultTypesRange, qubitValues);

    llvm::outs() << "Replaced MeasureOp\n"; // DEBUGGING

    // Collect all original outQubits and outBits
    std::vector<mlir::Value> outQubits;
    std::vector<mlir::Value> outBits;

    for (auto result : op->getResults()) {
      if (mlir::isa<::mqt::ir::opt::QubitType>(result.getType())) {
        outQubits.push_back(result);
      } else {
        outBits.push_back(result);
      }
    }

    llvm::outs() << "Found " << outQubits.size() << " outQubits\n"; // DEBUGGING
    llvm::outs() << "Found " << outBits.size() << " outBits\n";     // DEBUGGING

    // Iterate over all uses of the outQubits and replace with the inQubit
    int qubitIdx = 0;
    for (auto outQubit : outQubits) {
      op.emitRemark() << "Looking for users of " << outQubit;
      for (auto user : outQubit.getUsers()) {
        user->emitRemark() << "Replacing " << outQubit << " with "
                           << qubitValues[qubitIdx];
        user->replaceUsesOfWith(outQubit, qubitValues[qubitIdx]);
      }
      qubitIdx++;
    }

    llvm::outs() << "Replaced outQubits\n"; // DEBUGGING

    // Iterate over all uses of the outBits and replace with the new bits
    int bitIdx = 0;
    for (auto outBit : outBits) {
      catalystOp.emitRemark() << "Looking for users of " << outBit;
      for (auto user : outBit.getUsers()) {
        user->emitRemark() << "Replacing " << outBit << " with "
                           << catalystOp.getResult(qubitIdx + bitIdx);
        user->replaceUsesOfWith(outBit,
                                catalystOp.getResult(qubitIdx + bitIdx));
      }
      bitIdx++;
    }

    llvm::outs() << "Replaced outBits\n"; // DEBUGGING

    rewriter.eraseOp(op);

    llvm::outs() << "Erased MeasureOp\n"; // DEBUGGING

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

    patterns.add<ConvertMQTOptAlloc, ConvertMQTOptDealloc, ConvertMQTOptExtract,
                 ConvertMQTOptInsert, ConvertMQTOptMeasure>(typeConverter,
                                                            context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::XOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::YOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::ZOp>>(typeConverter,
                                                               context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RXOp>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RYOp>>(typeConverter,
                                                                context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::RZOp>>(typeConverter,
                                                                context);

    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::HOp>>(typeConverter,
                                                               context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::SWAPOp>>(typeConverter,
                                                                  context);
    patterns.add<ConvertMQTOptSimpleGate<::mqt::ir::opt::POp>>(typeConverter,
                                                               context);

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
