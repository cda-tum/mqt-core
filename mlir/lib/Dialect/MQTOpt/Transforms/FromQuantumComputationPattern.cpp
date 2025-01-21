#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <set>

namespace mqt::ir::opt {
/// Analysis pattern that filters out all quantum operations from a given
/// program and creates a quantum computation from them.
struct FromQuantumComputationPattern final : mlir::OpRewritePattern<AllocOp> {
  qc::QuantumComputation& circuit;

  explicit FromQuantumComputationPattern(mlir::MLIRContext* context,
                                         qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

  mlir::LogicalResult match(AllocOp op) const override {
    return op->hasAttr("to_replace") ? mlir::success() : mlir::failure();
  }

  ExtractOp createRegisterAccess(AllocOp& reg, size_t index,
                                 mlir::PatternRewriter& rewriter) const {
    return rewriter.create<ExtractOp>(
        reg.getLoc(), mlir::TypeRange{QubitRegisterType::get(rewriter.getContext()), QubitType::get(rewriter.getContext())}, reg.getResult(),
        nullptr, rewriter.getI64IntegerAttr(index));
  }

  XOp createXOp(mlir::Location loc,
                          mlir::Value inQubit,
                          mlir::ValueRange controlQubits,
                          mlir::PatternRewriter& rewriter) const {
    llvm::SmallVector<mlir::Type, 4> resultTypes(4, inQubit.getType());
    return rewriter.create<XOp>(loc, resultTypes, inQubit, controlQubits);
  }

  MeasureOp createMeasureOp(mlir::Location loc,
                          mlir::Value targetQubit,
                          mlir::PatternRewriter& rewriter) const {
    return rewriter.create<MeasureOp>(loc, mlir::TypeRange{QubitType::get(rewriter.getContext()), rewriter.getI1Type()}, targetQubit);
  }

  void updateReturnOperation(mlir::Operation* op, mlir::Operation* returnOperation, std::vector<mlir::Value>& measurementValues, mlir::PatternRewriter& rewriter) const {
    const auto cloned = rewriter.clone(*returnOperation);
    for(size_t i = 0; i < measurementValues.size(); i++) {
      cloned->setOperand(i + 1, measurementValues[i]);
    }
    rewriter.replaceOp(returnOperation, cloned);
  }

  void rewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    std::size_t numQubits = circuit.getNqubits();
    std::vector<mlir::Value> measurementValues(numQubits);

    auto newAlloc = rewriter.create<AllocOp>(
        op.getLoc(), QubitRegisterType::get(rewriter.getContext()), nullptr,
        rewriter.getIntegerAttr(rewriter.getI64Type(), numQubits));
    newAlloc->setAttr("mqt_core", rewriter.getUnitAttr());

    std::vector<mlir::Value> currentQubitVariables(numQubits);
    for (size_t i = 0; i < numQubits; i++) {
      currentQubitVariables[i] =
          createRegisterAccess(newAlloc, i, rewriter).getOutQubit();
    }

    for (const auto& o : circuit) {
      std::vector<int> controlQubitIndices;
      std::vector<mlir::Value> controlQubitsVector;
      std::vector<mlir::Value> controlValuesVector;
      for (const auto& control : o->getControls()) {
        controlQubitIndices.push_back(control.qubit);
        controlQubitsVector.push_back(currentQubitVariables[control.qubit]);

        mlir::Type boolType = rewriter.getI1Type();
        mlir::TypedAttr constAttr = mlir::IntegerAttr::get(
            boolType, control.type == qc::Control::Type::Pos);
        auto constant = rewriter.create<mlir::arith::ConstantOp>(
            op->getLoc(), boolType, constAttr);

        controlValuesVector.push_back(constant.getResult());
      }
      mlir::ValueRange controlQubits(controlQubitsVector);
      mlir::ValueRange controlValues(controlValuesVector);

      if (o->getType() == qc::OpType::X) {
        XOp newXOp = createXOp(op->getLoc(), currentQubitVariables[o->getTargets()[0]],
                               controlQubits,
                               rewriter);
        currentQubitVariables[o->getTargets()[0]] = newXOp.getOutQubits()[0];
        for (size_t i = 0; i < o->getControls().size(); i++) {
          currentQubitVariables[controlQubitIndices[i]] =
              newXOp.getOutQubits()[i + 1];
        }
      } else if(o->getType() == qc::OpType::Measure) {
        MeasureOp newMeasureOp = createMeasureOp(op->getLoc(), currentQubitVariables[o->getTargets()[0]], rewriter);
        currentQubitVariables[o->getTargets()[0]] = newMeasureOp.getOutQubits()[0];
        measurementValues[o->getTargets()[0]] = newMeasureOp.getOutBits()[0];
      } else {
        llvm::outs() << "ERROR: Unsupported operation type " << o->getType() << "\n";
      }
    }

    auto returnOperation = *op->getUsers().begin();
    updateReturnOperation(op, returnOperation, measurementValues, rewriter);
      

    mlir::ValueRange finalValues(currentQubitVariables);
    rewriter.replaceOp(op, newAlloc);
  }
};

void populateFromQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                            qc::QuantumComputation& circuit) {
  patterns.add<FromQuantumComputationPattern>(patterns.getContext(), circuit);
}

} // namespace mqt::ir::opt
