#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <set>

namespace mlir::mqt {
/// Analysis pattern that filters out all quantum operations from a given
/// program and creates a quantum computation from them.
struct FromQuantumComputationPattern final : OpRewritePattern<AllocOp> {
  std::set<Operation*>& handledOperations;
  qc::QuantumComputation& circuit;

  explicit FromQuantumComputationPattern(MLIRContext* context,
                                       std::set<Operation*>& handled, qc::QuantumComputation& qc)
      : OpRewritePattern(context), handledOperations(handled), circuit(qc) {}

  LogicalResult match(AllocOp op) const override {
    return op->hasAttr("to_replace") ? success() : failure();
  }

  ExtractOp createRegisterAccess(AllocOp& reg, size_t index, PatternRewriter& rewriter) const {
    return rewriter.create<ExtractOp>(reg.getLoc(), 
        QubitType::get(rewriter.getContext()), 
        reg.getResult(),
        nullptr,
        rewriter.getI64IntegerAttr(index));
  }

  CustomOp createCustomOp(std::string name, mlir::Location loc, mlir::ValueRange inputs, mlir::ValueRange controlQubits, mlir::ValueRange controlValues, PatternRewriter& rewriter) const {
    return rewriter.create<CustomOp>(loc, 
        name, 
        inputs,
        controlQubits,
        controlValues);
  }

  void rewrite(AllocOp op, PatternRewriter& rewriter) const override {
    handledOperations.insert(op);

    std::size_t numQubits = circuit.getNqubits();

    static std::map<qc::OpType, std::string> opNames = {
        {qc::OpType::H, "Hadamard"},
        {qc::OpType::X, "PauliX"},
        {qc::OpType::Y, "PauliY"},
        {qc::OpType::Z, "PauliZ"}
    };


    auto newAlloc = rewriter.create<AllocOp>(op.getLoc(), QuregType::get(rewriter.getContext()), nullptr, rewriter.getIntegerAttr(rewriter.getI64Type(), numQubits));
    newAlloc->setAttr("mqt_core", rewriter.getUnitAttr());

    std::vector<Value> currentQubitVariables(numQubits);
    for(size_t i = 0; i < numQubits; i++) {
        currentQubitVariables[i] = createRegisterAccess(newAlloc, i, rewriter).getResult();
    }

    for(const auto& o : circuit) {
        std::string opName = "";
        std::vector<int> controlQubitIndices;
        std::vector<Value> controlQubitsVector;
        std::vector<Value> controlValuesVector;
        for(const auto& control : o->getControls()) {
          controlQubitIndices.push_back(control.qubit);
          controlQubitsVector.push_back(currentQubitVariables[control.qubit]);

          mlir::Type boolType = rewriter.getI1Type();
          mlir::TypedAttr constAttr = mlir::IntegerAttr::get(boolType, control.type == qc::Control::Type::Pos);
          auto constant = rewriter.create<mlir::arith::ConstantOp>(op->getLoc(), boolType, constAttr);

          controlValuesVector.push_back(constant.getResult());
          opName += "C";
        }
        mlir::ValueRange controlQubits(controlQubitsVector);
        mlir::ValueRange controlValues(controlValuesVector);
        CustomOp newOp = nullptr;
        switch(o->getType()) {
            case qc::OpType::H:
            case qc::OpType::X:
            case qc::OpType::Y:
            case qc::OpType::Z:
                newOp = createCustomOp(
                    opName + opNames[o->getType()],
                    op->getLoc(),
                    {currentQubitVariables[o->getTargets()[0]]},
                    controlQubits,
                    controlValues,
                    rewriter);//.getOutQubits()[0];
                break;
            default:
                llvm::outs() << "ERROR: Unsupported operation type!\n";
                break;
        }
        for(size_t i = 0; i < o->getTargets().size(); i++) {
            currentQubitVariables[o->getTargets()[i]] = newOp.getOutQubits()[i];
        }
        for(size_t i = 0; i < o->getControls().size(); i++) {
            currentQubitVariables[controlQubitIndices[i]] = newOp.getOutCtrlQubits()[i];
        }
    }

    mlir::ValueRange finalValues(currentQubitVariables);
    rewriter.replaceOp(op, newAlloc);

  }
};

void populateFromQuantumComputationPatterns(RewritePatternSet& patterns,
                                          std::set<Operation*>& handled, qc::QuantumComputation& circuit) {
  patterns.add<FromQuantumComputationPattern>(patterns.getContext(), handled, circuit);
}

} // namespace mlir::mqt
