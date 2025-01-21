#include "ir/QuantumComputation.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/OpType.hpp"
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
struct ToQuantumComputationPattern final : mlir::OpRewritePattern<AllocOp> {
  qc::QuantumComputation& circuit;

  explicit ToQuantumComputationPattern(mlir::MLIRContext* context,
                                       qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

  mlir::LogicalResult match(AllocOp op) const override {
    return (op->hasAttr("to_replace") || op->hasAttr("mqt_core")) ? mlir::failure()
                                                                  : mlir::success();
  }

  size_t findQubitIndex(mlir::Value input, std::vector<mlir::Value>& currentQubitVariables) const {
    size_t arrayIndex = 0;
    if (auto opResult = mlir::dyn_cast<mlir::OpResult>(input)) {
      arrayIndex = opResult.getResultNumber();
    } else {
      throw std::runtime_error("Operand is not an operation result. This should never happen!");
    }
    for(size_t i = 0; i < currentQubitVariables.size(); i++) {
      size_t qubitArrayIndex = 0;
      if (auto opResult = mlir::dyn_cast<mlir::OpResult>(currentQubitVariables[i])) {
        qubitArrayIndex = opResult.getResultNumber();
      } else {
        throw std::runtime_error("Qubit is not an operation result. This should never happen!");
      }

      if(currentQubitVariables[i] == input && arrayIndex == qubitArrayIndex) {
        return i;
      }
    }

    throw std::runtime_error("Qubit was not found in list of previously defined qubits");
  }

  void handleMeasureOp(MeasureOp op, std::vector<mlir::Value>& currentQubitVariables) const {
    const auto ins = op.getInQubits();
    const auto outs = op.getOutQubits();

    std::vector<size_t> insIndices(ins.size());
    std::transform(
        ins.begin(), ins.end(), insIndices.begin(),
        [&currentQubitVariables, this](mlir::Value val) {
          return findQubitIndex(val, currentQubitVariables);
        });

    for (size_t i = 0; i < insIndices.size(); i++) {
      currentQubitVariables[insIndices[i]] = outs[i];
      circuit.measure(insIndices[i], insIndices[i]);
    }
  }

  void handleXOp(XOp op,
                      std::vector<mlir::Value>& currentQubitVariables) const {
    const auto in = op.getInQubit();
    const auto ctrlIns = op.getCtrlQubits();
    const auto outs = op.getOutQubits();


    std::vector<size_t> insIndices(1 + ctrlIns.size());

    const auto found = std::find(currentQubitVariables.begin(),
                                       currentQubitVariables.end(), in);
    if (currentQubitVariables.end() == found) {
      throw std::runtime_error("Qubit not found!");
    }
    insIndices[0] = findQubitIndex(in, currentQubitVariables);


    std::transform(
        ctrlIns.begin(), ctrlIns.end(), (insIndices.begin() + 1),
        [&currentQubitVariables, this](mlir::Value val) {
          return findQubitIndex(val, currentQubitVariables);
        });

    for (size_t i = 0; i < insIndices.size(); i++) {
      currentQubitVariables[insIndices[i]] = outs[i];
    }
    const auto index0 = insIndices[0];

    insIndices.erase(insIndices.begin());
    auto operation = qc::StandardOperation(qc::Controls{insIndices.cbegin(), insIndices.cend()}, index0, qc::OpType::X);
    circuit.push_back(operation);
  }

  void deleteRecursively(mlir::Operation* op, mlir::PatternRewriter& rewriter) const {
    if (op->getName().getStringRef() == "mqtopt.allocQubitRegister") {
      return; // Do not delete extract operations.
    }
    if (!op->getUsers().empty()) {
      return; // Do not delete operations with users.
    }
    rewriter.eraseOp(op);

    for (auto operand : op->getOperands()) {
      if (auto* defOp = operand.getDefiningOp()) {
        deleteRecursively(defOp, rewriter);
      }
    }
  }

  void updateMQTOptInputs(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Value qureg, size_t measureCount) const {
    size_t i = 0;
    const auto cloned = rewriter.clone(*op);
    rewriter.setInsertionPoint(cloned);
    for (auto operand : op->getOperands()) {
      i++;
      const auto type = operand.getType();
      if (mlir::isa<QubitType>(type)) {
        throw std::runtime_error("Interleaving of qubits with non MQTOpt-operations not supported by round-trip pass!");
      }
      if (mlir::isa<QubitRegisterType>(type)) {
        cloned->setOperand(i - 1, qureg);
      }
      if (mlir::isa<mlir::IntegerType>(type)) {
        auto newInput = rewriter.create<mlir::arith::ConstantOp>(
            op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        cloned->setOperand(i - 1, newInput.getResult());
      }
    }

    if(i != measureCount + 1) {
      throw std::runtime_error("Measure count does not match number of return operands!");
    }
    rewriter.replaceOp(op, cloned);
  }

  void rewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::outs() << "\n-----------------GENERAL----------------\n";

    if (!op.getSizeAttr().has_value()) {
      throw std::runtime_error("Qubit allocation only supported with attr size!");
    } else {
      llvm::outs() << "Allocating " << *op.getSizeAttr() << " qubits\n";
    }

    auto newAlloc = rewriter.create<AllocOp>(
        op.getLoc(), QubitRegisterType::get(rewriter.getContext()), nullptr,
        rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    newAlloc->setAttr("to_replace", rewriter.getUnitAttr());
    size_t measureCount = 0;

    const std::size_t numQubits = *op.getSizeAttr();

    std::vector<mlir::Value> currentQubitVariables(numQubits);
    std::vector<mlir::Operation*> toVisit{op};

    std::string regName;
    llvm::raw_string_ostream os(regName);
    op.getResult().print(os);

    circuit.addQubitRegister(numQubits, regName);
    std::set<mlir::Operation*> visited{};
    std::set<mlir::Operation*> visitedQuantum{};

    while (!toVisit.empty()) {
      mlir::Operation* current = *toVisit.begin();
      llvm::outs() << "Visiting " << current->getName() << "\n";
      toVisit.erase(toVisit.begin());
      if (visited.find(current) != visited.end()) {
        continue;
      }
      visited.insert(current);

      if (current->getName().getStringRef() == "mqtopt.x") {
        XOp xOp = mlir::dyn_cast<XOp>(current);
        handleXOp(xOp, currentQubitVariables);
      } else if (current->getName().getStringRef() == "mqtopt.extractQubit") {
        ExtractOp extractOp = mlir::dyn_cast<ExtractOp>(current);

        if (!extractOp.getIndexAttr().has_value()) {
          throw std::runtime_error("Qubit extraction only supported with attr index!");
        } else {
          currentQubitVariables[*extractOp.getIndexAttr()] =
              extractOp.getOutQubit();
        }
      } else if (current->getName().getStringRef() == "mqtopt.allocQubitRegister") {
        // Do nothing for now TODO
      } else if (current->getName().getStringRef() == "mqtopt.measure") {
        measureCount++;
        MeasureOp measureOp = mlir::dyn_cast<MeasureOp>(current);
        handleMeasureOp(measureOp, currentQubitVariables);
      }
      else {
        llvm::outs() << "Unknown operation: " << current->getName() << "\n";
        continue;
        // TODO: y, z, h
      }
      visitedQuantum.insert(current);

      for (mlir::Operation* user : current->getUsers()) {
        if (visited.find(user) != visited.end() ||
            toVisit.end() != std::find(toVisit.begin(), toVisit.end(), user)) {
          continue;
        }
        toVisit.push_back(user);
      }
    }

    llvm::outs() << "----------------------------------------\n\n";

    llvm::outs() << "-------------------QC-------------------\n";
    std::stringstream ss{};
    circuit.print(ss);
    const auto circuitString = ss.str();
    llvm::outs() << circuitString << "\n";
    llvm::outs() << "----------------------------------------\n\n";

    for (auto* operation : visited) {
      if (operation->getDialect()->getNamespace() != "mqtopt") {
        updateMQTOptInputs(operation, rewriter, newAlloc.getQureg(), measureCount);
      }
    }
    for (auto* operation : visitedQuantum) {
      if (operation->getUsers().empty()) {
        deleteRecursively(operation, rewriter);
      }
    }
    
    rewriter.replaceOp(op, newAlloc);

    llvm::outs() << "--------------END-----------------------\n\n";
  }
};

void populateToQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                          qc::QuantumComputation& circuit) {
  patterns.add<ToQuantumComputationPattern>(patterns.getContext(), circuit);
}

} // namespace mqt::ir::opt
