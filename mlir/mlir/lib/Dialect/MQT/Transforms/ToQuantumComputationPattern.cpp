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
struct ToQuantumComputationPattern final : OpRewritePattern<AllocOp> {
  std::set<Operation*>& handledOperations;

  explicit ToQuantumComputationPattern(MLIRContext* context,
                                       std::set<Operation*>& handled)
      : OpRewritePattern(context), handledOperations(handled) {}

  LogicalResult match(AllocOp op) const override {
    if (handledOperations.find(op) == handledOperations.end()) {
      return success();
    }

    return failure();
  }

  void handleCustomOp(CustomOp op, qc::QuantumComputation& circuit,
                      std::vector<Value>& currentQubitVariables) const {
    const auto ins = op.getInQubits();
    const auto outs = op.getOutQubits();

    std::vector<size_t> ins_indices(ins.size());
    std::transform(
        ins.begin(), ins.end(), ins_indices.begin(),
        [&currentQubitVariables](Value val) {
          const auto found = std::find(currentQubitVariables.begin(),
                                       currentQubitVariables.end(), val);
          if (currentQubitVariables.end() == found) {
            llvm::outs() << "ERROR: Qubit not found!\n";
            return -1UL;
          }
          return (size_t)std::distance(currentQubitVariables.begin(), found);
        });

    for (size_t i = 0; i < ins.size(); i++) {
      currentQubitVariables[ins_indices[i]] = outs[i];
    }

    const auto name = op.getGateName();
    if (name == "Hadamard") {
      circuit.h(ins_indices[0]);
    }
    if (name == "PauliX") {
      circuit.x(ins_indices[0]);
    }
    if (name == "PauliY") {
      circuit.y(ins_indices[0]);
    }
    if (name == "PauliZ") {
      circuit.z(ins_indices[0]);
    }
    if (name == "CNOT") {
      circuit.cx(ins_indices[0], ins_indices[1]);
    }
  }

  void rewrite(AllocOp op, PatternRewriter& rewriter) const override {
    handledOperations.insert(op);

    llvm::outs() << "\n-----------------GENERAL----------------\n";

    if (!op.getNqubitsAttr().has_value()) {
      llvm::outs()
          << "ERROR: Qubit allocation only supported with attr size!\n";
      return;
    } else {
      llvm::outs() << "Allocating " << *op.getNqubitsAttr() << " qubits\n";
    }

    const std::size_t numQubits = *op.getNqubitsAttr();

    std::vector<Value> currentQubitVariables(numQubits);
    qc::QuantumComputation circuit(numQubits);
    std::vector<Operation*> toVisit{};
    std::set<Operation*> visited{op};

    for (Operation* user : op->getUsers()) {
      if (user->getName().getStringRef() == "mqt.extract") {
        ExtractOp extractOp = dyn_cast<ExtractOp>(user);

        if (!extractOp.getIdxAttr().has_value()) {
          llvm::outs()
              << "ERROR: Qubit extraction only supported with attr index!\n";
          return;
        } else {
          currentQubitVariables[*extractOp.getIdxAttr()] =
              extractOp.getResult();
          toVisit.push_back(extractOp);
        }
      }
    }

    while (!toVisit.empty()) {
      Operation* current = *toVisit.begin();
      toVisit.erase(toVisit.begin());

      if (visited.find(current) != visited.end()) {
        continue;
      }
      visited.insert(current);

      for (Operation* user : current->getUsers()) {
        if (visited.find(user) != visited.end() ||
            toVisit.end() != std::find(toVisit.begin(), toVisit.end(), user)) {
          continue;
        }
        if (user->getName().getStringRef() == "mqt.custom") {
          CustomOp customOp = dyn_cast<CustomOp>(user);
          handleCustomOp(customOp, circuit, currentQubitVariables);
        } else {
          // TODO: Handle measurements
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
  }
};

void populateToQuantumComputationPatterns(RewritePatternSet& patterns,
                                          std::set<Operation*>& handled) {
  patterns.add<ToQuantumComputationPattern>(patterns.getContext(), handled);
}

} // namespace mlir::mqt
