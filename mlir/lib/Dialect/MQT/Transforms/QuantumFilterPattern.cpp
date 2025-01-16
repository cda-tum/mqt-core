#include "mlir/Dialect/MQT/IR/MQTOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "ir/QuantumComputation.hpp"

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <set>

namespace mlir::mqt {
/// Analysis pattern that filters out all quantum operations from a given program.
struct QuantumFilterPattern final : OpRewritePattern<AllocOp> {
  std::set<Operation*>& handledOperations;

  explicit QuantumFilterPattern(MLIRContext* context, std::set<Operation*>& handled)
      : OpRewritePattern(context), handledOperations(handled) {}

  LogicalResult match(AllocOp op) const override {
    if (handledOperations.find(op) == handledOperations.end()) {
      return success();
    }
    
    return failure();
  }

  void rewrite(AllocOp op, PatternRewriter& rewriter) const override {
    handledOperations.insert(op);

    llvm::outs() << "\n-----------------GENERAL----------------\n";

    if(!op.getNqubitsAttr().has_value()) {
      llvm::outs() << "ERROR: Qubit allocation only supported with attr size!\n";
      return;
    } else {
      llvm::outs() << "Allocating " << *op.getNqubitsAttr() << " qubits\n";
    }

    const std::size_t numQubits = *op.getNqubitsAttr();

    std::vector<Operation*> individualQubitOps(numQubits);
    qc::QuantumComputation circuit(numQubits);

    for (Operation* user : op->getUsers()) {
      if(user->getName().getStringRef() == "mqt.extract") {
        ExtractOp extractOp = dyn_cast<ExtractOp>(user);

        if(!extractOp.getIdxAttr().has_value()) {
          llvm::outs() << "ERROR: Qubit extraction only supported with attr index!\n";
          return;
        } else {
          individualQubitOps[*extractOp.getIdxAttr()] = user;
        }
      }
    }

    llvm::outs() << "-----------------CIRCUIT----------------\n";

    for(std::size_t i = 0; i < numQubits; i++) {
      llvm::outs() << "q" << i;
      std::set<Operation*> toVisit{individualQubitOps[i]};
      while(!toVisit.empty()) {
        Operation* current = *toVisit.begin();
        toVisit.erase(toVisit.begin());

        for (Operation* user : current->getUsers()) {
          toVisit.insert(user);
          if(user->getName().getStringRef() == "mqt.custom") {
            CustomOp customOp = dyn_cast<CustomOp>(user);
            const auto name = customOp.getGateName();
            llvm::outs() << " -> ";
            if(name == "PauliX") {
              llvm::outs() << "X";
              circuit.x(i);
            } else if(name == "PauliY") {
              llvm::outs() << "Y";
              circuit.y(i);
            } else if(name == "PauliZ") {
              llvm::outs() << "Z";
              circuit.z(i);
            } else if(name == "Hadamard") {
              llvm::outs() << "H";
              circuit.h(i);
            } else {
              llvm::outs() << "unknown";
            }
          } else {
            llvm::outs() << " -> unknown";
          }
        }
      }
      llvm::outs() << "\n";
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

void populateToQuantumComputationPatterns(RewritePatternSet& patterns, std::set<Operation*>& handled) {
  patterns.add<QuantumFilterPattern>(patterns.getContext(), handled);
}

} // namespace mlir::mqt
