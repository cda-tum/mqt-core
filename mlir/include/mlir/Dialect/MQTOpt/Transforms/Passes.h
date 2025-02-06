#pragma once

#include "ir/QuantumComputation.hpp"

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

} // namespace mlir

namespace mqt::ir::opt {

#define GEN_PASS_DECL
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

void populateCancelSelfInversePatterns(mlir::RewritePatternSet& patterns);
void populateBranchOperationPushdownPatterns(mlir::RewritePatternSet& patterns);
void populateToQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                          qc::QuantumComputation& circuit);
void populateFromQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                            qc::QuantumComputation& circuit);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"
} // namespace mqt::ir::opt
