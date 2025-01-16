#pragma once

#include <set>

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

namespace mqt {

#define GEN_PASS_DECL
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

void populateThePassPatterns(RewritePatternSet& patterns);
void populateThePassWithHadamardRewritePattern(RewritePatternSet& patterns);
void populateThePassWithCXRewritePattern(RewritePatternSet& patterns);
void populateToQuantumComputationPatterns(RewritePatternSet& patterns,
                             std::set<Operation*>& handledOperations);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

} // namespace mqt
} // namespace mlir
