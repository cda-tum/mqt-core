#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
class RewritePatternSet;
} // namespace mlir

namespace mqt::ir::opt {
#define GEN_PASS_DECL
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

void populateThePassPatterns(mlir::RewritePatternSet& patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"
} // namespace mqt::ir::opt
