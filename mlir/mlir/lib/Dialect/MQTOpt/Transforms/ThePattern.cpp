#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {
/// Multistep rewrite using "match" and "rewrite". This allows for separating
/// the concerns of matching and rewriting.
struct ThePattern final : mlir::OpInterfaceRewritePattern<UnitaryInterface> {
  explicit ThePattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult match(UnitaryInterface op) const override {
    return mlir::failure();
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {}
};

void populateThePassPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<ThePattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
