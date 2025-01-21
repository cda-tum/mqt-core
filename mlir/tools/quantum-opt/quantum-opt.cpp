#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mqt::ir::opt::registerMQTOptPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  registry.insert<mqt::ir::opt::MQTOptDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
