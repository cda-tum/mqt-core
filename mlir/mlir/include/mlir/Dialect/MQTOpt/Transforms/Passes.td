#ifndef MQTOpt_PASSES
#define MQTOpt_PASSES

include "mlir/Pass/PassBase.td"

def ThePass : Pass<"the-pass", "ModuleOp"> {
  let summary = "That is THE pass for the MQTOpt dialect";
  let description = [{
    This pass does something very important.
    It is the most important pass in the MQTOpt dialect.
  }];
}

#endif // MQTOpt_PASSES
