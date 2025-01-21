//===-- Passes.td - Arith pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MQTO_PASSES
#define MQTO_PASSES

include "mlir/Pass/PassBase.td"

def MQTCoreRoundTrip : Pass<"mqt-core-round-trip", "mlir::ModuleOp"> {
  let summary = "This pass performs a round trip to MQT Core's QuantumComputation and back";
  let description = [{
    This pass starts by extracting all quantum-computation-relevant operations and then
    translates them into an MQT Core quantum computation, deleting all quantum operations
    in the process. Then, the quantum computation is read in a second pattern and used
    to generate new MLIR code from it.
  }];
}

#endif // MQTO_PASSES
