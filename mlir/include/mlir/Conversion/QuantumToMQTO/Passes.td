//===-- Passes.td - Arith pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MQT_PASSES
#define MQT_PASSES

include "mlir/Pass/PassBase.td"

def QuantumToMQTOptPass : Pass<"the-pass", "ModuleOp"> {
  let summary = "That is THE pass for the MQT dialect";
  let description = [{
    This pass does something very important.
    It is the most important pass in the MQT dialect.
  }];
}

def ToQuantumComputation : Pass<"to-quantum-computation", "ModuleOp"> {
  let summary = "This pass creates a quantum computation from a given program in the MQT-Opt dialect";
  let description = [{
    This pass starts by extracting all quantum-computation-relevant operations and then
    translates them into an MQT Core quantum computation.
  }];
}

#endif // MQT_PASSES
