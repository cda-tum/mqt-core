// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// This test checks the `quantum-sink` pass
// The X-Gate applied in the main block can be pushed down into the two child blocks
// This test does not use block parameters to pass the state of the qubit to the blocks, but the
// `quantum-sink` pass should still be able to work with them.
// The result of this pass should be the same as for the `branch-opt.mlir` test.

// RUN: quantum-opt %s --quantum-sink | FileCheck %s

module {
  func.func @main() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK-NOT: %[[ANY:.*]] = mqtopt.x()
    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit

    // CHECK: %[[Q0_1:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_0]])
    %q0_1, %c0_0 = "mqtopt.measure"(%q0_0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: cf.cond_br %[[C0_0]], ^[[THEN:.*]], ^[[ELSE:.*]]
    cf.cond_br %c0_0, ^then, ^else

  // CHECK: ^[[THEN]]:
  ^then:
    // CHECK: %[[Q1_1T:.*]] = mqtopt.x() %[[Q1_0]]
    // CHECK: cf.br ^[[CONTINUE:.*]](%[[Q1_1T]] : !mqtopt.Qubit)
    cf.br ^continue(%q1_1 : !mqtopt.Qubit)

  // CHECK: ^[[ELSE]]:
  ^else:
    // CHECK: %[[Q1_1E:.*]] = mqtopt.x() %[[Q1_0]]
    // CHECK: %[[Q1_2E:.*]] = mqtopt.x() %[[Q1_1E]]
    %q1_3 = mqtopt.x() %q1_1 : !mqtopt.Qubit
    // CHECK: cf.br ^[[CONTINUE]](%[[Q1_2E]] : !mqtopt.Qubit)
    cf.br ^continue(%q1_3 : !mqtopt.Qubit)

  ^continue(%q1_4 : !mqtopt.Qubit):
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return
  }
}
