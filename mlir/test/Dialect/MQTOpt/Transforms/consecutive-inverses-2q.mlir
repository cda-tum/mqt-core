// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// This test checks if two-qubit consecutive self-inverses are canceled correctly.
// For this, the operations must involve exactly the same qubits.

// RUN: quantum-opt %s --cancel-consecutive-inverses | FileCheck %s

module {
  func.func @main() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // --------------------- Check for operations that should not be canceled -----------------------------------
    // CHECK: %[[Q12_1:.*]]:2 = mqtopt.x() %[[Q1_0]] ctrl %[[Q2_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q12_2:.*]]:2 = mqtopt.x() %[[Q12_1]]#1 ctrl %[[Q12_1]]#0 : !mqtopt.Qubit, !mqtopt.Qubit

    // --------------------- Check for operations that should be canceled -----------------------------------
    // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit

    %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q1_3, %q2_1 = mqtopt.x() %q1_2 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q2_2, %q1_4 = mqtopt.x() %q2_1 ctrl %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q0_0]])  <{index_attr = 0 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_5:.*]] = "mqtopt.insertQubit"(%[[Reg_4]], %[[Q12_2]]#1)  <{index_attr = 1 : i64}>
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_6:.*]] = "mqtopt.insertQubit"(%[[Reg_5]], %[[Q12_2]]#0)  <{index_attr = 2 : i64}>
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_6]])
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
    return
  }
}
