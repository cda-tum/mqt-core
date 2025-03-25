// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --quantum-to-mqtopt | FileCheck %s

// CHECK-LABEL: func @bar()
func.func @bar() {
  // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
  %c0_i64 = arith.constant 0 : i64
  %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

  // CHECK: %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  // CHECK: %[[Q1:.*]] = "mqtopt.extractQubit"(%[[Q0]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  // CHECK: %[[Q2:.*]] = "mqtopt.extractQubit"(%[[Q1]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg, %out_qubit     = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qureg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qureg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

  // CHECK: %[[H:.*]] = mqtopt.H() %[[Q0]] : !mqtopt.Qubit
  %1 = mqtopt.H() %out_qubit : !mqtopt.Qubit

  // CHECK: %[[CX1:.*]]:2 = mqtopt.x() %[[Q1]] ctrl %[[H]] : !mqtopt.Qubit, !mqtopt.Qubit
  %2:2 = mqtopt.x() %out_qubit_1 ctrl %1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[CX2:.*]]:2 = mqtopt.x() %[[Q2]] ctrl %[[CX1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
  %3:2 = mqtopt.x() %out_qubit_3 ctrl %2#1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[R0:.*]] = "mqtopt.insertQubit"(%[[QREG]], %[[CX1]]#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[R0]], %[[CX2]]#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[CX2]]#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %4 = "mqtopt.insertQubit"(%out_qureg_2, %2#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %5 = "mqtopt.insertQubit"(%4, %3#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %6 = "mqtopt.insertQubit"(%5, %3#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

  // CHECK: "mqtopt.deallocQubitRegister"(%[[R2]]) : (!mqtopt.QubitRegister) -> ()
  "mqtopt.deallocQubitRegister"(%6) : (!mqtopt.QubitRegister) -> ()

  return
}
