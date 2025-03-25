// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --mqtopt-to-quantum | FileCheck %s

// CHECK-LABEL: func @bar()
func.func @bar() {
  // CHECK: %[[PHI:.*]] = arith.constant 3.000000e-01 : f64
  %cst = arith.constant 3.000000e-01 : f64

  // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
  %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

  // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit
  %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qureg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qureg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

  // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit
  // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[H]] : !quantum.bit
  // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
  // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
  %1 = mqtopt.H() %out_qubit : !mqtopt.Qubit
  %2 = mqtopt.x() %1 : !mqtopt.Qubit
  %3 = mqtopt.y() %2 : !mqtopt.Qubit
  %4 = mqtopt.z() %3 : !mqtopt.Qubit

  // CHECK: %[[CX:.*]]:2 = quantum.custom "CNOT"() %[[Q1]], %[[Q0]]
  // CHECK: %[[CY:.*]]:2 = quantum.custom "CY"() %[[CX]]#1, %[[CX]]#0
  // CHECK: %[[CZ:.*]]:2 = quantum.custom "CZ"() %[[CY]]#1, %[[CY]]#0
  // CHECK: %[[SWAP:.*]]:2 = quantum.custom "SWAP"() %[[CZ]]#1, %[[CZ]]#0
  // CHECK: %[[TOF:.*]]:3 = quantum.custom "Toffoli"() %[[Q2]], %[[SWAP]]#1, %[[SWAP]]#0
  %5:2 = mqtopt.x() %out_qubit_1, %out_qubit : !mqtopt.Qubit, !mqtopt.Qubit
  %6:2 = mqtopt.y() %5#1, %5#0 : !mqtopt.Qubit, !mqtopt.Qubit
  %7:2 = mqtopt.z() %6#1, %6#0 : !mqtopt.Qubit, !mqtopt.Qubit
  %8:2 = mqtopt.swap() %7#1, %7#0 : !mqtopt.Qubit, !mqtopt.Qubit
  %9:3 = mqtopt.x() %out_qubit_3, %8#1, %8#0 : !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[PHI]]) %[[TOF]]#0
  // CHECK: %[[RY:.*]] = quantum.custom "RY"(%[[PHI]]) %[[RX]]
  // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%[[PHI]]) %[[RY]]
  // CHECK: %[[P:.*]] = quantum.custom "PhaseShift"(%[[PHI]]) %[[RZ]]
  %10 = mqtopt.rx(%cst) %9#0 : !mqtopt.Qubit
  %11 = mqtopt.ry(%cst) %10 : !mqtopt.Qubit
  %12 = mqtopt.rz(%cst) %11 : !mqtopt.Qubit
  %13 = mqtopt.p(%cst) %12 : !mqtopt.Qubit

  // CHECK: %[[CRX:.*]]:2 = quantum.custom "CRX"(%[[PHI]]) %[[TOF]]#1, %[[P]]
  // CHECK: %[[CRY:.*]]:2 = quantum.custom "CRY"(%[[PHI]]) %[[CRX]]#1, %[[CRX]]#0
  // CHECK: %[[CRZ:.*]]:2 = quantum.custom "CRY"(%[[PHI]]) %[[CRY]]#1, %[[CRY]]#0
  // CHECK: %[[CPS:.*]]:2 = quantum.custom "ControlledPhaseShift"(%[[PHI]]) %[[CRZ]]#1, %[[CRZ]]#0
  %14:2 = mqtopt.rx(%cst) %9#1, %13 : !mqtopt.Qubit, !mqtopt.Qubit
  %15:2 = mqtopt.ry(%cst) %14#1, %14#0 : !mqtopt.Qubit, !mqtopt.Qubit
  %16:2 = mqtopt.ry(%cst) %15#1, %15#0 : !mqtopt.Qubit, !mqtopt.Qubit
  %17:2 = mqtopt.p(%cst) %16#1, %16#0 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[R0:.*]] = quantum.insert %[[QREG]][ 2], %[[TOF]]#2
  // CHECK: %[[R1:.*]] = quantum.insert %[[R0]][ 1], %[[CPS]]#1
  // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 0], %[[CPS]]#0
  %18 = "mqtopt.insertQubit"(%out_qureg_2, %9#2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %19 = "mqtopt.insertQubit"(%18, %17#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %20 = "mqtopt.insertQubit"(%19, %17#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

  // CHECK: quantum.dealloc %[[R2]] : !quantum.reg
  "mqtopt.deallocQubitRegister"(%20) : (!mqtopt.QubitRegister) -> ()
  return
}
