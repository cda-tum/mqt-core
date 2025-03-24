// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// XFAIL: *
// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %c0_i64 = arith.constant 0 : i64

        %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

        %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qureg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %out_qureg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qureg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %1 = mqtopt.H() %out_qubit : !mqtopt.Qubit

        %2:2 = mqtopt.x() %1, %out_qubit_1 : !mqtopt.Qubit, !mqtopt.Qubit
        %3:2 = mqtopt.x() %2#1, %out_qubit_3 : !mqtopt.Qubit, !mqtopt.Qubit

        %4 = "mqtopt.insertQubit"(%out_qureg_2, %2#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %5 = "mqtopt.insertQubit"(%4, %3#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %6 = "mqtopt.insertQubit"(%5, %3#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        "mqtopt.deallocQubitRegister"(%6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}
