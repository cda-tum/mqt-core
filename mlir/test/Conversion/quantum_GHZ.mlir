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

        %0 = quantum.alloc( 3) : !quantum.reg

        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

        %out_h = quantum.custom "Hadamard"() %1 : !quantum.bit

        %out_qubits:2 = quantum.custom "CNOT"() %out_h, %2 : !quantum.bit, !quantum.bit
        %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#1, %3 : !quantum.bit, !quantum.bit

        %4 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
        %5 = quantum.insert %4[ 1], %out_qubits_0#0 : !quantum.reg, !quantum.bit
        %6 = quantum.insert %5[ 2], %out_qubits_0#1 : !quantum.reg, !quantum.bit

        quantum.dealloc %6 : !quantum.reg
        return
    }
}
