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
        // CHECK: %{{.*}} = quantum.alloc( 1) : !quantum.reg
        %r0 = quantum.alloc( 3) : !quantum.reg
        %phi0 = arith.constant 3.000000e-01 : f64

        %q0 = quantum.extract %r0[ 0] : !quantum.reg -> !quantum.bit
        %q1 = quantum.extract %r0[ 1] : !quantum.reg -> !quantum.bit
        %q2 = quantum.extract %r0[ 2] : !quantum.reg -> !quantum.bit

        // Apply single-qubit gates to %q0
        %out_h = quantum.custom "Hadamard"() %q0 : !quantum.bit
        %out_x = quantum.custom "PauliX"() %out_h : !quantum.bit
        %out_y = quantum.custom "PauliY"() %out_x : !quantum.bit
        %out_z = quantum.custom "PauliZ"() %out_y : !quantum.bit

        // Apply multi-qubit gates
        %cnot:2 = quantum.custom "CNOT"() %q1, %q0 : !quantum.bit, !quantum.bit
        %cy:2 = quantum.custom "CY"() %cnot#1, %cnot#0 : !quantum.bit, !quantum.bit
        %cz:2 = quantum.custom "CZ"() %cy#1, %cy#0 : !quantum.bit, !quantum.bit
        %swap:2 = quantum.custom "SWAP"() %cz#1, %cz#0 : !quantum.bit, !quantum.bit
        %toffoli:3 = quantum.custom "Toffoli"() %q2, %swap#1, %swap#0 : !quantum.bit, !quantum.bit, !quantum.bit

        // Apply (multi-qubit) rotation gates
        %rx = quantum.custom "RX"(%phi0) %toffoli#0 : !quantum.bit
        %ry = quantum.custom "RY"(%phi0) %rx : !quantum.bit
        %rz = quantum.custom "RZ"(%phi0) %ry : !quantum.bit
        %phaseShift = quantum.custom "PhaseShift"(%phi0) %rz : !quantum.bit
        %crx:2 = quantum.custom "CRX"(%phi0) %toffoli#1, %phaseShift : !quantum.bit, !quantum.bit
        %cry:2 = quantum.custom "CRY"(%phi0) %crx#1, %crx#0 : !quantum.bit, !quantum.bit
        %crz:2 = quantum.custom "CRY"(%phi0) %cry#1, %cry#0 : !quantum.bit, !quantum.bit
        %cps:2 = quantum.custom "ControlledPhaseShift"(%phi0) %crz#1, %crz#0 : !quantum.bit, !quantum.bit

        // Rebuild register using insertion of modified qubits
        %r1_2 = quantum.insert %r0[2], %toffoli#2 : !quantum.reg, !quantum.bit
        %r1_1 = quantum.insert %r1_2[1], %cps#1 : !quantum.reg, !quantum.bit
        %r1_0 = quantum.insert %r1_1[0], %cps#0 : !quantum.reg, !quantum.bit

        quantum.dealloc %r1_0 : !quantum.reg

        return
    }
}
