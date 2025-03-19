// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bell()
    func.func @bell() -> (!mqtopt.QubitRegister, i1, i1) {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
        %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit

        %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit

        %q1_3 = mqtopt.x() %q1_2 : !mqtopt.Qubit

        %q0_3, %c0_0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_4, %c1_0 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        return %reg_2, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}
