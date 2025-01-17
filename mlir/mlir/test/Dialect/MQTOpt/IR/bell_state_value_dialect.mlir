// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() -> (i1, i1) {
        %0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c2_i64 = arith.constant 2 : i64
        %1 = "mqtopt.allocQubitRegister"(%c2_i64) : (i64) -> !mqtopt.QubitRegister
        %c0_i64 = arith.constant 0 : i64
        %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%1, %c0_i64) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %2 = "mqtopt.operation"(%out_qubit) : (!mqtopt.Qubit) -> !mqtopt.Qubit
        %3:2 = "mqtopt.operation"(%2, %out_qubit_1) <{op_type = "CX"}> : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit)
        %out_qubit_2, %out_bit = "mqtopt.measure"(%3#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %out_qubit_3, %out_bit_4 = "mqtopt.measure"(%3#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %4 = "mqtopt.insertQubit"(%out_qureg, %out_qubit_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %5 = "mqtopt.insertQubit"(%out_qureg_0, %out_qubit_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%4) : (!mqtopt.QubitRegister) -> ()
        "mqtopt.deallocQubitRegister"(%5) : (!mqtopt.QubitRegister) -> ()
        return %out_bit, %out_bit_4 : i1, i1
    }
}
