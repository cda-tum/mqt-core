// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() -> (i1, i1) {
        %0 = "mqto.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqto.QubitRegister
        %out_qureg, %out_qubit = "mqto.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqto.QubitRegister) -> (!mqto.QubitRegister, !mqto.Qubit)
        %c2_i64 = arith.constant 2 : i64
        %1 = "mqto.allocQubitRegister"(%c2_i64) : (i64) -> !mqto.QubitRegister
        %c0_i64 = arith.constant 0 : i64
        %out_qureg_0, %out_qubit_1 = "mqto.extractQubit"(%1, %c0_i64) : (!mqto.QubitRegister, i64) -> (!mqto.QubitRegister, !mqto.Qubit)
        %2 = "mqto.operation"(%out_qubit) <{gate_name = "Hadamard"}> : (!mqto.Qubit) -> !mqto.Qubit
        %3:2 = "mqto.operation"(%2, %out_qubit_1) <{gate_name = "CX"}> : (!mqto.Qubit, !mqto.Qubit) -> (!mqto.Qubit, !mqto.Qubit)
        %out_qubit_2, %out_bit = "mqto.measure"(%3#0) : (!mqto.Qubit) -> (!mqto.Qubit, i1)
        %out_qubit_3, %out_bit_4 = "mqto.measure"(%3#1) : (!mqto.Qubit) -> (!mqto.Qubit, i1)
        %4 = "mqto.insertQubit"(%out_qureg, %out_qubit_2) <{index_attr = 0 : i64}> : (!mqto.QubitRegister, !mqto.Qubit) -> !mqto.QubitRegister
        %5 = "mqto.insertQubit"(%out_qureg_0, %out_qubit_3) <{index_attr = 0 : i64}> : (!mqto.QubitRegister, !mqto.Qubit) -> !mqto.QubitRegister
        "mqto.deallocQubitRegister"(%4) : (!mqto.QubitRegister) -> ()
        "mqto.deallocQubitRegister"(%5) : (!mqto.QubitRegister) -> ()
        return %out_bit, %out_bit_4 : i1, i1
    }
}
