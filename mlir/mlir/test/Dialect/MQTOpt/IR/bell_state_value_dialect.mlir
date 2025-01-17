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
        %out_qubit_2 = mqtopt.x() %out_qubit : !mqtopt.Qubit
        %out_qubit_3, %ctrl_out_qubits = mqtopt.x() %out_qubit_2 ctrl %out_qubit_1 : !mqtopt.Qubit, !mqtopt.Qubit
        %out_qubits, %out_bits = "mqtopt.measure"(%out_qubit_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %out_qubits_4, %out_bits_5 = "mqtopt.measure"(%ctrl_out_qubits) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %2 = "mqtopt.insertQubit"(%out_qureg, %out_qubits) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %3 = "mqtopt.insertQubit"(%out_qureg_0, %out_qubits_4) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%2) : (!mqtopt.QubitRegister) -> ()
        "mqtopt.deallocQubitRegister"(%3) : (!mqtopt.QubitRegister) -> ()
        return %out_bits, %out_bits_5 : i1, i1
    }
}
