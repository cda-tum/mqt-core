// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() -> !mqt.bit {
        // CHECK: %{{.*}} = mqt.alloc( 1) : !mqt.reg
        %0 = mqt.alloc( 2) : !mqt.reg
        %1 = mqt.extract %0[ 0] : !mqt.reg -> !mqt.bit
        %2 = mqt.extract %0[ 0] : !mqt.reg -> !mqt.bit

        %out_qubits = mqt.custom "Hadamard"() %1 : !mqt.bit
        
        %out_qubits_0:2 = mqt.custom "CNOT"() %out_qubits, %2 : !mqt.bit, !mqt.bit
        return %out_qubits : !mqt.bit
    }
}
