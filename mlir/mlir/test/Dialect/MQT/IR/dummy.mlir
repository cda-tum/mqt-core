// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() -> !mqt.bit {
        // CHECK: %{{.*}} = mqt.alloc( 1) : !mqt.reg
        %0 = mqt.alloc( 1) : !mqt.reg
        %1 = mqt.extract %0[ 0] : !mqt.reg -> !mqt.bit
        %out_qubits = mqt.custom "PauliZ"() %1 : !mqt.bit
        return %out_qubits : !mqt.bit
    }
}
