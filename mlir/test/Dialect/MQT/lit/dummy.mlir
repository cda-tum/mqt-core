// RUN: quantum-opt %s | quantum-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        // CHECK: %{{.*}} = mqt.alloc( 1) : !mqt.reg
        %0 = mqt.alloc( 1) : !mqt.reg
        %1 = mqt.extract %0[ 0] : !mqt.reg -> !mqt.bit
        %out_qubits = mqt.custom "Hadamard"() %1 : !mqt.bit
        return
    }
}
