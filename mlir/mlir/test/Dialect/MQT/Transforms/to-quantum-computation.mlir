// XFAIL: *
// RUN: quantum-opt %s | FileCheck %s

module {
    func.func @bell() {
        %0 = mqt.alloc( 2) : !mqt.reg
        %q0_0 = mqt.extract %0[ 0] : !mqt.reg -> !mqt.bit
        %q1_0 = mqt.extract %0[ 1] : !mqt.reg -> !mqt.bit
        %q0_1 = mqt.custom "Hadamard"() %q0_0 : !mqt.bit
        %q1_1 = mqt.custom "Hadamard"() %q1_0 : !mqt.bit

        %q1_2 = mqt.custom "PauliX"() %q1_1 : !mqt.bit

        %q01_0:2 = mqt.custom "CNOT"() %q0_1, %q1_2 : !mqt.bit, !mqt.bit

        %q1_3 = mqt.custom "PauliZ"() %q01_0#1 : !mqt.bit
        %q0_2 = mqt.custom "PauliY"() %q01_0#0 : !mqt.bit
        return
    }
}
