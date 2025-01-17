// XFAIL: *
// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() -> (i1, i1) {
        // Lukas: Any thoughts?
        //%r0 = "mqt.allocQubitRegister" () {"size" = 1 : i32} : () -> !mqt.QubitRegister
        //%q0 = "mqt.extractQubit" (%r0) {"index" = 0 : i32} : (!mqt.QubitRegister) -> !mqt.Qubit

        %i = arith.constant 0 : i32
        %q1 = "mqt.extractQubit" (%i) : (i32) -> !mqt.Qubit
        %q2 = "mqt.extractQubit" {"index" = 1 : i32} : () -> !mqt.Qubit

        "mqt.operation" (%q0) {"gate_name" = "Hadamard" : string} : (!mqt.Qubit) -> ()
        "mqt.operation" (%q0, %q1) {"gate_name" = "CX" : string} : (!mqt.Qubit, !mqt.Qubit) -> ()

        %c0 = "mqt.measure" (%q0) : (!mqt.Qubit) -> i1
        %c1 = "mqt.measure" (%q1) : (!mqt.Qubit) -> i1

        "mqt.reset" (%q0, %q1) : (!mqt.Qubit, !mqt.Qubit) -> ()

        //"mqt.deallocQubitRegister" (%r0) : (!mqt.QubitRegister) -> ()

        return (%c0, %c1) : (i1, i1)
    }
}
