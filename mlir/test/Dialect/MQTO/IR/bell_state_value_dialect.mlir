// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() -> (i1, i1) {

        %r0 = "mqto.allocQubitRegister" () {"size" = 2 : i32} : () -> !mqto.QubitRegister
        %r0_1, %q0 = "mqto.extractQubit" (%r0) {"idx" = 0 : i32} : (!mqto.QubitRegister) -> (!mqto.QubitRegister, !mqto.Qubit)

        %k = arith.constant 2 : i32
        %r1 = "mqto.allocQubitRegister" (%k) : (i32) -> !mqto.QubitRegister
        %i = arith.constant 0 : i32
        %r1_1, %q1 = "mqto.extractQubit" (%r1, %i) : (!mqto.QubitRegister, i32) -> (!mqto.QubitRegister, !mqto.Qubit)

        %q0_1 = "mqto.operation" (%q0) {"gate_name" = "Hadamard" : string} : (!mqto.Qubit) -> !mqto.Qubit
        %q0_2, %q1_1 = "mqto.operation" (%q0_1, %q1) {"gate_name" = "CX" : string} : (!mqto.Qubit, !mqto.Qubit) -> (!mqto.Qubit, !mqto.Qubit)

        %q0_3, %c0 = "mqto.measure" (%q0_2) : (!mqto.Qubit) -> (!mqto.Qubit, i1)
        %q1_2, %c1 = "mqto.measure" (%q1_1) : (!mqto.Qubit) -> (!mqto.Qubit, i1)

        %q0_4, %q1_3 = "mqto.reset" (%q0_3, %q1_2) : (!mqto.Qubit, !mqto.Qubit) -> (!mqto.Qubit, !mqto.Qubit)

        %r0_2 = "mqto.insertQubit" (%r0_1, %q0_4) {"index" = 0 : i32} : (!mqto.QubitRegister, !mqto.Qubit) -> !mqto.QubitRegister
        %r1_2 = "mqto.insertQubit" (%r1_1, %q1_3) {"index" = 0 : i32} : (!mqto.QubitRegister, !mqto.Qubit) -> !mqto.QubitRegister

        "mqt.deallocQubitRegister" (%r0_2) : (!mqto.QubitRegister) -> ()
        "mqt.deallocQubitRegister" (%r1_2) : (!mqto.QubitRegister) -> ()

        return (%c0, %c1) : (i1, i1)
    }
}
