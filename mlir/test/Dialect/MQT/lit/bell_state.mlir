// RUN: quantum-opt %s | quantum-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() {

        %r0 = "mqt.allocQubitRegister" () {"size" = 2 : i32} : () -> !mqt.QubitRegister
        
        %r0_1, %q0 = "mqt.extractQubit" (%r0) {"idx" = 0 : i32} : (!mqt.QubitRegister) -> (!mqt.QubitRegister, !mqt.Qubit)

        %q0_1 = "mqt.operation" (%q0) {"gate_name" = "Hadamard" : string} : (!mqt.Qubit) -> !mqt.Qubit


        %k = arith.constant 2 : i32
        %r1 = "mqt.allocQubitRegister" (%k) : (i32) -> !mqt.QubitRegister
        
        %i = arith.constant 0 : i32
        %r1_1, %q1 = "mqt.extractQubit" (%r1, %i) : (!mqt.QubitRegister, i32) -> (!mqt.QubitRegister, !mqt.Qubit)

 
        %q0_2, %q1_1 = "mqt.operation" (%q0_1, %q1) {"gate_name" = "CX" : string} : (!mqt.Qubit, !mqt.Qubit) -> (!mqt.Qubit, !mqt.Qubit)

        %q0_3, %c1 = "mqt.measure" (%q0_2) : (!mqt.Qubit) -> (!mqt.Qubit, i1)
        %q1_2, %c2 = "mqt.measure" (%q1_1) : (!mqt.Qubit) -> (!mqt.Qubit, i1)

        %q0_4, %q1_3 = "mqt.reset" (%q0_3, %q1_2) : (!mqt.Qubit, !mqt.Qubit) -> (!mqt.Qubit, !mqt.Qubit)

        %r0_2 = "mqt.insertQubit" (%r0_1, %q0_4) {"index" = 0 : i32} : (!mqt.QubitRegister, !mqt.Qubit) -> !mqt.QubitRegister

        %r1_2 = "mqt.insertQubit" (%r1_1, %q1_3) {"index" = 0 : i32} : (!mqt.QubitRegister, !mqt.Qubit) -> !mqt.QubitRegister

        "mqt.deallocQubitRegister" (%r0_2) : (!mqt.QubitRegister) -> ()
        "mqt.deallocQubitRegister" (%r1_2) : (!mqt.QubitRegister) -> ()

        return
    }
}
