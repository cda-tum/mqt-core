module {
    func.func @bell() {
        %0 = mqt.alloc( 2) : !mqt.reg
        %q1_0 = mqt.extract %0[ 0] : !mqt.reg -> !mqt.bit
        %q2_0 = mqt.extract %0[ 1] : !mqt.reg -> !mqt.bit
        %q1_1 = mqt.custom "Hadamard"() %q1_0 : !mqt.bit
        %q2_1 = mqt.custom "Hadamard"() %q2_0 : !mqt.bit

        %q2_2 = mqt.custom "PauliX"() %q2_1 : !mqt.bit
        return
    }
}
