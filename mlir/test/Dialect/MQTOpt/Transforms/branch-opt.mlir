module {
  func.func @main() {
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.y() %q1_1 : !mqtopt.Qubit
    %q1_3 = mqtopt.z() %q1_2 : !mqtopt.Qubit
    %q1_4 = mqtopt.x() %q1_3 : !mqtopt.Qubit

    %q0_1, %c0_0 = "mqtopt.measure"(%q0_0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    cf.cond_br %c0_0, ^then, ^else

  ^then:
    cf.br ^continue

  ^else:
    %q1_5 = mqtopt.x() %q1_4 : !mqtopt.Qubit
    %q1_6 = mqtopt.z() %q1_5 : !mqtopt.Qubit
    %q1_7 = mqtopt.y() %q1_6 : !mqtopt.Qubit
    %q1_8 = mqtopt.x() %q1_7 : !mqtopt.Qubit
    cf.br ^continue

  ^continue:
    return
  }
}
