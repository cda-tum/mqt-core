#include "DDpackage.h"
#include <iostream>
#include <cmath>


namespace dd = dd_package;

dd::DDedge BellCicuit1() {
    /***** define Hadamard gate acting on q0 *****/

    // set control/target:
    //    -1 don't care
    //    0 neg. control
    //    1 pos. control
    //    2 target

    int line[2] = {-1,2};
    dd::DDedge h_gate = DDmvlgate(dd::Hm, 2, line);

    /***** define cx gate with control q0 and target q1*****/
    line[0] = 2;
    line[1] = 1;

    dd::DDedge cx_gate = DDmvlgate(dd::Nm, 2, line);

    //Multiply matrices to get functionality of circuit
    return dd::DDmultiply(cx_gate, h_gate);
}

dd::DDedge BellCicuit2() {
    /***** define Hadamard gate acting on q1 *****/
    int line[2] = {2,-1};
    dd::DDedge h_gate_q1 = DDmvlgate(dd::Hm, 2, line);

    line[0] = -1;
    line[1] = 2;
    dd::DDedge h_gate_q0 = DDmvlgate(dd::Hm, 2, line);

    /***** define cx gate with control q0 and target q1*****/
    line[0] = 1;
    line[1] = 2;
    dd::DDedge cx_gate = DDmvlgate(dd::Nm, 2, line);

    //Multiply matrices to get functionality of circuit
    return dd::DDmultiply(dd::DDmultiply(h_gate_q1, h_gate_q0), dd::DDmultiply(cx_gate, h_gate_q1));
}

int main() {

    //Initialize package
    dd::DDinit(false);

    // create Bell circuit 1
    dd::DDedge bell_circuit1 = BellCicuit1();

    // create Bell circuit 2
    dd::DDedge bell_circuit2 = BellCicuit2();

    /***** Equivalence checking *****/
    if(bell_circuit1.p == bell_circuit2.p && bell_circuit1.w == bell_circuit2.w) {
        std::cout << "Circuits are equal!" << std::endl;
    } else {
        std::cout << "Circuits are not equal!" << std::endl;
    }


    /***** Simulation *****/
    //Generate vector in basis state |00>
    dd::DDedge zero_state = dd::DDzeroState(2);

    //Simulate the bell_circuit with initial state |00>
    dd::DDedge bell_state = dd::DDmultiply(bell_circuit1, zero_state);
    dd::DDedge bell_state2 = dd::DDmultiply(bell_circuit2, zero_state);


    dd::DDdotExportVector(bell_state, "bell_state.dot");

    //print result
    dd::DDprintVector(bell_state);

    std::cout << "Bell states have a fidelity of " << dd::DDfidelity(bell_state, bell_state2) << "\n";
    std::cout << "Bell state and zero state have a fidelity of " << dd::DDfidelity(bell_state, zero_state) << "\n";


    /***** Custom gates *****/
    dd::DD_matrix m;
    m[0][0] = dd::Cmake(std::sqrt(1/2.0L), 0);
    m[0][1] = dd::Cmake(std::sqrt(1/2.0L), 0);
    m[1][0] = dd::Cmake(std::sqrt(1/2.0L), 0);
    m[1][1] = dd::Cmake(-std::sqrt(1/2.0L), 0);

    int line[1] = {2}; // target on first line

    const dd::complex_value one{1,0}, zero{0,0};
    dd::DD_matrix X{{{zero}, {one}}, {{one}, {zero}}};

    dd::DDedge Xgate = DDmvlgate(X, 1, line);
    dd::DDedge id = DDmultiply(Xgate, Xgate);
    dd::DDdotExportMatrix(id, "id.dot");

    dd::DDedge my_very_own_gate = dd::DDmvlgate(m, 1, line);

    /***** print DDs as SVG file *****/
    dd::DDdotExportMatrix(bell_circuit1, "bell_circuit1.dot");
    dd::DDdotExportMatrix(bell_circuit2, "bell_circuit2.dot");

    dd::DDdotExportMatrix(my_very_own_gate, "my_very_own_gate.dot");

    dd::DDstatistics();
}
