#pragma once

#include "Gate.hpp"

#include <string>

namespace qasm3 {
// from
// https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/stdgates.inc
const std::string STDGATES =
    "// OpenQASM 3 standard gate library\n"
    "\n"
    //    "// phase gate\n"
    //    "gate p(lambda) a { ctrl @ gphase(lambda) a; }\n"
    //    "\n"
    "// Pauli gate: bit-flip or NOT gate\n"
    "gate x a { U(pi, 0, pi) a; }\n"
    "// Pauli gate: bit and phase flip\n"
    "gate y a { U(pi, pi/2, pi/2) a; }\n"
    "// Pauli gate: phase flip\n"
    "gate z a { p(pi) a; }\n"
    "\n"
    "// Clifford gate: Hadamard\n"
    "gate h a { U(pi/2, 0, pi) a; }\n"
    //    "// Clifford gate: sqrt(Z) or S gate\n"
    // unsupported gates due to `pow` modifier.
    //    "gate s a { pow(1/2) @ z a; }\n"
    //    "// Clifford gate: inverse of sqrt(Z)\n"
    //    "gate sdg a { inv @ pow(1/2) @ z a; }\n"
    //    "\n"
    //    "// sqrt(S) or T gate\n"
    //    "gate t a { pow(1/2) @ s a; }\n"
    //    "// inverse of sqrt(S)\n"
    //    "gate tdg a { inv @ pow(1/2) @ s a; }\n"
    //    "\n"
    //    "// sqrt(NOT) gate\n"
    //    "gate sx a { pow(1/2) @ x a; }\n"
    // "\n"
    "// Rotation around X-axis\n"
    "gate rx(theta) a { U(theta, -pi/2, pi/2) a; }\n"
    "// rotation around Y-axis\n"
    "gate ry(theta) a { U(theta, 0, 0) a; }\n"
    "// rotation around Z axis\n"
    "gate rz(lambda) a { gphase(-lambda/2); U(0, 0, lambda) a; }\n"
    "\n"
    "// controlled-NOT\n"
    "gate cx c, t { ctrl @ x c, t; }\n"
    "// controlled-Y\n"
    "gate cy a, b { ctrl @ y a, b; }\n"
    "// controlled-Z\n"
    "gate cz a, b { ctrl @ z a, b; }\n"
    "// controlled-phase\n"
    "gate cp(lambda) a, b { ctrl @ p(lambda) a, b; }\n"
    "// controlled-rx\n"
    "gate crx(theta) a, b { ctrl @ rx(theta) a, b; }\n"
    "// controlled-ry\n"
    "gate cry(theta) a, b { ctrl @ ry(theta) a, b; }\n"
    "// controlled-rz\n"
    "gate crz(theta) a, b { ctrl @ rz(theta) a, b; }\n"
    "// controlled-H\n"
    "gate ch a, b { ctrl @ h a, b; }\n"
    //    "\n"
    //    "// swap\n"
    //    "gate swap a, b { cx a, b; cx b, a; cx a, b; }\n"
    "\n"
    "// Toffoli\n"
    "gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }\n"
    "// controlled-swap\n"
    "gate cswap a, b, c { ctrl @ swap a, b, c; }\n"
    "\n"
    "// four parameter controlled-U gate with relative phase\n"
    "gate cu(theta, phi, lambda, gamma) c, t { p(gamma) c; ctrl @ U(theta, "
    "phi, lambda) c, t; }\n"
    "\n"
    "// Gates for OpenQASM 2 backwards compatibility\n"
    "// CNOT\n"
    "gate CX c, t { ctrl @ U(pi, 0, pi) c, t; }\n"
    "// phase gate\n"
    "gate phase(lambda) q { U(0, 0, lambda) q; }\n"
    "// controlled-phase\n"
    "gate cphase(lambda) a, b { ctrl @ phase(lambda) a, b; }\n"
    "// identity or idle gate\n"
    "gate id a { U(0, 0, 0) a; }\n"
    "// IBM Quantum experience gates\n"
    "gate u1(lambda) q { U(0, 0, lambda) q; }\n"
    "gate u2(phi, lambda) q { gphase(-(phi+lambda)/2); U(pi/2, phi, lambda) q; "
    "}\n"
    "gate u3(theta, phi, lambda) q { gphase(-(phi+lambda)/2); U(theta, phi, "
    "lambda) q; }\n"
    "";

const std::string QE1LIB = "gate rccx a, b, c {\n"
                           "  u2(0, pi) c; u1(pi/4) c; \n"
                           "  cx b, c; u1(-pi/4) c; \n"
                           "  cx a, c; u1(pi/4) c; \n"
                           "  cx b, c; u1(-pi/4) c; \n"
                           "  u2(0, pi) c; \n"
                           "}\n"
                           "gate rc3x a,b,c,d {\n"
                           "  u2(0,pi) d; u1(pi/4) d; \n"
                           "  cx c,d; u1(-pi/4) d; u2(0,pi) d; \n"
                           "  cx a,d; u1(pi/4) d; \n"
                           "  cx b,d; u1(-pi/4) d; \n"
                           "  cx a,d; u1(pi/4) d; \n"
                           "  cx b,d; u1(-pi/4) d; \n"
                           "  u2(0,pi) d; u1(pi/4) d; \n"
                           "  cx c,d; u1(-pi/4) d; \n"
                           "  u2(0,pi) d; \n"
                           "}\n"
                           "gate c3x a,b,c,d {\n"
                           "  h d; cu1(-pi/4) a,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(pi/4) b,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(-pi/4) b,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/4) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/4) c,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/4) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/4) c,d; h d; \n"
                           "}\n"
                           "gate c3sqrtx a,b,c,d {\n"
                           "  h d; cu1(-pi/8) a,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(pi/8) b,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(-pi/8) b,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/8) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/8) c,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/8) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/8) c,d; h d; \n"
                           "}\n"
                           "gate c4x a,b,c,d,e {\n"
                           "  h e; cu1(-pi/2) d,e; h e; \n"
                           "  c3x a,b,c,d; \n"
                           "  h e; cu1(pi/2) d,e; h e; \n"
                           "  c3x a,b,c,d; \n"
                           "  c3sqrtx a,b,c,e; \n"
                           "}\n";

const std::map<std::string, std::shared_ptr<Gate>> STANDARD_GATES = {
    // gates from which all other gates can be constructed.
    {"gphase",
     std::make_shared<StandardGate>(StandardGate({0, 0, 1, qc::GPhase}))},
    {"U", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::U}))},

    // The controlled gphase is problematic so we natively support the phase
    // gate.
    {"p", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::P}))},
    // we use mqt's native swap gate instead of the one from the stdgates.inc.
    {"swap", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::SWAP}))},
    {"iswap",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::iSWAP}))},

    // gates from the stdgates.inc file which can't be parsed at the moment as
    // the `pow` modifier is unsupported at the moment.
    {"s", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::S}))},
    {"sdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Sdg}))},
    {"t", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::T}))},
    {"tdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Tdg}))},
    {"sx", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::SX}))},
    {"sxdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::SXdg}))},
    {"teleport", std::make_shared<StandardGate>(
                     StandardGate({0, 3, 0, qc::Teleportation}))},
};

const std::map<std::string, std::shared_ptr<Gate>> QASM2_COMPAT_GATES = {
    // natively supported gates for backward compatibility with OpenQASM 2.0
    {"rxx", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RXX}))},
    {"ryy", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RYY}))},
    {"rzz", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RZZ}))},
    {"rzx", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::RZX}))},
    {"dcx", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::DCX}))},
    {"ecr", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::ECR}))},
    {"xx_minus_yy",
     std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::XXminusYY}))},
    {"xx_plus_yy",
     std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::XXplusYY}))},
};
} // namespace qasm3
