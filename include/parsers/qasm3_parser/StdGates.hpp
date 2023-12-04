#pragma once

#include "Gate.hpp"

#include <string>

namespace qasm3 {
// from
// https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/stdgates.inc
const std::string STDGATES =
    "// OpenQASM 3 standard gate library\n"
    "\n"
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
    "// phase gate\n"
    "gate phase(lambda) q { U(0, 0, lambda) q; }\n"
    "// controlled-phase\n"
    "gate cphase(lambda) a, b { ctrl @ phase(lambda) a, b; }\n"
    "// IBM Quantum experience gates\n"
    "gate u1(lambda) q { U(0, 0, lambda) q; }\n"
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

    // natively supported gates
    {"p", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::P}))},
    {"x", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::X}))},
    {"y", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Y}))},
    {"z", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Z}))},
    {"h", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::H}))},
    {"rx", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::RX}))},
    {"ry", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::RY}))},
    {"rz", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::RZ}))},
    {"cx", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::X}))},
    {"CX", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::X}))},
    {"id", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::I}))},
    {"u2", std::make_shared<StandardGate>(StandardGate({0, 1, 2, qc::U2}))},
    {"u3", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::U}))},
    {"s", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::S}))},
    {"sdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Sdg}))},
    {"t", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::T}))},
    {"tdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::Tdg}))},
    {"sx", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::SX}))},
    {"sxdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::SXdg}))},
    {"teleport", std::make_shared<StandardGate>(
                     StandardGate({0, 3, 0, qc::Teleportation}))},
    {"swap", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::SWAP}))},
    {"iswap",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::iSWAP}))},
    {"iswapdg",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::iSWAPdg}))},
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
