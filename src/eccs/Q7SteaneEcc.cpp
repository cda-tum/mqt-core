/*
* This file is part of JKQ QFR library which is released under the MIT license.
* See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
*/

#include "eccs/Q7SteaneEcc.hpp"

//7 data qubits, 6 for measuring -> 13 qubits per physical qubit (6 classical for measuring at end)
Q7SteaneEcc::Q7SteaneEcc(qc::QuantumComputation &qc, int measureFq, bool decomposeMC, bool cliffOnly) : Ecc(
           {ID::Q7Steane, 7, 6, Q7SteaneEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q7SteaneEcc::initMappedCircuit() {
   //method is overridden because we need 2 kinds of classical measurement output registers
   qc.stripIdleQubits(true, false);
   statistics.nInputQubits = qc.getNqubits();
   statistics.nInputClassicalBits = (int) qc.getNcbits();
   statistics.nOutputQubits = qc.getNqubits() * ecc.nRedundantQubits + ecc.nCorrectingBits;
   statistics.nOutputClassicalBits = statistics.nInputClassicalBits + ecc.nCorrectingBits+7;
   qcMapped.addQubitRegister(statistics.nOutputQubits);
   qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
   qcMapped.addClassicalRegister(3, "qeccX");
   qcMapped.addClassicalRegister(3, "qeccZ");
}

void Q7SteaneEcc::writeEncoding() {
   measureAndCorrect();
   decodingDone = false;
}

void Q7SteaneEcc::measureAndCorrect() {
   const int nQubits = qc.getNqubits();
   const int ancStart = nQubits * ecc.nRedundantQubits;
   const int clAncStart = static_cast<int>(qc.getNcbits());

   for (int i = 0; i < nQubits; i++) {
       qcMapped.reset(static_cast<dd::Qubit>(ancStart));
       qcMapped.reset(static_cast<dd::Qubit>(ancStart + 1));
       qcMapped.reset(static_cast<dd::Qubit>(ancStart + 2));
       qcMapped.reset(static_cast<dd::Qubit>(ancStart + 3));
       qcMapped.reset(static_cast<dd::Qubit>(ancStart + 4));
       qcMapped.reset(static_cast<dd::Qubit>(ancStart + 5));

       qcMapped.h(static_cast<dd::Qubit>(ancStart));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 1));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 2));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 3));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 4));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 5));

       auto c0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
       auto c1 = dd::Control{dd::Qubit(ancStart + 1), dd::Control::Type::pos};
       auto c2 = dd::Control{dd::Qubit(ancStart + 2), dd::Control::Type::pos};
       auto c3 = dd::Control{dd::Qubit(ancStart + 3), dd::Control::Type::pos};
       auto c4 = dd::Control{dd::Qubit(ancStart + 4), dd::Control::Type::pos};
       auto c5 = dd::Control{dd::Qubit(ancStart + 5), dd::Control::Type::pos};

       //K1: XIXIXIX
       writeX(static_cast<dd::Qubit>(i + nQubits * 0), c0);
       writeX(static_cast<dd::Qubit>(i + nQubits * 2), c0);
       writeX(static_cast<dd::Qubit>(i + nQubits * 4), c0);
       writeX(static_cast<dd::Qubit>(i + nQubits * 6), c0);

       //K2: IXXIIXX
       writeX(static_cast<dd::Qubit>(i + nQubits * 1), c1);
       writeX(static_cast<dd::Qubit>(i + nQubits * 2), c1);
       writeX(static_cast<dd::Qubit>(i + nQubits * 5), c1);
       writeX(static_cast<dd::Qubit>(i + nQubits * 6), c1);

       //K3: IIIXXXX
       writeX(static_cast<dd::Qubit>(i + nQubits * 3), c2);
       writeX(static_cast<dd::Qubit>(i + nQubits * 4), c2);
       writeX(static_cast<dd::Qubit>(i + nQubits * 5), c2);
       writeX(static_cast<dd::Qubit>(i + nQubits * 6), c2);

       //K2: ZIZIZIZ
       writeZ(static_cast<dd::Qubit>(i + nQubits * 0), c3);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 2), c3);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 4), c3);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 6), c3);

       //K3: IZZIIZZ
       writeZ(static_cast<dd::Qubit>(i + nQubits * 1), c4);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 2), c4);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 5), c4);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 6), c4);

       //K1: IIIZZZZ
       writeZ(static_cast<dd::Qubit>(i + nQubits * 3), c5);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 4), c5);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 5), c5);
       writeZ(static_cast<dd::Qubit>(i + nQubits * 6), c5);


       qcMapped.h(static_cast<dd::Qubit>(ancStart));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 1));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 2));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 3));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 4));
       qcMapped.h(static_cast<dd::Qubit>(ancStart + 5));

       qcMapped.measure(static_cast<dd::Qubit>(ancStart), clAncStart);
       qcMapped.measure(static_cast<dd::Qubit>(ancStart + 1), clAncStart + 1);
       qcMapped.measure(static_cast<dd::Qubit>(ancStart + 2), clAncStart + 2);
       qcMapped.measure(static_cast<dd::Qubit>(ancStart + 3), clAncStart + 3);
       qcMapped.measure(static_cast<dd::Qubit>(ancStart + 4), clAncStart + 4);
       qcMapped.measure(static_cast<dd::Qubit>(ancStart + 5), clAncStart + 5);

       //correct Z_i for i+1 = c0*1+c1*2+c2*4
       //correct X_i for i+1 = c3*1+c4*2+c5*4
       for (unsigned int j = 0; j < 7; j++) {
           if (cliffordGatesOnly) {
               std::unique_ptr<qc::Operation> opS0 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(),
                                                                                             i + j * nQubits, qc::S);
               const auto pairZ0 = std::make_pair(dd::Qubit(clAncStart), dd::QubitCount(3));
               qcMapped.emplace_back<qc::ClassicControlledOperation>(opS0, pairZ0, j + 1U);

               std::unique_ptr<qc::Operation> opS1 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(),
                                                                                             i + j * nQubits, qc::S);
               const auto pairZ1 = std::make_pair(dd::Qubit(clAncStart), dd::QubitCount(3));
               qcMapped.emplace_back<qc::ClassicControlledOperation>(opS1, pairZ1, j + 1U);

               std::unique_ptr<qc::Operation> opH2 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(),
                                                                                             i + j * nQubits, qc::H);
               const auto pairX2 = std::make_pair(dd::Qubit(clAncStart + 3), dd::QubitCount(3));
               qcMapped.emplace_back<qc::ClassicControlledOperation>(opH2, pairX2, j + 1U);

               std::unique_ptr<qc::Operation> opS3 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(),
                                                                                             i + j * nQubits, qc::S);
               const auto pairX3 = std::make_pair(dd::Qubit(clAncStart + 3), dd::QubitCount(3));
               qcMapped.emplace_back<qc::ClassicControlledOperation>(opS3, pairX3, j + 1U);

               std::unique_ptr<qc::Operation> opS4 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(),
                                                                                             i + j * nQubits, qc::S);
               const auto pairX4 = std::make_pair(dd::Qubit(clAncStart + 3), dd::QubitCount(3));
               qcMapped.emplace_back<qc::ClassicControlledOperation>(opS4, pairX4, j + 1U);

               std::unique_ptr<qc::Operation> opH5 = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(),
                                                                                             i + j * nQubits, qc::H);
               const auto pairX5 = std::make_pair(dd::Qubit(clAncStart + 3), dd::QubitCount(3));
               qcMapped.emplace_back<qc::ClassicControlledOperation>(opH5, pairX5, j + 1U);
           } else {
               writeClassicalControl(dd::Qubit(clAncStart), 3, j+1U, qc::Z, i + j * nQubits );
               writeClassicalControl(dd::Qubit(clAncStart+3), 3, j+1U, qc::X, i + j * nQubits );
           }
       }

   }
}

void Q7SteaneEcc::writeDecoding() {
   const int nQubits = qc.getNqubits();
   const int clAncStart = static_cast<int>(qc.getNcbits());
   unsigned int correction_needed[] = {1,2,4,7};//values with odd amount of '1' bits
   //use exiting registers qeccX and qeccZ for decoding

   for (int i = 0; i < nQubits; i++) {
       //#|###|###
       //0|111|111
       //odd amount of 1's -> x[0] = 1
       //measure from index 1 (not 0) to 6, =qubit 2 to 7
       for (int j = 1; j < 7; j++) { //skip q[0], since values are mapped onto q[0]
           qcMapped.measure(static_cast<dd::Qubit>(i + j * nQubits), clAncStart+j-1);
       }
       for(unsigned int value: correction_needed) {
           writeClassicalControl(clAncStart, 3, value, qc::X, i);
           writeClassicalControl(clAncStart+3, 3, value, qc::X, i);
       }

   }
   decodingDone = true;
}

void Q7SteaneEcc::mapGate(const std::unique_ptr<qc::Operation> &gate) {
   if (decodingDone && gate->getType() != qc::Measure) {
       writeEncoding();
   }
   const int nQubits = qc.getNqubits();
   qc::NonUnitaryOperation *measureGate;
   switch (gate->getType()) {
       case qc::I:
           break;
       case qc::X:
       case qc::H:
       case qc::Y:
       case qc::Z:
           for (std::size_t t = 0; t < gate->getNtargets(); t++) {
               int i = static_cast<unsigned char>(gate->getTargets()[t]);
               if (gate->getNcontrols() == 2 && decomposeMultiControlledGates) {

                   auto &ctrls = gate->getControls();
                   int idx = 0;
                   int ctrl2[2] = {-1, -1};
                   bool ctrl2T[2] = {true, true};
                   for (const auto &ct: ctrls) {
                       ctrl2[idx] = static_cast<unsigned char>(ct.qubit);
                       ctrl2T[idx] = ct.type == dd::Control::Type::pos;
                       idx++;
                   }
                   if (gate->getType() == qc::X) {
                       for (int j = 0; j < 7; j++) {
                           writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                        ctrl2T[1]);
                       }
                   } else if (gate->getType() == qc::Z) {
                       for (int j = 0; j < 7; j++) {
                           qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                           writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                        ctrl2T[1]);
                           qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                       }
                   } else if (gate->getType() == qc::Y) {
                       for (int j = 0; j < 7; j++) {
                           writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                        ctrl2T[1]);
                           qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                           writeToffoli(i + j * nQubits, ctrl2[0] + j * nQubits, ctrl2T[0], ctrl2[1] + j * nQubits,
                                        ctrl2T[1]);
                           qcMapped.h(static_cast<dd::Qubit>(i + j * nQubits));
                       }
                   } else {
                       gateNotAvailableError(gate);
                   }
               } else if (gate->getNcontrols() > 2 && decomposeMultiControlledGates) {
                   gateNotAvailableError(gate);
               } else if (gate->getNcontrols()) {
                   auto &ctrls = gate->getControls();
                   for (int j = 0; j < 7; j++) {
                       dd::Controls ctrls2;
                       for (const auto &ct: ctrls) {
                           ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                       }
                       //qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, ctrls2, i+j*nQubits, gate->getType());
                       writeGeneric(static_cast<dd::Qubit>(i + j * nQubits), ctrls2, gate->getType());
                   }
               } else {
                   for (int j = 0; j < 7; j++) {
                       writeGeneric(static_cast<dd::Qubit>(i + j * nQubits), gate->getType());
                       //qcMapped.emplace_back<qc::StandardOperation>(nQubits*ecc.nRedundantQubits, i+j*nQubits, gate->getType());
                   }
               }
           }
           break;
           //locigal S = 3 physical S's
       case qc::S:
       case qc::Sdag:
           for (std::size_t t = 0; t < gate->getNtargets(); t++) {
               int i = static_cast<unsigned char>(gate->getTargets()[t]);
               if (gate->getNcontrols() > 1 && decomposeMultiControlledGates) {
                   gateNotAvailableError(gate);
               } else if (gate->getNcontrols() && cliffordGatesOnly) {
                   gateNotAvailableError(gate);
               } else if (gate->getNcontrols()) {
                   auto &ctrls = gate->getControls();
                   for (int j = 0; j < 7; j++) {
                       dd::Controls ctrls2;
                       for (const auto &ct: ctrls) {
                           ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                       }
                       qcMapped.emplace_back<qc::StandardOperation>(nQubits * ecc.nRedundantQubits, ctrls2,
                                                                    i + j * nQubits, gate->getType());
                       qcMapped.emplace_back<qc::StandardOperation>(nQubits * ecc.nRedundantQubits, ctrls2,
                                                                    i + j * nQubits, gate->getType());
                       qcMapped.emplace_back<qc::StandardOperation>(nQubits * ecc.nRedundantQubits, ctrls2,
                                                                    i + j * nQubits, gate->getType());
                   }
               } else {
                   if (gate->getType() == qc::S) {
                       for (int j = 0; j < 7; j++) {
                           qcMapped.s(static_cast<dd::Qubit>(i + j * nQubits));
                           qcMapped.s(static_cast<dd::Qubit>(i + j * nQubits));
                           qcMapped.s(static_cast<dd::Qubit>(i + j * nQubits));
                       }
                   } else {
                       for (int j = 0; j < 7; j++) {
                           writeSdag(static_cast<dd::Qubit>(i + j * nQubits));
                           writeSdag(static_cast<dd::Qubit>(i + j * nQubits));
                           writeSdag(static_cast<dd::Qubit>(i + j * nQubits));
                       }
                   }
               }
           }
           break;
       case qc::Measure:
           if (!decodingDone) {
               measureAndCorrect();
               writeDecoding();
           }
           measureGate = (qc::NonUnitaryOperation *) gate.get();
           for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
               qcMapped.measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), measureGate->getTargets()[j]);
           }
           break;
       case qc::T:
       case qc::Tdag:
           for (std::size_t t = 0; t < gate->getNtargets(); t++) {
               int i = static_cast<unsigned char>(gate->getTargets()[t]);
               if (gate->getControls().empty()) {
                   //arXiv:1610.03309v1
                   qcMapped.x(dd::Qubit(i + 5 * nQubits), dd::Control{dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos});
                   qcMapped.x(dd::Qubit(i + 0 * nQubits), dd::Control{dd::Qubit(i + 5 * nQubits), dd::Control::Type::pos});
                   if (gate->getType() == qc::T) {
                       qcMapped.t(dd::Qubit(i + 0 * nQubits));
                   } else {
                       qcMapped.tdag(dd::Qubit(i + 0 * nQubits));
                   }
                   qcMapped.x(dd::Qubit(i + 0 * nQubits), dd::Control{dd::Qubit(i + 5 * nQubits), dd::Control::Type::pos});
                   qcMapped.x(dd::Qubit(i + 5 * nQubits), dd::Control{dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos});
               } else {
                   gateNotAvailableError(gate);
               }
           }
           break;
       case qc::V:
       case qc::Vdag:
       case qc::U3:
       case qc::U2:
       case qc::Phase:
       case qc::SX:
       case qc::SXdag:
       case qc::RX:
       case qc::RY:
       case qc::RZ:
       case qc::SWAP:
       case qc::iSWAP:
       case qc::Peres:
       case qc::Peresdag:
       case qc::Compound:
       case qc::ClassicControlled:
       default:
           gateNotAvailableError(gate);
   }
}

