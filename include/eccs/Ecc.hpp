/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_Ecc_HPP
#define QFR_Ecc_HPP

#include "EccStatistics.hpp"
#include "QuantumComputation.hpp"

class Ecc {
public:
    enum class ID {
        Id,
        Q3Shor,
        Q9Shor,
        Q5Laflamme,
        Q7Steane,
        Q9Surface,
        Q18Surface,
        QxCustom
    };
    struct Info {
        ID          id;
        int         nRedundantQubits;
        int         nCorrectingBits; //in total (not per qubit); usually number of clbits needed for correcting one qubit
        std::string name;
    };

    const Info ecc;

    Ecc(Info ecc, qc::QuantumComputation& qc, int measureFrequency):
        ecc(std::move(ecc)),
        qcOriginal(qc), measureFrequency(measureFrequency) {
    }
    virtual ~Ecc() = default;

    qc::QuantumComputation& apply();

    virtual std::string getName() {
        return ecc.name;
    }

protected:
    qc::QuantumComputation& qcOriginal;
    qc::QuantumComputation  qcMapped;
    EccStatistics           statistics{};
    const int               measureFrequency;
    bool                    isDecoded    = true;
    bool                    gatesWritten = false;

    virtual void initMappedCircuit();

    virtual void writeEncoding() = 0;

    virtual void measureAndCorrect() = 0;

    virtual void writeDecoding() = 0;

    virtual void mapGate(const qc::Operation& gate) = 0;

    inline void gateNotAvailableError(const qc::Operation& gate) const {
        throw qc::QFRException(std::string("Gate ") + gate.getName() + " not supported to encode in error code " + ecc.name + "!");
    }

    void writeToffoli(int target, int c1, bool p1, int c2, bool p2);

    //void                  writeGeneric(dd::Qubit target, qc::OpType type);
    //[[maybe_unused]] void writeGeneric(dd::Qubit target, const dd::Control& control, qc::OpType type);
    //void                  writeGeneric(dd::Qubit target, const dd::Controls& controls, qc::OpType type);

    void writeClassicalControl(dd::Qubit control, int qubitCount, unsigned int value, qc::OpType opType, int target);

    //static, since some codes need to store those functions into function pointers
    static void writeXstatic(dd::Qubit target, dd::Control control, qc::QuantumComputation* qcMapped);
    static void writeZstatic(dd::Qubit target, dd::Control control, qc::QuantumComputation* qcMapped);
};

#endif //QFR_Ecc_HPP
