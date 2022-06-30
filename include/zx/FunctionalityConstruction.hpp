/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "QuantumComputation.hpp"
#include "ZXDiagram.hpp"
#include "operations/Operation.hpp"

namespace zx {
    class FunctionalityConstruction {
        using op_it = qc::QuantumComputation::const_iterator;

    public:
        static ZXDiagram buildFunctionality(const qc::QuantumComputation* qc);

        static bool transformableToZX(const qc::QuantumComputation* qc);

        static bool transformableToZX(qc::Operation* op);

    protected:
        static bool
                     checkSwap(op_it it, op_it end, Qubit ctrl, Qubit target, const qc::Permutation& p);
        static void  addZSpider(ZXDiagram& diag, zx::Qubit qubit,
                                std::vector<Vertex>& qubits,
                                const Expression& phase = Expression(), EdgeType type = EdgeType::Simple);
        static void  addXSpider(ZXDiagram& diag, Qubit qubit,
                                std::vector<Vertex>& qubits,
                                const Expression& phase = Expression(), EdgeType type = EdgeType::Simple);
        static void  addCnot(ZXDiagram& diag, Qubit ctrl, Qubit target,
                             std::vector<Vertex>& qubits);
        static void  addCphase(ZXDiagram& diag, const PiRational& phase, Qubit ctrl, Qubit target,
                               std::vector<Vertex>& qubits);
        static void  addSwap(ZXDiagram& diag, Qubit ctrl, Qubit target,
                             std::vector<Vertex>& qubits);
        static void  addCcx(ZXDiagram& diag, Qubit ctrl_0, Qubit ctrl_1, Qubit target,
                            std::vector<Vertex>& qubits);
        static op_it parse_op(ZXDiagram& diag, op_it it, op_it end,
                              std::vector<Vertex>& qubits, const qc::Permutation& p);
    };

} // namespace zx
