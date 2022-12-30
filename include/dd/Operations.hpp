/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Definitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/StandardOperation.hpp"

#include <variant>

namespace qc {
    using VectorDD        = dd::vEdge;
    using MatrixDD        = dd::mEdge;
    using DensityMatrixDD = dd::dEdge;
} // namespace qc

namespace dd {
    // single-target Operations
    template<class Config>
    qc::MatrixDD getStandardOperationDD(const qc::StandardOperation* op, std::unique_ptr<dd::Package<Config>>& dd, const dd::Controls& controls, dd::Qubit target, bool inverse) {
        GateMatrix gm;

        const auto  type       = op->getType();
        const auto  nqubits    = static_cast<dd::QubitCount>(op->getNqubits());
        const auto  startQubit = static_cast<std::size_t>(op->getStartingQubit());
        const auto& parameter  = op->getParameter();

        switch (type) {
            case qc::I: gm = dd::Imat; break;
            case qc::H: gm = dd::Hmat; break;
            case qc::X: {
                qc::MatrixDD e{};
                if (controls.size() > 1U) { // Toffoli
                    e = dd->toffoliTable.lookup(nqubits, controls, target);
                    if (e.p == nullptr) {
                        e = dd->makeGateDD(dd::Xmat, nqubits, controls, target, startQubit);
                        dd->toffoliTable.insert(nqubits, controls, target, e);
                    }
                    return e;
                }
                gm = dd::Xmat;
                break;
            }
            case qc::Y: gm = dd::Ymat; break;
            case qc::Z: gm = dd::Zmat; break;
            case qc::S: gm = inverse ? dd::Sdagmat : dd::Smat; break;
            case qc::Sdag: gm = inverse ? dd::Smat : dd::Sdagmat; break;
            case qc::T: gm = inverse ? dd::Tdagmat : dd::Tmat; break;
            case qc::Tdag: gm = inverse ? dd::Tmat : dd::Tdagmat; break;
            case qc::V: gm = inverse ? dd::Vdagmat : dd::Vmat; break;
            case qc::Vdag: gm = inverse ? dd::Vmat : dd::Vdagmat; break;
            case qc::U3: gm = inverse ? dd::U3mat(-parameter[1U], -parameter[0U], -parameter[2U]) : dd::U3mat(parameter[0U], parameter[1U], parameter[2U]); break;
            case qc::U2: gm = inverse ? dd::U2mat(-parameter[1U] + dd::PI, -parameter[0U] - dd::PI) : dd::U2mat(parameter[0U], parameter[1U]); break;
            case qc::Phase: gm = inverse ? dd::Phasemat(-parameter[0U]) : dd::Phasemat(parameter[0U]); break;
            case qc::SX: gm = inverse ? dd::SXdagmat : dd::SXmat; break;
            case qc::SXdag: gm = inverse ? dd::SXmat : dd::SXdagmat; break;
            case qc::RX: gm = inverse ? dd::RXmat(-parameter[0U]) : dd::RXmat(parameter[0U]); break;
            case qc::RY: gm = inverse ? dd::RYmat(-parameter[0U]) : dd::RYmat(parameter[0U]); break;
            case qc::RZ: gm = inverse ? dd::RZmat(-parameter[0U]) : dd::RZmat(parameter[0U]); break;
            default:
                std::ostringstream oss{};
                oss << "DD for gate" << op->getName() << " not available!";
                throw qc::QFRException(oss.str());
        }
        return dd->makeGateDD(gm, nqubits, controls, target, startQubit);
    }

    // two-target Operations
    template<class Config>
    qc::MatrixDD getStandardOperationDD(const qc::StandardOperation* op, std::unique_ptr<dd::Package<Config>>& dd, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, bool inverse) {
        const auto type       = op->getType();
        const auto nqubits    = static_cast<dd::QubitCount>(op->getNqubits());
        const auto startQubit = static_cast<std::size_t>(op->getStartingQubit());

        switch (type) {
            case qc::SWAP:
                return dd->makeSWAPDD(nqubits, controls, target0, target1, startQubit);
            case qc::iSWAP:
                if (inverse) {
                    return dd->makeiSWAPinvDD(nqubits, controls, target0, target1, startQubit);
                } else {
                    return dd->makeiSWAPDD(nqubits, controls, target0, target1, startQubit);
                }
            case qc::Peres:
                if (inverse) {
                    return dd->makePeresdagDD(nqubits, controls, target0, target1, startQubit);
                } else {
                    return dd->makePeresDD(nqubits, controls, target0, target1, startQubit);
                }
            case qc::Peresdag:
                if (inverse) {
                    return dd->makePeresDD(nqubits, controls, target0, target1, startQubit);
                } else {
                    return dd->makePeresdagDD(nqubits, controls, target0, target1, startQubit);
                }
            default:
                std::ostringstream oss{};
                oss << "DD for gate" << op->getName() << " not available!";
                throw qc::QFRException(oss.str());
        }
    }

    // The methods with a permutation parameter apply these Operations according to the mapping specified by the permutation, e.g.
    //      if perm[0] = 1 and perm[1] = 0
    //      then cx 0 1 will be translated to cx perm[0] perm[1] == cx 1 0

    template<class Config>
    qc::MatrixDD getDD(const qc::Operation* op, std::unique_ptr<dd::Package<Config>>& dd, qc::Permutation& permutation, bool inverse = false) {
        const auto type    = op->getType();
        const auto nqubits = op->getNqubits();

        // check whether the operation can be handled by the underlying DD package
        if (nqubits > dd::Package<Config>::MAX_POSSIBLE_QUBITS) {
            throw qc::QFRException("Requested too many qubits to be handled by the DD package. Qubit datatype only allows up to " +
                                   std::to_string(dd::Package<Config>::MAX_POSSIBLE_QUBITS) + " qubits, while " +
                                   std::to_string(nqubits) + " were requested. If you want to use more than " +
                                   std::to_string(dd::Package<Config>::MAX_POSSIBLE_QUBITS) + " qubits, you have to recompile the package with a wider Qubit type in `export/dd_package/include/dd/Definitions.hpp!`");
        }

        // if a permutation is provided and the current operation is a SWAP, this routine just updates the permutation
        if (!permutation.empty() && type == qc::SWAP && !op->isControlled()) {
            const auto& targets = op->getTargets();

            const auto target0 = targets.at(0U);
            const auto target1 = targets.at(1U);
            // update permutation
            std::swap(permutation.at(target0), permutation.at(target1));
            return dd->makeIdent(static_cast<dd::QubitCount>(nqubits));
        }

        if (type == qc::ShowProbabilities || type == qc::Barrier || type == qc::Snapshot) {
            return dd->makeIdent(static_cast<dd::QubitCount>(nqubits));
        }

        if (const auto* standardOp = dynamic_cast<const qc::StandardOperation*>(op)) {
            auto targets  = op->getTargets();
            auto controls = op->getControls();
            if (!permutation.empty()) {
                targets  = permutation.apply(targets);
                controls = permutation.apply(controls);
            }

            // convert controls to DD controls
            dd::Controls ddControls{};
            for (const auto& c: controls) {
                const auto& qubit = static_cast<dd::Qubit>(c.qubit);
                if (c.type == qc::Control::Type::Pos) {
                    ddControls.emplace(dd::Control{qubit, dd::Control::Type::pos});
                } else {
                    ddControls.emplace(dd::Control{qubit, dd::Control::Type::neg});
                }
            }

            if (op->getType() == qc::SWAP || op->getType() == qc::iSWAP || op->getType() == qc::Peres || op->getType() == qc::Peresdag) {
                assert(targets.size() == 2);
                const auto target0 = static_cast<dd::Qubit>(targets[0U]);
                const auto target1 = static_cast<dd::Qubit>(targets[1U]);
                return getStandardOperationDD(standardOp, dd, ddControls, target0, target1, inverse);
            }
            assert(targets.size() == 1);
            const auto target0 = static_cast<dd::Qubit>(targets[0U]);
            return getStandardOperationDD(standardOp, dd, ddControls, target0, inverse);
        }

        if (const auto* compoundOp = dynamic_cast<const qc::CompoundOperation*>(op)) {
            auto e = dd->makeIdent(static_cast<dd::QubitCount>(op->getNqubits()));
            if (inverse) {
                for (const auto& operation: *compoundOp) {
                    e = dd->multiply(e, getInverseDD(operation.get(), dd, permutation));
                }
            } else {
                for (const auto& operation: *compoundOp) {
                    e = dd->multiply(getDD(operation.get(), dd, permutation), e);
                }
            }
            return e;
        }

        if (const auto* classicOp = dynamic_cast<const qc::ClassicControlledOperation*>(op)) {
            return getDD(classicOp->getOperation(), dd, permutation, inverse);
        }

        assert(op->isNonUnitaryOperation());
        throw qc::QFRException("DD for non-unitary operation not available!");
    }

    template<class Config>
    qc::MatrixDD getDD(const qc::Operation* op, std::unique_ptr<dd::Package<Config>>& dd, bool inverse = false) {
        qc::Permutation perm{};
        return getDD(op, dd, perm, inverse);
    }

    template<class Config>
    qc::MatrixDD getInverseDD(const qc::Operation* op, std::unique_ptr<dd::Package<Config>>& dd) {
        return getDD(op, dd, true);
    }

    template<class Config>
    qc::MatrixDD getInverseDD(const qc::Operation* op, std::unique_ptr<dd::Package<Config>>& dd, qc::Permutation& permutation) {
        return getDD(op, dd, permutation, true);
    }

    template<class Config>
    void dumpTensor(qc::Operation* op, std::ostream& of, std::vector<std::size_t>& inds, std::size_t& gateIdx, std::unique_ptr<dd::Package<Config>>& dd);

    // apply swaps 'on' DD in order to change 'from' to 'to'
    // where |from| >= |to|
    template<class DDType, class Config>
    void changePermutation(DDType& on, qc::Permutation& from, const qc::Permutation& to, std::unique_ptr<dd::Package<Config>>& dd, bool regular = true) {
        assert(from.size() >= to.size());

        // iterate over (k,v) pairs of second permutation
        for (const auto& [i, goal]: to) {
            // search for key in the first map
            auto it = from.find(i);
            if (it == from.end()) {
                throw qc::QFRException("[changePermutation] Key " + std::to_string(it->first) + " was not found in first permutation. This should never happen.");
            }
            auto current = it->second;

            // permutations agree for this key value
            if (current == goal) {
                continue;
            }

            // search for goal value in first permutation
            qc::Qubit j = 0;
            for (const auto& [key, value]: from) {
                if (value == goal) {
                    j = key;
                    break;
                }
            }

            // swap i and j
            auto       saved  = on;
            const auto swapDD = dd->makeSWAPDD(static_cast<dd::QubitCount>(on.p->v + 1), {}, static_cast<dd::Qubit>(from.at(i)), static_cast<dd::Qubit>(from.at(j)));
            if constexpr (std::is_same_v<DDType, qc::VectorDD>) {
                on = dd->multiply(swapDD, on);
            } else {
                // the regular flag only has an effect on matrix DDs
                if (regular) {
                    on = dd->multiply(swapDD, on);
                } else {
                    on = dd->multiply(on, swapDD);
                }
            }

            dd->incRef(on);
            dd->decRef(saved);
            dd->garbageCollect();

            // update permutation
            from.at(i) = goal;
            from.at(j) = current;
        }
    }

} // namespace dd
