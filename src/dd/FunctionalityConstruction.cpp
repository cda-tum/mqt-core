/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "dd/FunctionalityConstruction.hpp"

namespace dd {
    template<class Config>
    MatrixDD buildFunctionality(const QuantumComputation* qc, std::unique_ptr<dd::Package<Config>>& dd) {
        const auto nqubits = static_cast<dd::QubitCount>(qc->getNqubits());
        if (nqubits == 0U) {
            return MatrixDD::one;
        }

        if (const auto* grover = dynamic_cast<const qc::Grover*>(qc)) {
            return buildFunctionality(grover, dd);
        }

        auto permutation = qc->initialLayout;
        auto e           = dd->createInitialMatrix(nqubits, qc->ancillary);

        for (const auto& op: *qc) {
            auto tmp = dd->multiply(getDD(op.get(), dd, permutation), e);

            dd->incRef(tmp);
            dd->decRef(e);
            e = tmp;

            dd->garbageCollect();
        }
        // correct permutation if necessary
        changePermutation(e, permutation, qc->outputPermutation, dd);
        e = dd->reduceAncillae(e, qc->ancillary);
        e = dd->reduceGarbage(e, qc->garbage);

        return e;
    }

    template<class Config>
    MatrixDD buildFunctionalityRecursive(const QuantumComputation* qc, std::unique_ptr<dd::Package<Config>>& dd) {
        if (qc->getNqubits() == 0U) {
            return MatrixDD::one;
        }

        if (const auto* grover = dynamic_cast<const qc::Grover*>(qc)) {
            return buildFunctionalityRecursive(grover, dd);
        }

        auto permutation = qc->initialLayout;

        if (qc->size() == 1U) {
            auto e = getDD(qc->front().get(), dd, permutation);
            dd->incRef(e);
            return e;
        }

        std::stack<MatrixDD> s{};
        auto                 depth = static_cast<std::size_t>(std::ceil(std::log2(qc->size())));
        buildFunctionalityRecursive(qc, depth, 0, s, permutation, dd);
        auto e = s.top();
        s.pop();

        // correct permutation if necessary
        changePermutation(e, permutation, qc->outputPermutation, dd);
        e = dd->reduceAncillae(e, qc->ancillary);
        e = dd->reduceGarbage(e, qc->garbage);

        return e;
    }

    template<class Config>
    bool buildFunctionalityRecursive(const QuantumComputation* qc, std::size_t depth, std::size_t opIdx, std::stack<MatrixDD>& s, Permutation& permutation, std::unique_ptr<dd::Package<Config>>& dd) {
        // base case
        if (depth == 1U) {
            auto e = getDD(qc->at(opIdx).get(), dd, permutation);
            ++opIdx;
            if (opIdx == qc->size()) { // only one element was left
                s.push(e);
                dd->incRef(e);
                return false;
            }
            auto f = getDD(qc->at(opIdx).get(), dd, permutation);
            s.push(dd->multiply(f, e)); // ! reverse multiplication
            dd->incRef(s.top());
            return (opIdx != qc->size() - 1U);
        }

        // in case no operations are left after the first recursive call nothing has to be done
        const size_t leftIdx = opIdx & ~(1UL << (depth - 1U));
        if (!buildFunctionalityRecursive(qc, depth - 1U, leftIdx, s, permutation, dd)) {
            return false;
        }

        const size_t rightIdx = opIdx | (1UL << (depth - 1));
        const auto   success  = buildFunctionalityRecursive(qc, depth - 1U, rightIdx, s, permutation, dd);

        // get latest two results from stack and push their product on the stack
        auto e = s.top();
        s.pop();
        auto f = s.top();
        s.pop();
        s.push(dd->multiply(e, f)); // ordering because of stack structure

        // reference counting
        dd->decRef(e);
        dd->decRef(f);
        dd->incRef(s.top());
        dd->garbageCollect();

        return success;
    }

    template<class Config>
    MatrixDD buildFunctionality(const qc::Grover* qc, std::unique_ptr<dd::Package<Config>>& dd) {
        QuantumComputation groverIteration(qc->getNqubits());
        qc->oracle(groverIteration);
        qc->diffusion(groverIteration);

        auto iteration = buildFunctionality(&groverIteration, dd);

        auto e = iteration;
        dd->incRef(e);

        for (std::size_t i = 0U; i < qc->iterations - 1U; ++i) {
            auto f = dd->multiply(iteration, e);
            dd->incRef(f);
            dd->decRef(e);
            e = f;
            dd->garbageCollect();
        }

        QuantumComputation setup(qc->getNqubits());
        qc->setup(setup);
        auto g = buildFunctionality(&setup, dd);
        auto f = dd->multiply(e, g);
        dd->incRef(f);
        dd->decRef(e);
        dd->decRef(g);
        e = f;

        dd->decRef(iteration);
        return e;
    }

    template<class Config>
    MatrixDD buildFunctionalityRecursive(const qc::Grover* qc, std::unique_ptr<dd::Package<Config>>& dd) {
        QuantumComputation groverIteration(qc->getNqubits());
        qc->oracle(groverIteration);
        qc->diffusion(groverIteration);

        auto              iter = buildFunctionalityRecursive(&groverIteration, dd);
        auto              e    = iter;
        std::bitset<128U> iterBits(qc->iterations);
        auto              msb = static_cast<std::size_t>(std::floor(std::log2(qc->iterations)));
        auto              f   = iter;
        dd->incRef(f);
        bool zero = !iterBits[0U];
        for (std::size_t j = 1U; j <= msb; ++j) {
            auto tmp = dd->multiply(f, f);
            dd->incRef(tmp);
            dd->decRef(f);
            f = tmp;
            if (iterBits[j]) {
                if (zero) {
                    dd->incRef(f);
                    dd->decRef(e);
                    e    = f;
                    zero = false;
                } else {
                    auto g = dd->multiply(e, f);
                    dd->incRef(g);
                    dd->decRef(e);
                    e = g;
                    dd->garbageCollect();
                }
            }
        }
        dd->decRef(f);

        // apply state preparation setup
        qc::QuantumComputation statePrep(qc->getNqubits());
        qc->setup(statePrep);
        auto s   = buildFunctionality(&statePrep, dd);
        auto tmp = dd->multiply(e, s);
        dd->incRef(tmp);
        dd->decRef(s);
        dd->decRef(e);
        e = tmp;

        return e;
    }

    template<class DDPackage>
    MatrixDD buildFunctionality(GoogleRandomCircuitSampling* qc, std::unique_ptr<DDPackage>& dd, const std::optional<std::size_t> ncycles) {
        if (ncycles.has_value() && (*ncycles < qc->cycles.size() - 2U)) {
            qc->removeCycles(qc->cycles.size() - 2U - *ncycles);
        }

        const auto  nqubits     = static_cast<dd::QubitCount>(qc->getNqubits());
        Permutation permutation = qc->initialLayout;
        auto        e           = dd->makeIdent(nqubits);
        dd->incRef(e);
        for (const auto& cycle: qc->cycles) {
            auto f = dd->makeIdent(nqubits);
            for (const auto& op: cycle) {
                f = dd->multiply(getDD(op.get(), dd, permutation), f);
            }
            auto g = dd->multiply(f, e);
            dd->decRef(e);
            dd->incRef(g);
            e = g;
            dd->garbageCollect();
        }
        return e;
    }

    template MatrixDD buildFunctionality(const qc::QuantumComputation* qc, std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd);
    template MatrixDD buildFunctionalityRecursive(const qc::QuantumComputation* qc, std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd);
    template bool     buildFunctionalityRecursive(const qc::QuantumComputation* qc, const std::size_t depth, const std::size_t opIdx, std::stack<MatrixDD>& s, qc::Permutation& permutation, std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd);
    template MatrixDD buildFunctionality(const qc::Grover* qc, std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd);
    template MatrixDD buildFunctionalityRecursive(const qc::Grover* qc, std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd);
    template MatrixDD buildFunctionality(GoogleRandomCircuitSampling* qc, std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd, const std::optional<std::size_t> ncycles);
} // namespace dd
