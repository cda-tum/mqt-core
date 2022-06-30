#pragma once

#include "QuantumComputation.hpp"

class SymbolicQuantumComputation: public QuantumComputation {
public:
    void
    u3(dd::Qubit target, const Symbolic& lambda, const Symbolic& phi, const Symbolic& theta) {
        checkQubitRange(target);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::U3, lambda, phi, theta);
    }
    void u3(dd::Qubit target, const dd::Control& control, const Symbolic& lambda, const Symbolic& phi, const Symbolic& theta) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::U3, lambda, phi, theta);
    }
    void u3(dd::Qubit target, const dd::Controls& controls, const Symbolic& lambda, const Symbolic& phi, const Symbolic& theta) {
        checkQubitRange(target, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target, qc::U3, lambda, phi, theta);
    }

    void u2(dd::Qubit target, const Symbolic& lambda, const Symbolic& phi) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::U2, lambda, phi);
    }
    void u2(dd::Qubit target, const dd::Control& control, const Symbolic& lambda, const Symbolic& phi) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::U2, lambda, phi);
    }
    void u2(dd::Qubit target, const dd::Controls& controls, const Symbolic& lambda, const Symbolic& phi) {
        checkQubitRange(target, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target, qc::U2, lambda, phi);
    }

    void phase(dd::Qubit target, const Symbolic& lambda) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::Phase, lambda);
    }
    void phase(dd::Qubit target, const dd::Control& control, const Symbolic& lambda) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::Phase, lambda);
    }
    void phase(dd::Qubit target, const dd::Controls& controls, const Symbolic& lambda) {
        checkQubitRange(target, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Phase, lambda);
    }

    void sx(dd::Qubit target) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::SX);
    }
    void sx(dd::Qubit target, const dd::Control& control) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::SX);
    }
    void sx(dd::Qubit target, const dd::Controls& controls) {
        checkQubitRange(target, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target, qc::SX);
    }

    void sxdag(dd::Qubit target) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::SXdag);
    }
    void sxdag(dd::Qubit target, const dd::Control& control) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::SXdag);
    }
    void sxdag(dd::Qubit target, const dd::Controls& controls) {
        checkQubitRange(target, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target, qc::SXdag);
    }

    void rx(dd::Qubit target, const Symbolic& lambda) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::RX, lambda);
    }
    void rx(dd::Qubit target, const dd::Control& control, const Symbolic& lambda) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::RX, lambda);
    }
    void rx(dd::Qubit target, const dd::Controls& controls, const Symbolic& lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RX, lambda); }

    void ry(dd::Qubit target, const Symbolic& lambda) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::RY, lambda);
    }
    void ry(dd::Qubit target, const dd::Control& control, const Symbolic& lambda) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::RY, lambda);
    }
    void ry(dd::Qubit target, const dd::Controls& controls, const Symbolic& lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RY, lambda); }

    void rz(dd::Qubit target, const Symbolic& lambda) {
        checkQubitRange(target);
        emplace_back<StandardOperation>(getNqubits(), target, qc::RZ, lambda);
    }
    void rz(dd::Qubit target, const dd::Control& control, const Symbolic& lambda) {
        checkQubitRange(target, control);
        emplace_back<StandardOperation>(getNqubits(), control, target, qc::RZ, lambda);
    }
    void rz(dd::Qubit target, const dd::Controls& controls, const Symbolic& lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RZ, lambda); }

    void swap(dd::Qubit target0, dd::Qubit target1) {
        checkQubitRange(target0, target1);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::SWAP);
    }
    void swap(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) {
        checkQubitRange(target0, target1, control);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::SWAP);
    }
    void swap(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) {
        checkQubitRange(target0, target1, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::SWAP);
    }

    void iswap(dd::Qubit target0, dd::Qubit target1) {
        checkQubitRange(target0, target1);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::iSWAP);
    }
    void iswap(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) {
        checkQubitRange(target0, target1, control);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::iSWAP);
    }
    void iswap(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) {
        checkQubitRange(target0, target1, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::iSWAP);
    }

    void peres(dd::Qubit target0, dd::Qubit target1) {
        checkQubitRange(target0, target1);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::Peres);
    }
    void peres(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) {
        checkQubitRange(target0, target1, control);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::Peres);
    }
    void peres(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) {
        checkQubitRange(target0, target1, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::Peres);
    }

    void peresdag(dd::Qubit target0, dd::Qubit target1) {
        checkQubitRange(target0, target1);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::Peresdag);
    }
    void peresdag(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) {
        checkQubitRange(target0, target1, control);
        emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::Peresdag);
    }
    void peresdag(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) {
        checkQubitRange(target0, target1, controls);
        emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::Peresdag);
    }

    void measure(dd::Qubit qubit, std::size_t clbit) {
        checkQubitRange(qubit);
        emplace_back<NonUnitaryOperation>(getNqubits(), qubit, clbit);
    }
    void measure(const std::vector<dd::Qubit>&   qubitRegister,
                 const std::vector<std::size_t>& classicalRegister) {
        checkQubitRange(qubitRegister);
        emplace_back<NonUnitaryOperation>(getNqubits(), qubitRegister,
                                          classicalRegister);
    }

    void reset(dd::Qubit target) {
        checkQubitRange(target);
        emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<dd::Qubit>{target}, qc::Reset);
    }
    void reset(const std::vector<dd::Qubit>& targets) {
        checkQubitRange(targets);
        emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Reset);
    }

    void barrier(dd::Qubit target) {
        checkQubitRange(target);
        emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<dd::Qubit>{target}, qc::Barrier);
    }
    void barrier(const std::vector<dd::Qubit>& targets) {
        checkQubitRange(targets);
        emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Barrier);
    }
};
