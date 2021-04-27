/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_DEFINITIONS_HPP
#define QFR_DEFINITIONS_HPP

#include "dd/Package.hpp"

#include <forward_list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace qc {
    class QFRException: public std::invalid_argument {
        std::string msg;

    public:
        explicit QFRException(std::string msg):
            std::invalid_argument("QFR Exception"), msg(std::move(msg)) {}

        [[nodiscard]] const char* what() const noexcept override {
            return msg.c_str();
        }
    };

    template<class IdxType, class SizeType>
    using Register          = std::pair<IdxType, SizeType>;
    using QuantumRegister   = Register<dd::Qubit, dd::QubitCount>;
    using ClassicalRegister = Register<std::size_t, std::size_t>;
    template<class RegisterType>
    using RegisterMap          = std::map<std::string, RegisterType, std::greater<>>;
    using QuantumRegisterMap   = RegisterMap<QuantumRegister>;
    using ClassicalRegisterMap = RegisterMap<ClassicalRegister>;
    using RegisterNames        = std::vector<std::pair<std::string, std::string>>;

    using VectorDD = dd::Package::vEdge;
    using MatrixDD = dd::Package::mEdge;

    using Targets = std::vector<dd::Qubit>;

    struct Permutation: public std::map<dd::Qubit, dd::Qubit> {
        [[nodiscard]] inline dd::Controls apply(const dd::Controls& controls) const {
            dd::Controls c{};
            for (const auto& control: controls) {
                c.emplace(dd::Control{at(control.qubit), control.type});
            }
            return c;
        }
        [[nodiscard]] inline Targets apply(const Targets& targets) const {
            Targets t{};
            for (const auto& target: targets) {
                t.emplace_back(at(target));
            }
            return t;
        }
    };

    constexpr dd::fp PARAMETER_TOLERANCE = 10e-6;

    // forward declaration
    class Operation;

    // supported file formats
    enum Format {
        Real,
        OpenQASM,
        GRCS,
        Qiskit,
        TFC,
        QC
    };

    using DAG          = std::vector<std::forward_list<std::unique_ptr<Operation>*>>;
    using DAGIterator  = std::forward_list<std::unique_ptr<Operation>*>::iterator;
    using DAGIterators = std::vector<DAGIterator>;
} // namespace qc

#endif //QFR_DEFINITIONS_HPP
