/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Definitions.hpp"
#include "OpType.hpp"
#include "dd/ComplexValue.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace qc {
    // Operation Constants
    constexpr std::size_t MAX_PARAMETERS    = 3;  // Max. parameters of an operation
    constexpr std::size_t MAX_STRING_LENGTH = 20; // Ensure short-string-optimizations

    class Operation {
    protected:
        dd::Controls                       controls{};
        Targets                            targets{};
        std::array<dd::fp, MAX_PARAMETERS> parameter{};

        dd::QubitCount nqubits    = 0;
        dd::Qubit      startQubit = 0;
        OpType         type       = None; // Op type
        char           name[MAX_STRING_LENGTH]{};

        static bool isWholeQubitRegister(const RegisterNames& reg, std::size_t start, std::size_t end) {
            return !reg.empty() && reg[start].first == reg[end].first && (start == 0 || reg[start].first != reg[start - 1].first) && (end == reg.size() - 1 || reg[end].first != reg[end + 1].first);
        }

    public:
        Operation() = default;

        Operation(const Operation& op)     = delete;
        Operation(Operation&& op) noexcept = default;

        Operation& operator=(const Operation& op)     = delete;
        Operation& operator=(Operation&& op) noexcept = default;

        // Virtual Destructor
        virtual ~Operation() = default;

        [[nodiscard]] virtual std::unique_ptr<Operation> clone() const = 0;

        // Getters
        [[nodiscard]] virtual const Targets& getTargets() const {
            return targets;
        }
        virtual Targets& getTargets() {
            return targets;
        }
        [[nodiscard]] virtual std::size_t getNtargets() const {
            return targets.size();
        }

        [[nodiscard]] virtual const dd::Controls& getControls() const {
            return controls;
        }
        virtual dd::Controls& getControls() {
            return controls;
        }
        [[nodiscard]] virtual std::size_t getNcontrols() const {
            return controls.size();
        }

        [[nodiscard]] dd::QubitCount getNqubits() const {
            return nqubits;
        }

        [[nodiscard]] const std::array<dd::fp, MAX_PARAMETERS>& getParameter() const {
            return parameter;
        }
        std::array<dd::fp, MAX_PARAMETERS>& getParameter() {
            return parameter;
        }

        [[nodiscard]] const char* getName() const {
            return name;
        }
        [[nodiscard]] virtual OpType getType() const {
            return type;
        }

        [[nodiscard]] virtual dd::Qubit getStartingQubit() const {
            return startQubit;
        }

        [[nodiscard]] virtual std::set<dd::Qubit> getUsedQubits() const {
            const auto&         opTargets  = getTargets();
            const auto&         opControls = getControls();
            std::set<dd::Qubit> usedQubits = {opTargets.begin(), opTargets.end()};
            for (const auto& control: opControls) {
                usedQubits.insert(control.qubit);
            }
            return usedQubits;
        }

        // Setter
        virtual void setNqubits(dd::QubitCount nq) {
            nqubits = nq;
        }

        virtual void setTargets(const Targets& t) {
            targets = t;
        }

        virtual void setControls(const dd::Controls& c) {
            controls = c;
        }

        virtual void setName();

        virtual void setGate(OpType g) {
            type = g;
            setName();
        }

        virtual void setParameter(const std::array<dd::fp, MAX_PARAMETERS>& p) {
            Operation::parameter = p;
        }

        [[nodiscard]] inline virtual bool isUnitary() const {
            return true;
        }

        [[nodiscard]] inline virtual bool isStandardOperation() const {
            return false;
        }

        [[nodiscard]] inline virtual bool isCompoundOperation() const {
            return false;
        }

        [[nodiscard]] inline virtual bool isNonUnitaryOperation() const {
            return false;
        }

        [[nodiscard]] inline virtual bool isClassicControlledOperation() const {
            return false;
        }

        [[nodiscard]] inline virtual bool isSymbolicOperation() const {
            return false;
        }

        [[nodiscard]] inline virtual bool isControlled() const {
            return !controls.empty();
        }

        [[nodiscard]] inline virtual bool actsOn(dd::Qubit i) const {
            for (const auto& t: targets) {
                if (t == i)
                    return true;
            }

            if (controls.count(i) > 0)
                return true;

            return false;
        }

        [[nodiscard]] virtual bool equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const;
        [[nodiscard]] virtual bool equals(const Operation& op) const {
            return equals(op, {}, {});
        }

        virtual std::ostream& printParameters(std::ostream& os) const;
        virtual std::ostream& print(std::ostream& os) const;
        virtual std::ostream& print(std::ostream& os, const Permutation& permutation) const;

        friend std::ostream& operator<<(std::ostream& os, const Operation& op) {
            return op.print(os);
        }

        virtual void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg) const                         = 0;
        virtual void dumpQiskit(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, const char* anc_reg_name) const = 0;
    };
} // namespace qc
