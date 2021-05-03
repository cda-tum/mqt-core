/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_OPERATION_H
#define QFR_OPERATION_H

#include "Definitions.hpp"
#include "dd/Package.hpp"

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

    // Supported Operations
    enum OpType : std::uint8_t {
        None,
        // Standard Operations
        I,
        H,
        X,
        Y,
        Z,
        S,
        Sdag,
        T,
        Tdag,
        V,
        Vdag,
        U3,
        U2,
        Phase,
        SX,
        SXdag,
        RX,
        RY,
        RZ,
        SWAP,
        iSWAP,
        Peres,
        Peresdag,
        // Compound Operation
        Compound,
        // Non Unitary Operations
        Measure,
        Reset,
        Snapshot,
        ShowProbabilities,
        Barrier,
        Teleportation,
        // Classically-controlled Operation
        ClassicControlled
    };

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

        virtual MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, const Targets& targets) const = 0;

        virtual MatrixDD getDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, const Targets& targets) const = 0;

    public:
        Operation()                        = default;
        Operation(const Operation& op)     = delete;
        Operation(Operation&& op) noexcept = default;
        Operation& operator=(const Operation& op) = delete;
        Operation& operator=(Operation&& op) noexcept = default;
        // Virtual Destructor
        virtual ~Operation() = default;

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

        [[nodiscard]] const dd::Controls& getControls() const {
            return controls;
        }
        dd::Controls& getControls() {
            return controls;
        }
        [[nodiscard]] std::size_t getNcontrols() const {
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

        // Public Methods
        // The methods with a permutation parameter apply these operations according to the mapping specified by the permutation, e.g.
        //      if perm[0] = 1 and perm[1] = 0
        //      then cx 0 1 will be translated to cx perm[0] perm[1] == cx 1 0
        virtual MatrixDD getDD(std::unique_ptr<dd::Package>& dd) const {
            return getDD(dd, controls, targets);
        }
        virtual MatrixDD getDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const {
            return getDD(dd, permutation.apply(controls), permutation.apply(targets));
        }

        virtual MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd) const {
            return getInverseDD(dd, controls, targets);
        }
        virtual MatrixDD getInverseDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const {
            return getInverseDD(dd, permutation.apply(controls), permutation.apply(targets));
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
#endif //QFR_OPERATION_H
