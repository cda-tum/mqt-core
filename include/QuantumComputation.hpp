/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Definitions.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/StandardOperation.hpp"
#include "operations/SymbolicOperation.hpp"
#include "parsers/qasm_parser/Parser.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qc {
    class CircuitOptimizer;

    class QuantumComputation {
    public:
        using iterator       = typename std::vector<std::unique_ptr<Operation>>::iterator;
        using const_iterator = typename std::vector<std::unique_ptr<Operation>>::const_iterator;

        friend class CircuitOptimizer;

    protected:
        std::vector<std::unique_ptr<Operation>> ops{};
        std::size_t                             nqubits     = 0;
        std::size_t                             nclassics   = 0;
        std::size_t                             nancillae   = 0;
        std::size_t                             maxControls = 0;
        std::string                             name;

        // register names are used as keys, while the values are `{startIndex, length}` pairs
        QuantumRegisterMap   qregs{};
        ClassicalRegisterMap cregs{};
        QuantumRegisterMap   ancregs{};

        std::mt19937_64 mt;
        std::size_t     seed = 0;

        std::unordered_set<sym::Variable> occuringVariables;

        void importOpenQASM(std::istream& is);
        void importReal(std::istream& is);
        int  readRealHeader(std::istream& is);
        void readRealGateDescriptions(std::istream& is, int line);
        void importTFC(std::istream& is);
        int  readTFCHeader(std::istream& is, std::map<std::string, Qubit>& varMap);
        void readTFCGateDescriptions(std::istream& is, int line, std::map<std::string, Qubit>& varMap);
        void importQC(std::istream& is);
        int  readQCHeader(std::istream& is, std::map<std::string, Qubit>& varMap);
        void readQCGateDescriptions(std::istream& is, int line, std::map<std::string, Qubit>& varMap);
        void importGRCS(std::istream& is);

        template<class RegisterType>
        static void printSortedRegisters(const RegisterMap<RegisterType>& regmap, const std::string& identifier, std::ostream& of) {
            // sort regs by start index
            std::map<decltype(RegisterType::first), std::pair<std::string, RegisterType>> sortedRegs{};
            for (const auto& reg: regmap) {
                sortedRegs.insert({reg.second.first, reg});
            }

            for (const auto& reg: sortedRegs) {
                of << identifier << " " << reg.second.first << "[" << reg.second.second.second << "];" << std::endl;
            }
        }
        template<class RegisterType>
        static void consolidateRegister(RegisterMap<RegisterType>& regs) {
            bool finished = false;
            while (!finished) {
                for (const auto& qreg: regs) {
                    finished     = true;
                    auto regname = qreg.first;
                    // check if lower part of register
                    if (regname.length() > 2 && regname.compare(regname.size() - 2, 2, "_l") == 0) {
                        auto lowidx = qreg.second.first;
                        auto lownum = qreg.second.second;
                        // search for higher part of register
                        auto highname = regname.substr(0, regname.size() - 1) + 'h';
                        auto it       = regs.find(highname);
                        if (it != regs.end()) {
                            auto highidx = it->second.first;
                            auto highnum = it->second.second;
                            // fusion of registers possible
                            if (lowidx + lownum == highidx) {
                                finished        = false;
                                auto targetname = regname.substr(0, regname.size() - 2);
                                auto targetidx  = lowidx;
                                auto targetnum  = lownum + highnum;
                                regs.insert({targetname, {targetidx, targetnum}});
                                regs.erase(regname);
                                regs.erase(highname);
                            }
                        }
                        break;
                    }
                }
            }
        }

        template<class RegisterType>
        static void createRegisterArray(const RegisterMap<RegisterType>& regs, RegisterNames& regnames, decltype(RegisterType::second) defaultnumber, const std::string& defaultname) {
            regnames.clear();

            std::stringstream ss;
            if (!regs.empty()) {
                // sort regs by start index
                std::map<decltype(RegisterType::first), std::pair<std::string, RegisterType>> sortedRegs{};
                for (const auto& reg: regs) {
                    sortedRegs.insert({reg.second.first, reg});
                }

                for (const auto& reg: sortedRegs) {
                    for (decltype(RegisterType::second) i = 0; i < reg.second.second.second; i++) {
                        ss << reg.second.first << "[" << i << "]";
                        regnames.push_back(std::make_pair(reg.second.first, ss.str()));
                        ss.str(std::string());
                    }
                }
            } else {
                for (decltype(RegisterType::second) i = 0; i < defaultnumber; i++) {
                    ss << defaultname << "[" << i << "]";
                    regnames.emplace_back(defaultname, ss.str());
                    ss.str(std::string());
                }
            }
        }

        [[nodiscard]] std::size_t getSmallestAncillary() const {
            for (std::size_t i = 0; i < ancillary.size(); ++i) {
                if (ancillary[i]) {
                    return i;
                }
            }
            return ancillary.size();
        }

        [[nodiscard]] std::size_t getSmallestGarbage() const {
            for (std::size_t i = 0; i < garbage.size(); ++i) {
                if (garbage[i]) {
                    return i;
                }
            }
            return garbage.size();
        }
        [[nodiscard]] bool isLastOperationOnQubit(const const_iterator& opIt) const {
            const auto end = ops.cend();
            return isLastOperationOnQubit(opIt, end);
        }
        void checkQubitRange(Qubit qubit) const;
        void checkQubitRange(Qubit qubit0, Qubit qubit1) const;
        void checkQubitRange(Qubit qubit, const Control& control) const;
        void checkQubitRange(Qubit qubit0, Qubit qubit1, const Control& control) const;
        void checkQubitRange(Qubit qubit, const Controls& controls) const;
        void checkQubitRange(Qubit qubit0, Qubit qubit1, const Controls& controls) const;
        void checkQubitRange(const std::vector<Qubit>& qubits) const;

    public:
        QuantumComputation() = default;
        explicit QuantumComputation(const std::size_t nq, const std::size_t s = 0):
            seed(s) {
            addQubitRegister(nq);
            addClassicalRegister(nq);
            if (seed != 0) {
                mt.seed(seed);
            } else {
                // create and properly seed rng
                std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> randomData{};
                std::random_device                                                    rd;
                std::generate(std::begin(randomData), std::end(randomData), [&rd]() { return rd(); });
                std::seed_seq seeds(std::begin(randomData), std::end(randomData));
                mt.seed(seeds);
            }
        }
        explicit QuantumComputation(const std::string& filename, const std::size_t s = 0U):
            seed(s) {
            import(filename);
            if (seed != 0U) {
                mt.seed(seed);
            } else {
                // create and properly seed rng
                std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> randomData{};
                std::random_device                                                    rd;
                std::generate(std::begin(randomData), std::end(randomData), [&rd]() { return rd(); });
                std::seed_seq seeds(std::begin(randomData), std::end(randomData));
                mt.seed(seeds);
            }
        }
        QuantumComputation(const QuantumComputation& qc)     = delete;
        QuantumComputation(QuantumComputation&& qc) noexcept = default;

        QuantumComputation& operator=(const QuantumComputation& qc) = delete;

        QuantumComputation& operator=(QuantumComputation&& qc) noexcept = default;

        virtual ~QuantumComputation() = default;

        [[nodiscard]] QuantumComputation clone() const {
            auto qc              = QuantumComputation(nqubits);
            qc.nqubits           = nqubits;
            qc.nclassics         = nclassics;
            qc.nancillae         = nancillae;
            qc.maxControls       = maxControls;
            qc.name              = name;
            qc.qregs             = qregs;
            qc.cregs             = cregs;
            qc.ancregs           = ancregs;
            qc.initialLayout     = initialLayout;
            qc.outputPermutation = outputPermutation;
            qc.ancillary         = ancillary;
            qc.garbage           = garbage;
            qc.seed              = seed;
            qc.mt                = mt;
            qc.occuringVariables = occuringVariables;

            for (auto const& op: ops) {
                qc.ops.emplace_back<>(op->clone());
            }
            return qc;
        }

        [[nodiscard]] virtual std::size_t         getNops() const { return ops.size(); }
        [[nodiscard]] std::size_t                 getNqubits() const { return nqubits + nancillae; }
        [[nodiscard]] std::size_t                 getNancillae() const { return nancillae; }
        [[nodiscard]] std::size_t                 getNqubitsWithoutAncillae() const { return nqubits; }
        [[nodiscard]] std::size_t                 getNcbits() const { return nclassics; }
        [[nodiscard]] std::string                 getName() const { return name; }
        [[nodiscard]] const QuantumRegisterMap&   getQregs() const { return qregs; }
        [[nodiscard]] const ClassicalRegisterMap& getCregs() const { return cregs; }
        [[nodiscard]] const QuantumRegisterMap&   getANCregs() const { return ancregs; }
        [[nodiscard]] decltype(mt)&               getGenerator() { return mt; }

        void setName(const std::string& n) { name = n; }

        // physical qubits are used as keys, logical qubits as values
        Permutation initialLayout{};
        Permutation outputPermutation{};

        std::vector<bool> ancillary{};
        std::vector<bool> garbage{};

        [[nodiscard]] std::size_t getNindividualOps() const;
        [[nodiscard]] std::size_t getNsingleQubitOps() const;
        [[nodiscard]] std::size_t getDepth() const;

        [[nodiscard]] std::string                   getQubitRegister(Qubit physicalQubitIndex) const;
        [[nodiscard]] std::string                   getClassicalRegister(Bit classicalIndex) const;
        [[gnu::pure]] static Qubit                  getHighestLogicalQubitIndex(const Permutation& permutation);
        [[nodiscard]] Qubit                         getHighestLogicalQubitIndex() const { return getHighestLogicalQubitIndex(initialLayout); };
        [[nodiscard]] std::pair<std::string, Qubit> getQubitRegisterAndIndex(Qubit physicalQubitIndex) const;
        [[nodiscard]] std::pair<std::string, Bit>   getClassicalRegisterAndIndex(Bit classicalIndex) const;

        [[nodiscard]] Qubit                    getIndexFromQubitRegister(const std::pair<std::string, Qubit>& qubit) const;
        [[nodiscard]] Bit                      getIndexFromClassicalRegister(const std::pair<std::string, std::size_t>& clbit) const;
        [[nodiscard]] bool                     isIdleQubit(Qubit physicalQubit) const;
        [[nodiscard]] bool                     isLastOperationOnQubit(const const_iterator& opIt, const const_iterator& end) const;
        [[nodiscard, gnu::pure]] bool          physicalQubitIsAncillary(Qubit physicalQubitIndex) const;
        [[nodiscard]] bool                     logicalQubitIsAncillary(const Qubit logicalQubitIndex) const { return ancillary[logicalQubitIndex]; }
        void                                   setLogicalQubitAncillary(const Qubit logicalQubitIndex) { ancillary[logicalQubitIndex] = true; }
        [[nodiscard]] bool                     logicalQubitIsGarbage(const Qubit logicalQubitIndex) const { return garbage[logicalQubitIndex]; }
        void                                   setLogicalQubitGarbage(Qubit logicalQubitIndex);
        [[nodiscard]] const std::vector<bool>& getAncillary() const { return ancillary; }
        [[nodiscard]] const std::vector<bool>& getGarbage() const { return garbage; }

        /// checks whether the given logical qubit exists in the initial layout.
        /// \param logicalQubitIndex the logical qubit index to check
        /// \return whether the given logical qubit exists in the initial layout and to which physical qubit it is mapped
        [[nodiscard, gnu::pure]] std::pair<bool, std::optional<Qubit>> containsLogicalQubit(Qubit logicalQubitIndex) const;

        void i(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::I);
        }
        void i(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::I);
        }
        void i(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::I);
        }

        void h(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::H);
        }
        void h(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::H);
        }
        void h(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::H);
        }

        void x(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::X);
        }
        void x(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::X);
        }
        void x(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::X);
        }

        void y(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::Y);
        }
        void y(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::Y);
        }
        void y(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Y);
        }

        void z(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::Z);
        }
        void z(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::Z);
        }
        void z(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Z);
        }

        void s(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::S);
        }
        void s(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::S);
        }
        void s(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::S);
        }

        void sdag(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::Sdag);
        }
        void sdag(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::Sdag);
        }
        void sdag(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Sdag);
        }

        void t(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::T);
        }
        void t(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::T);
        }
        void t(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::T);
        }

        void tdag(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::Tdag);
        }
        void tdag(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::Tdag);
        }
        void tdag(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Tdag);
        }

        void v(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::V);
        }
        void v(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::V);
        }
        void v(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::V);
        }

        void vdag(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::Vdag);
        }
        void vdag(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::Vdag);
        }
        void vdag(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Vdag);
        }

        void u3(Qubit target, fp lambda, fp phi, fp theta) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::U3, lambda, phi, theta);
        }
        void u3(Qubit target, const Control& control, fp lambda, fp phi, fp theta) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::U3, lambda, phi, theta);
        }
        void u3(Qubit target, const Controls& controls, fp lambda, fp phi, fp theta) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::U3, lambda, phi, theta);
        }
        void u3(Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta);
        void u3(Qubit target, const Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta);
        void u3(Qubit target, const Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta);

        void u2(Qubit target, fp lambda, fp phi) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::U2, lambda, phi);
        }
        void u2(Qubit target, const Control& control, fp lambda, fp phi) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::U2, lambda, phi);
        }
        void u2(Qubit target, const Controls& controls, fp lambda, fp phi) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::U2, lambda, phi);
        }
        void u2(Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi);
        void u2(Qubit target, const Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi);
        void u2(Qubit target, const Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi);

        void phase(Qubit target, fp lambda) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::Phase, lambda);
        }
        void phase(Qubit target, const Control& control, fp lambda) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::Phase, lambda);
        }
        void phase(Qubit target, const Controls& controls, fp lambda) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Phase, lambda);
        }
        void phase(Qubit target, const SymbolOrNumber& lambda);
        void phase(Qubit target, const Control& control, const SymbolOrNumber& lambda);
        void phase(Qubit target, const Controls& controls, const SymbolOrNumber& lambda);

        void sx(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::SX);
        }
        void sx(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::SX);
        }
        void sx(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::SX);
        }

        void sxdag(Qubit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::SXdag);
        }
        void sxdag(Qubit target, const Control& control) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::SXdag);
        }
        void sxdag(Qubit target, const Controls& controls) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::SXdag);
        }

        void rx(Qubit target, fp lambda) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::RX, lambda);
        }
        void rx(Qubit target, const Control& control, fp lambda) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::RX, lambda);
        }
        void rx(Qubit target, const Controls& controls, fp lambda) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RX, lambda);
        }
        void rx(Qubit target, const SymbolOrNumber& lambda);
        void rx(Qubit target, const Control& control, const SymbolOrNumber& lambda);
        void rx(Qubit target, const Controls& controls, const SymbolOrNumber& lambda);

        void ry(Qubit target, fp lambda) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::RY, lambda);
        }
        void ry(Qubit target, const Control& control, fp lambda) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::RY, lambda);
        }
        void ry(Qubit target, const Controls& controls, fp lambda) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RY, lambda);
        }

        void ry(Qubit target, const SymbolOrNumber& lambda);
        void ry(Qubit target, const Control& control, const SymbolOrNumber& lambda);
        void ry(Qubit target, const Controls& controls, const SymbolOrNumber& lambda);

        void rz(Qubit target, fp lambda) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(getNqubits(), target, qc::RZ, lambda);
        }
        void rz(Qubit target, const Control& control, fp lambda) {
            checkQubitRange(target, control);
            emplace_back<StandardOperation>(getNqubits(), control, target, qc::RZ, lambda);
        }
        void rz(Qubit target, const Controls& controls, fp lambda) {
            checkQubitRange(target, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RZ, lambda);
        }

        void rz(Qubit target, const SymbolOrNumber& lambda);
        void rz(Qubit target, const Control& control, const SymbolOrNumber& lambda);
        void rz(Qubit target, const Controls& controls, const SymbolOrNumber& lambda);

        void swap(Qubit target0, Qubit target1) {
            checkQubitRange(target0, target1);
            emplace_back<StandardOperation>(getNqubits(), Controls{}, target0, target1, qc::SWAP);
        }
        void swap(Qubit target0, Qubit target1, const Control& control) {
            checkQubitRange(target0, target1, control);
            emplace_back<StandardOperation>(getNqubits(), Controls{control}, target0, target1, qc::SWAP);
        }
        void swap(Qubit target0, Qubit target1, const Controls& controls) {
            checkQubitRange(target0, target1, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::SWAP);
        }

        void iswap(Qubit target0, Qubit target1) {
            checkQubitRange(target0, target1);
            emplace_back<StandardOperation>(getNqubits(), Controls{}, target0, target1, qc::iSWAP);
        }
        void iswap(Qubit target0, Qubit target1, const Control& control) {
            checkQubitRange(target0, target1, control);
            emplace_back<StandardOperation>(getNqubits(), Controls{control}, target0, target1, qc::iSWAP);
        }
        void iswap(Qubit target0, Qubit target1, const Controls& controls) {
            checkQubitRange(target0, target1, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::iSWAP);
        }

        void peres(Qubit target0, Qubit target1) {
            checkQubitRange(target0, target1);
            emplace_back<StandardOperation>(getNqubits(), Controls{}, target0, target1, qc::Peres);
        }
        void peres(Qubit target0, Qubit target1, const Control& control) {
            checkQubitRange(target0, target1, control);
            emplace_back<StandardOperation>(getNqubits(), Controls{control}, target0, target1, qc::Peres);
        }
        void peres(Qubit target0, Qubit target1, const Controls& controls) {
            checkQubitRange(target0, target1, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::Peres);
        }

        void peresdag(Qubit target0, Qubit target1) {
            checkQubitRange(target0, target1);
            emplace_back<StandardOperation>(getNqubits(), Controls{}, target0, target1, qc::Peresdag);
        }
        void peresdag(Qubit target0, Qubit target1, const Control& control) {
            checkQubitRange(target0, target1, control);
            emplace_back<StandardOperation>(getNqubits(), Controls{control}, target0, target1, qc::Peresdag);
        }
        void peresdag(Qubit target0, Qubit target1, const Controls& controls) {
            checkQubitRange(target0, target1, controls);
            emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::Peresdag);
        }

        void measure(Qubit qubit, std::size_t clbit) {
            checkQubitRange(qubit);
            emplace_back<NonUnitaryOperation>(getNqubits(), qubit, clbit);
        }

        void measure(Qubit qubit, const std::pair<std::string, Bit>& clbit) {
            checkQubitRange(qubit);
            if (const auto cRegister = cregs.find(clbit.first); cRegister != cregs.end()) {
                if (clbit.second >= cRegister->second.second) {
                    std::cerr << "The classical register \"" << clbit.first << "\" is too small!" << std::endl;
                }
                emplace_back<NonUnitaryOperation>(getNqubits(), qubit, cRegister->second.first + clbit.second);

            } else {
                std::cerr << "The classical register \"" << clbit.first << "\" does not exist!" << std::endl;
            }
        }

        void measure(const std::vector<Qubit>& qubitRegister,
                     const std::vector<Bit>&   classicalRegister) {
            checkQubitRange(qubitRegister);
            emplace_back<NonUnitaryOperation>(getNqubits(), qubitRegister,
                                              classicalRegister);
        }

        void reset(Qubit target) {
            checkQubitRange(target);
            emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<Qubit>{target}, qc::Reset);
        }
        void reset(const std::vector<Qubit>& targets) {
            checkQubitRange(targets);
            emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Reset);
        }

        void barrier(Qubit target) {
            checkQubitRange(target);
            emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<Qubit>{target}, qc::Barrier);
        }
        void barrier(const std::vector<Qubit>& targets) {
            checkQubitRange(targets);
            emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Barrier);
        }

        void classicControlled(const OpType op, const Qubit target, const ClassicalRegister& controlRegister, const std::uint64_t expectedValue = 1U, const fp lambda = 0., const fp phi = 0., const fp theta = 0.) {
            classicControlled(op, target, Controls{}, controlRegister, expectedValue, lambda, phi, theta);
        }
        void classicControlled(const OpType op, const Qubit target, const Control control, const ClassicalRegister& controlRegister, const std::uint64_t expectedValue = 1U, const fp lambda = 0., const fp phi = 0., const fp theta = 0.) {
            classicControlled(op, target, Controls{control}, controlRegister, expectedValue, lambda, phi, theta);
        }
        void classicControlled(const OpType op, const Qubit target, const Controls& controls, const ClassicalRegister& controlRegister, const std::uint64_t expectedValue = 1U, const fp lambda = 0., const fp phi = 0., const fp theta = 0.) {
            checkQubitRange(target, controls);
            std::unique_ptr<Operation> gate = std::make_unique<StandardOperation>(getNqubits(), controls, target, op, lambda, phi, theta);
            emplace_back<ClassicControlledOperation>(std::move(gate), controlRegister, expectedValue);
        }

        /// strip away qubits with no operations applied to them and which do not pop up in the output permutation
        /// \param force if true, also strip away idle qubits occurring in the output permutation
        void stripIdleQubits(bool force = false, bool reduceIOpermutations = true);

        void import(const std::string& filename);
        void import(const std::string& filename, Format format);
        void import(std::istream& is, Format format) {
            import(std::move(is), format);
        }
        void import(std::istream&& is, Format format);
        void initializeIOMapping();
        // append measurements to the end of the circuit according to the tracked output permutation
        void appendMeasurementsAccordingToOutputPermutation(const std::string& registerName = "c");
        // search for current position of target value in map and afterwards exchange it with the value at new position
        static void findAndSWAP(Qubit targetValue, Qubit newPosition, Permutation& map) {
            for (const auto& q: map) {
                if (q.second == targetValue) {
                    std::swap(map.at(newPosition), map.at(q.first));
                    break;
                }
            }
        }

        // this function augments a given circuit by additional registers
        void addQubitRegister(std::size_t, const std::string& regName = "q");
        void addClassicalRegister(std::size_t nc, const std::string& regName = "c");
        void addAncillaryRegister(std::size_t nq, const std::string& regName = "anc");
        // a function to combine all quantum registers (qregs and ancregs) into a single register (useful for circuits mapped to a device)
        void unifyQuantumRegisters(const std::string& regName = "q");

        // removes a specific logical qubit and returns the index of the physical qubit in the initial layout
        // as well as the index of the removed physical qubit's output permutation
        // i.e., initialLayout[physical_qubit] = logical_qubit and outputPermutation[physicalQubit] = output_qubit
        std::pair<Qubit, std::optional<Qubit>> removeQubit(Qubit logicalQubitIndex);

        // adds physical qubit as ancillary qubit and gives it the appropriate output mapping
        void addAncillaryQubit(Qubit physicalQubitIndex, std::optional<Qubit> outputQubitIndex);
        // try to add logical qubit to circuit and assign it to physical qubit with certain output permutation value
        void addQubit(Qubit logicalQubitIndex, Qubit physicalQubitIndex, std::optional<Qubit> outputQubitIndex);

        void updateMaxControls(const std::size_t ncontrols) {
            maxControls = std::max(ncontrols, maxControls);
        }

        void instantiate(const VariableAssignment& assignment);

        void addVariable(const SymbolOrNumber& expr);

        template<typename... Vars>
        void addVariables(const Vars&... vars) {
            (addVariable(vars), ...);
        }

        [[nodiscard]] bool isVariableFree() const {
            return std::all_of(ops.begin(), ops.end(), [](const auto& op) { return !op->isSymbolicOperation(); });
        }

        [[nodiscard]] const std::unordered_set<sym::Variable>& getVariables() const {
            return occuringVariables;
        }

        /**
         * printing
         */
        virtual std::ostream& print(std::ostream& os) const;

        friend std::ostream& operator<<(std::ostream& os, const QuantumComputation& qc) { return qc.print(os); }

        static void printBin(std::size_t n, std::stringstream& ss);

        virtual std::ostream& printStatistics(std::ostream& os) const;

        std::ostream& printRegisters(std::ostream& os = std::cout) const;

        static std::ostream& printPermutation(const Permutation& permutation, std::ostream& os = std::cout);

        virtual void dump(const std::string& filename, Format format);
        virtual void dump(const std::string& filename);
        virtual void dump(std::ostream& of, Format format) {
            dump(std::move(of), format);
        }
        virtual void dump(std::ostream&& of, Format format);
        virtual void dumpOpenQASM(std::ostream& of);

        // this convenience method allows to turn a circuit into a compound operation.
        std::unique_ptr<CompoundOperation> asCompoundOperation() {
            return std::make_unique<CompoundOperation>(getNqubits(), std::move(ops));
        }

        // this convenience method allows to turn a circuit into an operation.
        std::unique_ptr<Operation> asOperation() {
            if (ops.empty()) {
                return {};
            }
            if (ops.size() == 1) {
                auto op = std::move(ops.front());
                ops.clear();
                return op;
            }
            return asCompoundOperation();
        }

        virtual void reset() {
            ops.clear();
            nqubits   = 0;
            nclassics = 0;
            nancillae = 0;
            qregs.clear();
            cregs.clear();
            ancregs.clear();
            initialLayout.clear();
            outputPermutation.clear();
        }

        /**
         * Pass-Through
         */

        // Iterators (pass-through)
        auto               begin() noexcept { return ops.begin(); }
        [[nodiscard]] auto begin() const noexcept { return ops.begin(); }
        [[nodiscard]] auto cbegin() const noexcept { return ops.cbegin(); }
        auto               end() noexcept { return ops.end(); }
        [[nodiscard]] auto end() const noexcept { return ops.end(); }
        [[nodiscard]] auto cend() const noexcept { return ops.cend(); }
        auto               rbegin() noexcept { return ops.rbegin(); }
        [[nodiscard]] auto rbegin() const noexcept { return ops.rbegin(); }
        [[nodiscard]] auto crbegin() const noexcept { return ops.crbegin(); }
        auto               rend() noexcept { return ops.rend(); }
        [[nodiscard]] auto rend() const noexcept { return ops.rend(); }
        [[nodiscard]] auto crend() const noexcept { return ops.crend(); }

        // Capacity (pass-through)
        [[nodiscard]] bool        empty() const noexcept { return ops.empty(); }
        [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }
        [[nodiscard]] std::size_t max_size() const noexcept { return ops.max_size(); } // NOLINT (readability-identifier-naming)
        [[nodiscard]] std::size_t capacity() const noexcept { return ops.capacity(); }

        void reserve(const std::size_t newCap) { ops.reserve(newCap); }
        void shrink_to_fit() { ops.shrink_to_fit(); } // NOLINT (readability-identifier-naming)

        // Modifiers (pass-through)
        void     clear() noexcept { ops.clear(); }
        void     pop_back() { return ops.pop_back(); } // NOLINT (readability-identifier-naming)
        void     resize(std::size_t count) { ops.resize(count); }
        iterator erase(const_iterator pos) { return ops.erase(pos); }
        iterator erase(const_iterator first, const_iterator last) { return ops.erase(first, last); }

        template<class T>
        void push_back(const T& op) { // NOLINT (readability-identifier-naming)
            if (!ops.empty() && !op.isControlled() && !ops.back()->isControlled()) {
                std::cerr << op.getName() << std::endl;
            }

            ops.push_back(std::make_unique<T>(op));
        }

        template<class T, class... Args>
        void emplace_back(Args&&... args) { // NOLINT (readability-identifier-naming)
            ops.emplace_back(std::make_unique<T>(args...));
        }

        template<class T>
        void emplace_back(std::unique_ptr<T>& op) { // NOLINT (readability-identifier-naming)
            ops.emplace_back(std::move(op));
        }

        template<class T>
        void emplace_back(std::unique_ptr<T>&& op) { // NOLINT (readability-identifier-naming)
            ops.emplace_back(std::move(op));
        }

        template<class T>
        iterator insert(const_iterator pos, T&& op) { return ops.insert(pos, std::forward<T>(op)); }

        [[nodiscard]] const auto& at(const std::size_t i) const { return ops.at(i); }
        [[nodiscard]] const auto& front() const { return ops.front(); }
        [[nodiscard]] const auto& back() const { return ops.back(); }
    };
} // namespace qc
