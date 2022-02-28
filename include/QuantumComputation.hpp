/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QUANTUMCOMPUTATION_H
#define QFR_QUANTUMCOMPUTATION_H

#include "Definitions.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/StandardOperation.hpp"
#include "parsers/qasm_parser/Parser.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace qc {
    static constexpr char DEFAULT_QREG[2]{"q"};
    static constexpr char DEFAULT_CREG[2]{"c"};
    static constexpr char DEFAULT_ANCREG[4]{"anc"};
    static constexpr char DEFAULT_MCTREG[4]{"mct"};

    class CircuitOptimizer;

    class QuantumComputation {
        friend class CircuitOptimizer;

    protected:
        std::vector<std::unique_ptr<Operation>> ops{};
        dd::QubitCount                          nqubits      = 0;
        std::size_t                             nclassics    = 0;
        dd::QubitCount                          nancillae    = 0;
        dd::QubitCount                          max_controls = 0;
        std::string                             name;

        // reg[reg_name] = {start_index, length}
        QuantumRegisterMap   qregs{};
        ClassicalRegisterMap cregs{};
        QuantumRegisterMap   ancregs{};

        std::mt19937_64 mt;
        std::size_t     seed = 0;

        void importOpenQASM(std::istream& is);
        void importReal(std::istream& is);
        int  readRealHeader(std::istream& is);
        void readRealGateDescriptions(std::istream& is, int line);
        void importTFC(std::istream& is);
        int  readTFCHeader(std::istream& is, std::map<std::string, dd::Qubit>& varMap);
        void readTFCGateDescriptions(std::istream& is, int line, std::map<std::string, dd::Qubit>& varMap);
        void importQC(std::istream& is);
        int  readQCHeader(std::istream& is, std::map<std::string, dd::Qubit>& varMap);
        void readQCGateDescriptions(std::istream& is, int line, std::map<std::string, dd::Qubit>& varMap);
        void importGRCS(std::istream& is);

        template<class RegisterType>
        static void printSortedRegisters(const RegisterMap<RegisterType>& regmap, const std::string& identifier, std::ostream& of) {
            // sort regs by start index
            std::map<decltype(RegisterType::first), std::pair<std::string, RegisterType>> sortedRegs{};
            for (const auto& reg: regmap) {
                sortedRegs.insert({reg.second.first, reg});
            }

            for (const auto& reg: sortedRegs) {
                of << identifier << " " << reg.second.first << "[" << static_cast<std::size_t>(reg.second.second.second) << "];" << std::endl;
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
        static void createRegisterArray(const RegisterMap<RegisterType>& regs, RegisterNames& regnames, decltype(RegisterType::second) defaultnumber, const char* defaultname) {
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
                        ss << reg.second.first << "[" << static_cast<std::size_t>(i) << "]";
                        regnames.push_back(std::make_pair(reg.second.first, ss.str()));
                        ss.str(std::string());
                    }
                }
            } else {
                for (decltype(RegisterType::second) i = 0; i < defaultnumber; i++) {
                    ss << defaultname << "[" << static_cast<std::size_t>(i) << "]";
                    regnames.push_back(std::make_pair(defaultname, ss.str()));
                    ss.str(std::string());
                }
            }
        }

        [[nodiscard]] std::size_t getSmallestAncillary() const {
            for (std::size_t i = 0; i < ancillary.size(); ++i) {
                if (ancillary[i])
                    return i;
            }
            return ancillary.size();
        }

        [[nodiscard]] std::size_t getSmallestGarbage() const {
            for (std::size_t i = 0; i < garbage.size(); ++i) {
                if (garbage[i])
                    return i;
            }
            return garbage.size();
        }
        [[nodiscard]] bool isLastOperationOnQubit(const decltype(ops.cbegin())& opIt) const {
            const auto end = ops.cend();
            return isLastOperationOnQubit(opIt, end);
        }

    public:
        QuantumComputation() = default;
        explicit QuantumComputation(std::size_t nqubits, std::size_t seed = 0):
            seed(seed) {
            addQubitRegister(nqubits);
            addClassicalRegister(nqubits);
            if (seed != 0) {
                mt.seed(seed);
            } else {
                // create and properly seed rng
                std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
                std::random_device                                                    rd;
                std::generate(std::begin(random_data), std::end(random_data), [&rd]() { return rd(); });
                std::seed_seq seeds(std::begin(random_data), std::end(random_data));
                mt.seed(seeds);
            }
        }
        explicit QuantumComputation(const std::string& filename, std::size_t seed = 0):
            seed(seed) {
            import(filename);
            if (seed != 0) {
                mt.seed(seed);
            } else {
                // create and properly seed rng
                std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
                std::random_device                                                    rd;
                std::generate(std::begin(random_data), std::end(random_data), [&rd]() { return rd(); });
                std::seed_seq seeds(std::begin(random_data), std::end(random_data));
                mt.seed(seeds);
            }
        }
        QuantumComputation(const QuantumComputation& qc)     = delete;
        QuantumComputation(QuantumComputation&& qc) noexcept = default;
        QuantumComputation& operator=(const QuantumComputation& qc) = delete;
        QuantumComputation& operator=(QuantumComputation&& qc) noexcept = default;
        virtual ~QuantumComputation()                                   = default;

        [[nodiscard]] QuantumComputation clone() const {
            auto qc              = QuantumComputation(nqubits);
            qc.nqubits           = nqubits;
            qc.nclassics         = nclassics;
            qc.nancillae         = nancillae;
            qc.max_controls      = max_controls;
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

            for (auto const& op: ops) {
                qc.ops.emplace_back<>(op->clone());
            }
            return qc;
        }

        [[nodiscard]] virtual std::size_t         getNops() const { return ops.size(); }
        [[nodiscard]] dd::QubitCount              getNqubits() const { return nqubits + nancillae; }
        [[nodiscard]] dd::QubitCount              getNancillae() const { return nancillae; }
        [[nodiscard]] dd::QubitCount              getNqubitsWithoutAncillae() const { return nqubits; }
        [[nodiscard]] std::size_t                 getNcbits() const { return nclassics; }
        [[nodiscard]] std::string                 getName() const { return name; }
        [[nodiscard]] const QuantumRegisterMap&   getQregs() const { return qregs; }
        [[nodiscard]] const ClassicalRegisterMap& getCregs() const { return cregs; }
        [[nodiscard]] const QuantumRegisterMap&   getANCregs() const { return ancregs; }

        void setName(const std::string& n) { name = n; }

        // initialLayout[physical_qubit] = logical_qubit
        Permutation initialLayout{};
        Permutation outputPermutation{};

        std::vector<bool> ancillary{};
        std::vector<bool> garbage{};

        [[nodiscard]] std::size_t getNindividualOps() const;

        [[nodiscard]] std::string                         getQubitRegister(dd::Qubit physicalQubitIndex) const;
        [[nodiscard]] std::string                         getClassicalRegister(std::size_t classicalIndex) const;
        static dd::Qubit                                  getHighestLogicalQubitIndex(const Permutation& map);
        [[nodiscard]] dd::Qubit                           getHighestLogicalQubitIndex() const { return getHighestLogicalQubitIndex(initialLayout); };
        [[nodiscard]] std::pair<std::string, dd::Qubit>   getQubitRegisterAndIndex(dd::Qubit physicalQubitIndex) const;
        [[nodiscard]] std::pair<std::string, std::size_t> getClassicalRegisterAndIndex(std::size_t classicalIndex) const;

        [[nodiscard]] dd::Qubit   getIndexFromQubitRegister(const std::pair<std::string, dd::Qubit>& qubit) const;
        [[nodiscard]] std::size_t getIndexFromClassicalRegister(const std::pair<std::string, std::size_t>& clbit) const;
        [[nodiscard]] bool        isIdleQubit(dd::Qubit physicalQubit) const;
        [[nodiscard]] bool        isLastOperationOnQubit(const decltype(ops.cbegin())& opIt, const decltype(ops.cend())& end) const;
        [[nodiscard]] bool        physicalQubitIsAncillary(dd::Qubit physicalQubitIndex) const;
        [[nodiscard]] bool        logicalQubitIsAncillary(dd::Qubit logicalQubitIndex) const { return ancillary[logicalQubitIndex]; }
        void                      setLogicalQubitAncillary(dd::Qubit logicalQubitIndex) { ancillary[logicalQubitIndex] = true; }
        [[nodiscard]] bool        logicalQubitIsGarbage(dd::Qubit logicalQubitIndex) const { return garbage[logicalQubitIndex]; }
        void                      setLogicalQubitGarbage(dd::Qubit logicalQubitIndex);
        MatrixDD                  createInitialMatrix(std::unique_ptr<dd::Package>& dd) const; // creates identity matrix, which is reduced with respect to the ancillary qubits

        void i(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::I); }
        void i(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::I); }
        void i(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::I); }

        void h(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::H); }
        void h(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::H); }
        void h(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::H); }

        void x(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::X); }
        void x(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::X); }
        void x(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::X); }

        void y(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::Y); }
        void y(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::Y); }
        void y(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Y); }

        void z(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::Z); }
        void z(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::Z); }
        void z(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Z); }

        void s(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::S); }
        void s(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::S); }
        void s(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::S); }

        void sdag(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::Sdag); }
        void sdag(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::Sdag); }
        void sdag(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Sdag); }

        void t(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::T); }
        void t(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::T); }
        void t(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::T); }

        void tdag(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::Tdag); }
        void tdag(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::Tdag); }
        void tdag(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Tdag); }

        void v(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::V); }
        void v(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::V); }
        void v(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::V); }

        void vdag(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::Vdag); }
        void vdag(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::Vdag); }
        void vdag(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Vdag); }

        void u3(dd::Qubit target, dd::fp lambda, dd::fp phi, dd::fp theta) { emplace_back<StandardOperation>(getNqubits(), target, qc::U3, lambda, phi, theta); }
        void u3(dd::Qubit target, const dd::Control& control, dd::fp lambda, dd::fp phi, dd::fp theta) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::U3, lambda, phi, theta); }
        void u3(dd::Qubit target, const dd::Controls& controls, dd::fp lambda, dd::fp phi, dd::fp theta) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::U3, lambda, phi, theta); }

        void u2(dd::Qubit target, dd::fp lambda, dd::fp phi) { emplace_back<StandardOperation>(getNqubits(), target, qc::U2, lambda, phi); }
        void u2(dd::Qubit target, const dd::Control& control, dd::fp lambda, dd::fp phi) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::U2, lambda, phi); }
        void u2(dd::Qubit target, const dd::Controls& controls, dd::fp lambda, dd::fp phi) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::U2, lambda, phi); }

        void phase(dd::Qubit target, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), target, qc::Phase, lambda); }
        void phase(dd::Qubit target, const dd::Control& control, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::Phase, lambda); }
        void phase(dd::Qubit target, const dd::Controls& controls, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::Phase, lambda); }

        void sx(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::SX); }
        void sx(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::SX); }
        void sx(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::SX); }

        void sxdag(dd::Qubit target) { emplace_back<StandardOperation>(getNqubits(), target, qc::SXdag); }
        void sxdag(dd::Qubit target, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::SXdag); }
        void sxdag(dd::Qubit target, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::SXdag); }

        void rx(dd::Qubit target, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), target, qc::RX, lambda); }
        void rx(dd::Qubit target, const dd::Control& control, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::RX, lambda); }
        void rx(dd::Qubit target, const dd::Controls& controls, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RX, lambda); }

        void ry(dd::Qubit target, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), target, qc::RY, lambda); }
        void ry(dd::Qubit target, const dd::Control& control, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::RY, lambda); }
        void ry(dd::Qubit target, const dd::Controls& controls, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RY, lambda); }

        void rz(dd::Qubit target, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), target, qc::RZ, lambda); }
        void rz(dd::Qubit target, const dd::Control& control, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), control, target, qc::RZ, lambda); }
        void rz(dd::Qubit target, const dd::Controls& controls, dd::fp lambda) { emplace_back<StandardOperation>(getNqubits(), controls, target, qc::RZ, lambda); }

        void swap(dd::Qubit target0, dd::Qubit target1) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::SWAP); }
        void swap(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::SWAP); }
        void swap(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::SWAP); }

        void iswap(dd::Qubit target0, dd::Qubit target1) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::iSWAP); }
        void iswap(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::iSWAP); }
        void iswap(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::iSWAP); }

        void peres(dd::Qubit target0, dd::Qubit target1) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::Peres); }
        void peres(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::Peres); }
        void peres(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::Peres); }

        void peresdag(dd::Qubit target0, dd::Qubit target1) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{}, target0, target1, qc::Peresdag); }
        void peresdag(dd::Qubit target0, dd::Qubit target1, const dd::Control& control) { emplace_back<StandardOperation>(getNqubits(), dd::Controls{control}, target0, target1, qc::Peresdag); }
        void peresdag(dd::Qubit target0, dd::Qubit target1, const dd::Controls& controls) { emplace_back<StandardOperation>(getNqubits(), controls, target0, target1, qc::Peresdag); }

        void measure(dd::Qubit qubit, std::size_t clbit) { emplace_back<NonUnitaryOperation>(getNqubits(), qubit, clbit); }
        void measure(const std::vector<dd::Qubit>& qubitRegister, const std::vector<std::size_t>& classicalRegister) { emplace_back<NonUnitaryOperation>(getNqubits(), qubitRegister, classicalRegister); }

        void reset(dd::Qubit target) { emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<dd::Qubit>{target}, qc::Reset); }
        void reset(const std::vector<dd::Qubit>& targets) { emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Reset); }

        void barrier(dd::Qubit target) { emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<dd::Qubit>{target}, qc::Barrier); }
        void barrier(const std::vector<dd::Qubit>& targets) { emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Barrier); }

        /// strip away qubits with no operations applied to them and which do not pop up in the output permutation
        /// \param force if true, also strip away idle qubits occurring in the output permutation
        void stripIdleQubits(bool force = false, bool reduceIOpermutations = true);
        // apply swaps 'on' DD in order to change 'from' to 'to'
        // where |from| >= |to|
        template<class DDType>
        static void changePermutation(DDType& on, Permutation& from, const Permutation& to, std::unique_ptr<dd::Package>& dd, bool regular = true) {
            assert(from.size() >= to.size());

            // iterate over (k,v) pairs of second permutation
            for (const auto& [i, goal]: to) {
                // search for key in the first map
                auto it = from.find(i);
                if (it == from.end()) {
                    throw QFRException("[changePermutation] Key " + std::to_string(it->first) + " was not found in first permutation. This should never happen.");
                }
                auto current = it->second;

                // permutations agree for this key value
                if (current == goal) continue;

                // search for goal value in first permutation
                dd::Qubit j = 0;
                for (const auto& [key, value]: from) {
                    if (value == goal) {
                        j = key;
                        break;
                    }
                }

                // swap i and j
                auto saved = on;
                if constexpr (std::is_same_v<DDType, VectorDD>) {
                    on = dd->multiply(dd->makeSWAPDD(on.p->v + 1, {}, from.at(i), from.at(j)), on);
                } else {
                    // the regular flag only has an effect on matrix DDs
                    if (regular) {
                        on = dd->multiply(dd->makeSWAPDD(on.p->v + 1, {}, from.at(i), from.at(j)), on);
                    } else {
                        on = dd->multiply(on, dd->makeSWAPDD(on.p->v + 1, {}, from.at(i), from.at(j)));
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

        void import(const std::string& filename);
        void import(const std::string& filename, Format format);
        void import(std::istream& is, Format format) {
            import(std::move(is), format);
        }
        void import(std::istream&& is, Format format);
        void initializeIOMapping();
        // search for current position of target value in map and afterwards exchange it with the value at new position
        static void findAndSWAP(dd::Qubit targetValue, dd::Qubit newPosition, Permutation& map) {
            for (const auto& q: map) {
                if (q.second == targetValue) {
                    std::swap(map.at(newPosition), map.at(q.first));
                    break;
                }
            }
        }

        // this function augments a given circuit by additional registers
        void addQubitRegister(std::size_t, const char* reg_name = DEFAULT_QREG);
        void addClassicalRegister(std::size_t nc, const char* reg_name = DEFAULT_CREG);
        void addAncillaryRegister(std::size_t nq, const char* reg_name = DEFAULT_ANCREG);

        // removes the a specific logical qubit and returns the index of the physical qubit in the initial layout
        // as well as the index of the removed physical qubit's output permutation
        // i.e., initialLayout[physical_qubit] = logical_qubit and outputPermutation[physicalQubit] = output_qubit
        std::pair<dd::Qubit, dd::Qubit> removeQubit(dd::Qubit logical_qubit_index);

        // adds physical qubit as ancillary qubit and gives it the appropriate output mapping
        void addAncillaryQubit(dd::Qubit physical_qubit_index, dd::Qubit output_qubit_index);
        // try to add logical qubit to circuit and assign it to physical qubit with certain output permutation value
        void addQubit(dd::Qubit logical_qubit_index, dd::Qubit physical_qubit_index, dd::Qubit output_qubit_index);

        void updateMaxControls(dd::QubitCount ncontrols) {
            max_controls = std::max(ncontrols, max_controls);
        }

        virtual VectorDD                           simulate(const VectorDD& in, std::unique_ptr<dd::Package>& dd) const;
        virtual std::map<std::string, std::size_t> simulate(const VectorDD& in, std::unique_ptr<dd::Package>& dd, std::size_t shots);
        virtual MatrixDD                           buildFunctionality(std::unique_ptr<dd::Package>& dd) const;
        virtual MatrixDD                           buildFunctionalityRecursive(std::unique_ptr<dd::Package>& dd) const;
        virtual bool                               buildFunctionalityRecursive(std::size_t depth, std::size_t opIdx, std::stack<MatrixDD>& s, Permutation& permutation, std::unique_ptr<dd::Package>& dd) const;

        virtual void extractProbabilityVector(const VectorDD& in, dd::ProbabilityVector& probVector, std::unique_ptr<dd::Package>& dd);
        virtual void extractProbabilityVectorRecursive(const VectorDD& currentState, decltype(ops.begin()) currentIt, std::map<std::size_t, char> measurements, dd::fp commonFactor, dd::ProbabilityVector& probVector, std::unique_ptr<dd::Package>& dd);
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
        virtual void dumpTensorNetwork(std::ostream& of) const;

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
        [[nodiscard]] bool   empty() const noexcept { return ops.empty(); }
        [[nodiscard]] size_t size() const noexcept { return ops.size(); }
        [[nodiscard]] size_t max_size() const noexcept { return ops.max_size(); }
        [[nodiscard]] size_t capacity() const noexcept { return ops.capacity(); }

        void reserve(size_t new_cap) { ops.reserve(new_cap); }
        void shrink_to_fit() { ops.shrink_to_fit(); }

        // Modifiers (pass-through)
        void                                              clear() noexcept { ops.clear(); }
        void                                              pop_back() { return ops.pop_back(); }
        void                                              resize(size_t count) { ops.resize(count); }
        std::vector<std::unique_ptr<Operation>>::iterator erase(std::vector<std::unique_ptr<Operation>>::const_iterator pos) { return ops.erase(pos); }
        std::vector<std::unique_ptr<Operation>>::iterator erase(std::vector<std::unique_ptr<Operation>>::const_iterator first, std::vector<std::unique_ptr<Operation>>::const_iterator last) { return ops.erase(first, last); }

        template<class T>
        void push_back(const T& op) {
            if (!ops.empty() && !op.isControlled() && !ops.back()->isControlled()) {
                std::cerr << op.getName() << std::endl;
            }

            ops.push_back(std::make_unique<T>(op));
        }

        template<class T, class... Args>
        void emplace_back(Args&&... args) {
            ops.emplace_back(std::make_unique<T>(args...));
        }

        template<class T>
        void emplace_back(std::unique_ptr<T>& op) {
            ops.emplace_back(std::move(op));
        }

        template<class T>
        std::vector<std::unique_ptr<Operation>>::iterator insert(std::vector<std::unique_ptr<Operation>>::const_iterator pos, T&& op) { return ops.insert(pos, std::forward<T>(op)); }

        [[nodiscard]] const auto& at(std::size_t i) const { return ops.at(i); }
    };
} // namespace qc
#endif //QFR_QUANTUMCOMPUTATION_H
