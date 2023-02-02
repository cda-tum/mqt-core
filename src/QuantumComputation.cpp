/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

#include <cassert>

namespace qc {

    /***
     * Public Methods
     ***/
    std::size_t QuantumComputation::getNindividualOps() const {
        std::size_t nops = 0;
        for (const auto& op: ops) {
            if (op->isCompoundOperation()) {
                auto&& comp = dynamic_cast<CompoundOperation*>(op.get());
                nops += comp->size();
            } else {
                ++nops;
            }
        }

        return nops;
    }

    std::size_t QuantumComputation::getNsingleQubitOps() const {
        std::size_t nops = 0;
        for (const auto& op: ops) {
            if (!op->isUnitary()) {
                continue;
            }

            if (op->isCompoundOperation()) {
                const auto* const comp = dynamic_cast<const CompoundOperation*>(op.get());
                for (const auto& subop: *comp) {
                    if (subop->isUnitary() && !subop->isControlled() && subop->getNtargets() == 1U) {
                        ++nops;
                    }
                }
            } else {
                if (!op->isControlled() && op->getNtargets() == 1U) {
                    ++nops;
                }
            }
        }
        return nops;
    }

    std::size_t QuantumComputation::getDepth() const {
        if (empty()) {
            return 0U;
        }

        std::vector<std::size_t> depths(getNqubits(), 0U);
        for (const auto& op: ops) {
            op->addDepthContribution(depths);
        }

        return *std::max_element(depths.begin(), depths.end());
    }

    void QuantumComputation::import(const std::string& filename) {
        const std::size_t dot       = filename.find_last_of('.');
        std::string       extension = filename.substr(dot + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) { return ::tolower(ch); });
        if (extension == "real") {
            import(filename, Format::Real);
        } else if (extension == "qasm") {
            import(filename, Format::OpenQASM);
        } else if (extension == "txt") {
            import(filename, Format::GRCS);
        } else if (extension == "tfc") {
            import(filename, Format::TFC);
        } else if (extension == "qc") {
            import(filename, Format::QC);
        } else {
            throw QFRException("[import] extension " + extension + " not recognized");
        }
    }

    void QuantumComputation::import(const std::string& filename, Format format) {
        const std::size_t slash = filename.find_last_of('/');
        const std::size_t dot   = filename.find_last_of('.');
        name                    = filename.substr(slash + 1, dot - slash - 1);

        auto ifs = std::ifstream(filename);
        if (ifs.good()) {
            import(ifs, format);
        } else {
            throw QFRException("[import] Error processing input stream: " + name);
        }
    }

    void QuantumComputation::import(std::istream&& is, Format format) {
        // reset circuit before importing
        reset();

        switch (format) {
            case Format::Real:
                importReal(is);
                break;
            case Format::OpenQASM:
                updateMaxControls(2);
                importOpenQASM(is);
                break;
            case Format::GRCS:
                importGRCS(is);
                break;
            case Format::TFC:
                importTFC(is);
                break;
            case Format::QC:
                importQC(is);
                break;
            default:
                throw QFRException("[import] format not recognized");
        }

        // initialize the initial layout and output permutation
        initializeIOMapping();
    }

    void QuantumComputation::initializeIOMapping() {
        // if no initial layout was found during parsing the identity mapping is assumed
        if (initialLayout.empty()) {
            for (Qubit i = 0; i < nqubits; ++i) {
                initialLayout.emplace(i, i);
            }
        }

        // try gathering (additional) output permutation information from measurements, e.g., a measurement
        //      `measure q[i] -> c[j];`
        // implies that the j-th (logical) output is obtained from measuring the i-th physical qubit.
        const bool outputPermutationFound = !outputPermutation.empty();

        // track whether the circuit contains measurements at the end of the circuit
        // if it does, then all qubits that are not measured shall be considered garbage outputs
        bool            outputPermutationFromMeasurements = false;
        std::set<Qubit> measuredQubits{};

        for (const auto& opIt: ops) {
            if (opIt->getType() == qc::Measure) {
                outputPermutationFromMeasurements = true;
                auto* op                          = dynamic_cast<NonUnitaryOperation*>(opIt.get());
                assert(op->getTargets().size() == op->getClassics().size());
                auto classicIt = op->getClassics().cbegin();
                for (const auto& q: op->getTargets()) {
                    const auto qubitidx = q;
                    // only the first measurement of a qubit is used to determine the output permutation
                    if (measuredQubits.count(qubitidx) != 0) {
                        continue;
                    }

                    const auto bitidx = *classicIt;
                    if (outputPermutationFound) {
                        // output permutation was already set before -> permute existing values
                        const auto current = outputPermutation.at(qubitidx);
                        if (static_cast<std::size_t>(qubitidx) != bitidx && static_cast<std::size_t>(current) != bitidx) {
                            for (auto& p: outputPermutation) {
                                if (static_cast<std::size_t>(p.second) == bitidx) {
                                    p.second = current;
                                    break;
                                }
                            }
                            outputPermutation.at(qubitidx) = static_cast<Qubit>(bitidx);
                        }
                    } else {
                        // directly set permutation if none was set beforehand
                        outputPermutation[qubitidx] = static_cast<Qubit>(bitidx);
                    }
                    measuredQubits.emplace(qubitidx);
                    ++classicIt;
                }
            }
        }

        // clear any qubits that were not measured from the output permutation
        // these will be marked garbage further down below
        if (outputPermutationFromMeasurements) {
            auto it = outputPermutation.begin();
            while (it != outputPermutation.end()) {
                if (measuredQubits.find(it->first) == measuredQubits.end()) {
                    it = outputPermutation.erase(it);
                } else {
                    ++it;
                }
            }
        }

        const bool buildOutputPermutation = outputPermutation.empty();
        for (const auto& [physicalIn, logicalIn]: initialLayout) {
            const bool isIdle = isIdleQubit(physicalIn);

            // if no output permutation was found, build it from the initial layout
            if (buildOutputPermutation && !isIdle) {
                outputPermutation.insert({physicalIn, logicalIn});
            }

            // if the qubit is not an output, mark it as garbage
            const bool isOutput = std::any_of(outputPermutation.begin(), outputPermutation.end(),
                                              [&logicalIn = logicalIn](const auto& p) { return p.second == logicalIn; });
            if (!isOutput) {
                setLogicalQubitGarbage(logicalIn);
            }

            // if the qubit is an ancillary and idle, mark it as garbage
            if (logicalQubitIsAncillary(logicalIn) && isIdle) {
                setLogicalQubitGarbage(logicalIn);
            }
        }
    }

    void QuantumComputation::addQubitRegister(std::size_t nq, const std::string& regName) {
        if (qregs.count(regName) != 0) {
            auto& reg = qregs.at(regName);
            if (reg.first + reg.second == nqubits + nancillae) {
                reg.second += nq;
            } else {
                throw QFRException("[addQubitRegister] Augmenting existing qubit registers is only supported for the last register in a circuit");
            }
        } else {
            qregs.try_emplace(regName, nqubits, nq);
        }
        assert(nancillae == 0); // should only reach this point if no ancillae are present

        for (std::size_t i = 0; i < nq; ++i) {
            auto j = nqubits + i;
            initialLayout.insert({j, j});
            outputPermutation.insert({j, j});
        }
        nqubits += nq;

        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }

        ancillary.resize(nqubits + nancillae);
        garbage.resize(nqubits + nancillae);
    }

    void QuantumComputation::addClassicalRegister(std::size_t nc, const std::string& regName) {
        if (cregs.count(regName) != 0) {
            throw QFRException("[addClassicalRegister] Augmenting existing classical registers is currently not supported");
        }
        if (nc == 0) {
            throw QFRException("[addClassicalRegister] New register size must be larger than 0");
        }

        cregs.try_emplace(regName, nclassics, nc);
        nclassics += nc;
    }

    void QuantumComputation::addAncillaryRegister(std::size_t nq, const std::string& regName) {
        const auto totalqubits = nqubits + nancillae;
        if (ancregs.count(regName) != 0) {
            auto& reg = ancregs.at(regName);
            if (reg.first + reg.second == totalqubits) {
                reg.second += nq;
            } else {
                throw QFRException("[addAncillaryRegister] Augmenting existing ancillary registers is only supported for the last register in a circuit");
            }
        } else {
            ancregs.try_emplace(regName, totalqubits, nq);
        }

        ancillary.resize(totalqubits + nq);
        garbage.resize(totalqubits + nq);
        for (std::size_t i = 0; i < nq; ++i) {
            auto j = static_cast<Qubit>(totalqubits + i);
            initialLayout.insert({j, j});
            outputPermutation.insert({j, j});
            ancillary[j] = true;
        }
        nancillae += nq;

        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }
    }

    // removes the i-th logical qubit and returns the index j it was assigned to in the initial layout
    // i.e., initialLayout[j] = i
    std::pair<Qubit, std::optional<Qubit>> QuantumComputation::removeQubit(const Qubit logicalQubitIndex) {
        // Find index of the physical qubit i is assigned to
        Qubit physicalQubitIndex = 0;
        for (const auto& [physical, logical]: initialLayout) {
            if (logical == logicalQubitIndex) {
                physicalQubitIndex = physical;
            }
        }

        // get register and register-index of the corresponding qubit
        auto reg = getQubitRegisterAndIndex(physicalQubitIndex);

        if (physicalQubitIsAncillary(physicalQubitIndex)) {
            // first index
            if (reg.second == 0) {
                // last remaining qubit of register
                if (ancregs[reg.first].second == 1) {
                    // delete register
                    ancregs.erase(reg.first);
                }
                // first qubit of register
                else {
                    ancregs[reg.first].first++;
                    ancregs[reg.first].second--;
                }
                // last index
            } else if (reg.second == ancregs[reg.first].second - 1) {
                // reduce count of register
                ancregs[reg.first].second--;
            } else {
                auto ancreg    = ancregs.at(reg.first);
                auto lowPart   = reg.first + "_l";
                auto lowIndex  = ancreg.first;
                auto lowCount  = reg.second;
                auto highPart  = reg.first + "_h";
                auto highIndex = ancreg.first + reg.second + 1;
                auto highCount = ancreg.second - reg.second - 1;

                ancregs.erase(reg.first);
                ancregs.try_emplace(lowPart, lowIndex, lowCount);
                ancregs.try_emplace(highPart, highIndex, highCount);
            }
            // reduce ancilla count
            nancillae--;
        } else {
            if (reg.second == 0) {
                // last remaining qubit of register
                if (qregs[reg.first].second == 1) {
                    // delete register
                    qregs.erase(reg.first);
                }
                // first qubit of register
                else {
                    qregs[reg.first].first++;
                    qregs[reg.first].second--;
                }
                // last index
            } else if (reg.second == qregs[reg.first].second - 1) {
                // reduce count of register
                qregs[reg.first].second--;
            } else {
                auto qreg      = qregs.at(reg.first);
                auto lowPart   = reg.first + "_l";
                auto lowIndex  = qreg.first;
                auto lowCount  = reg.second;
                auto highPart  = reg.first + "_h";
                auto highIndex = qreg.first + reg.second + 1;
                auto highCount = qreg.second - reg.second - 1;

                qregs.erase(reg.first);
                qregs.try_emplace(lowPart, lowIndex, lowCount);
                qregs.try_emplace(highPart, highIndex, highCount);
            }
            // reduce qubit count
            nqubits--;
        }

        // adjust initial layout permutation
        initialLayout.erase(physicalQubitIndex);

        // remove potential output permutation entry
        std::optional<Qubit> outputQubitIndex{};
        if (const auto it = outputPermutation.find(physicalQubitIndex); it != outputPermutation.end()) {
            outputQubitIndex = it->second;
            // erasing entry
            outputPermutation.erase(physicalQubitIndex);
        }

        // update all operations
        const auto totalQubits = nqubits + nancillae;
        for (auto& op: ops) {
            op->setNqubits(totalQubits);
        }

        // update ancillary and garbage tracking
        for (std::size_t i = logicalQubitIndex; i < totalQubits; ++i) {
            ancillary[i] = ancillary[i + 1];
            garbage[i]   = garbage[i + 1];
        }
        // unset last entry
        ancillary[totalQubits] = false;
        garbage[totalQubits]   = false;

        return {physicalQubitIndex, outputQubitIndex};
    }

    // adds j-th physical qubit as ancilla to the end of reg or creates the register if necessary
    void QuantumComputation::addAncillaryQubit(Qubit physicalQubitIndex, std::optional<Qubit> outputQubitIndex) {
        if (initialLayout.count(physicalQubitIndex) > 0 || outputPermutation.count(physicalQubitIndex) > 0) {
            throw QFRException("[addAncillaryQubit] Attempting to insert physical qubit that is already assigned");
        }

        bool fusionPossible = false;
        for (auto& ancreg: ancregs) {
            auto& ancStartIndex = ancreg.second.first;
            auto& ancCount      = ancreg.second.second;
            // 1st case: can append to start of existing register
            if (ancStartIndex == physicalQubitIndex + 1) {
                ancStartIndex--;
                ancCount++;
                fusionPossible = true;
                break;
            }
            // 2nd case: can append to end of existing register
            if (ancStartIndex + ancCount == physicalQubitIndex) {
                ancCount++;
                fusionPossible = true;
                break;
            }
        }

        if (ancregs.empty()) {
            ancregs.try_emplace("anc", physicalQubitIndex, 1);
        } else if (!fusionPossible) {
            auto newRegName = "anc_" + std::to_string(physicalQubitIndex);
            ancregs.try_emplace(newRegName, physicalQubitIndex, 1);
        }

        // index of logical qubit
        auto logicalQubitIndex = nqubits + nancillae;

        // resize ancillary and garbage tracking vectors
        ancillary.resize(logicalQubitIndex + 1U);
        garbage.resize(logicalQubitIndex + 1U);

        // increase ancillae count and mark as ancillary
        nancillae++;
        ancillary[logicalQubitIndex] = true;

        // adjust initial layout
        initialLayout.insert({physicalQubitIndex, logicalQubitIndex});

        // adjust output permutation
        if (outputQubitIndex.has_value()) {
            outputPermutation.insert({physicalQubitIndex, *outputQubitIndex});
        } else {
            // if a qubit is not relevant for the output, it is considered garbage
            garbage[logicalQubitIndex] = true;
        }

        // update all operations
        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }
    }

    void QuantumComputation::addQubit(const Qubit logicalQubitIndex, const Qubit physicalQubitIndex, const std::optional<Qubit> outputQubitIndex) {
        if (initialLayout.count(physicalQubitIndex) > 0 || outputPermutation.count(physicalQubitIndex) > 0) {
            throw QFRException("[addQubit] Attempting to insert physical qubit that is already assigned");
        }

        if (logicalQubitIndex > nqubits) {
            throw QFRException("[addQubit] There are currently only " + std::to_string(nqubits) +
                               " qubits in the circuit. Adding " + std::to_string(logicalQubitIndex) +
                               " is therefore not possible at the moment.");
            // TODO: this does not necessarily have to lead to an error. A new qubit register could be created and all ancillaries shifted
        }

        // check if qubit fits in existing register
        bool fusionPossible = false;
        for (auto& qreg: qregs) {
            auto& qStartIndex = qreg.second.first;
            auto& qCount      = qreg.second.second;
            // 1st case: can append to start of existing register
            if (qStartIndex == physicalQubitIndex + 1) {
                qStartIndex--;
                qCount++;
                fusionPossible = true;
                break;
            }
            // 2nd case: can append to end of existing register
            if (qStartIndex + qCount == physicalQubitIndex) {
                if (physicalQubitIndex == nqubits) {
                    // need to shift ancillaries
                    for (auto& ancreg: ancregs) {
                        ancreg.second.first++;
                    }
                }
                qCount++;
                fusionPossible = true;
                break;
            }
        }

        consolidateRegister(qregs);

        if (qregs.empty()) {
            qregs.try_emplace("q", physicalQubitIndex, 1);
        } else if (!fusionPossible) {
            auto newRegName = "q_" + std::to_string(physicalQubitIndex);
            qregs.try_emplace(newRegName, physicalQubitIndex, 1);
        }

        // increase qubit count
        nqubits++;
        // adjust initial layout
        initialLayout.insert({physicalQubitIndex, logicalQubitIndex});
        if (outputQubitIndex.has_value()) {
            // adjust output permutation
            outputPermutation.insert({physicalQubitIndex, *outputQubitIndex});
        }
        // update all operations
        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }

        // update ancillary and garbage tracking
        for (auto i = nqubits + nancillae - 1; i > logicalQubitIndex; --i) {
            ancillary[i] = ancillary[i - 1];
            garbage[i]   = garbage[i - 1];
        }
        // unset new entry
        ancillary[logicalQubitIndex] = false;
        garbage[logicalQubitIndex]   = false;
    }

    std::ostream& QuantumComputation::print(std::ostream& os) const {
        const auto width = ops.empty() ? 1 : static_cast<int>(std::log10(ops.size()) + 1.);
        if (!ops.empty()) {
            os << std::setw(width) << "i"
               << ": \t\t\t";
        } else {
            os << "i: \t\t\t";
        }
        for (const auto& [physical, logical]: initialLayout) {
            if (ancillary[logical]) {
                os << "\033[31m" << logical << "\t\033[0m";
            } else {
                os << logical << "\t";
            }
        }
        os << std::endl;
        size_t i = 0U;
        for (const auto& op: ops) {
            os << std::setw(width) << ++i << ": \t";
            op->print(os, initialLayout);
            os << std::endl;
        }
        if (!ops.empty()) {
            os << std::setw(width) << "o"
               << ": \t\t\t";
        } else {
            os << "o: \t\t\t";
        }
        for (const auto& physicalQubit: initialLayout) {
            auto it = outputPermutation.find(physicalQubit.first);
            if (it == outputPermutation.end()) {
                if (garbage[physicalQubit.second]) {
                    os << "\033[31m|\t\033[0m";
                } else {
                    os << "|\t";
                }
            } else {
                os << it->second << "\t";
            }
        }
        os << std::endl;
        return os;
    }

    void QuantumComputation::printBin(std::size_t n, std::stringstream& ss) {
        if (n > 1) {
            printBin(n / 2, ss);
        }
        ss << n % 2;
    }

    std::ostream& QuantumComputation::printStatistics(std::ostream& os) const {
        os << "QC Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits) << std::endl;
        os << "\tanc: " << static_cast<std::size_t>(nancillae) << std::endl;
        os << "\tm: " << ops.size() << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    void QuantumComputation::dump(const std::string& filename) {
        const std::size_t dot       = filename.find_last_of('.');
        std::string       extension = filename.substr(dot + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return ::tolower(c); });
        if (extension == "real") {
            dump(filename, Format::Real);
        } else if (extension == "qasm") {
            dump(filename, Format::OpenQASM);
        } else if (extension == "qc") {
            dump(filename, Format::QC);
        } else if (extension == "tfc") {
            dump(filename, Format::TFC);
        } else if (extension == "tensor") {
            dump(filename, Format::Tensor);
        } else {
            throw QFRException("[dump] Extension " + extension + " not recognized/supported for dumping.");
        }
    }

    void QuantumComputation::dumpOpenQASM(std::ostream& of) {
        // Add missing physical qubits
        if (!qregs.empty()) {
            for (Qubit physicalQubit = 0; physicalQubit < initialLayout.rbegin()->first; ++physicalQubit) {
                if (initialLayout.count(physicalQubit) == 0) {
                    const auto logicalQubit = getHighestLogicalQubitIndex() + 1;
                    addQubit(logicalQubit, physicalQubit, std::nullopt);
                }
            }
        }

        // dump initial layout and output permutation
        Permutation inverseInitialLayout{};
        for (const auto& q: initialLayout) {
            inverseInitialLayout.insert({q.second, q.first});
        }
        of << "// i";
        for (const auto& q: inverseInitialLayout) {
            of << " " << static_cast<std::size_t>(q.second);
        }
        of << std::endl;

        Permutation inverseOutputPermutation{};
        for (const auto& q: outputPermutation) {
            inverseOutputPermutation.insert({q.second, q.first});
        }
        of << "// o";
        for (const auto& q: inverseOutputPermutation) {
            of << " " << q.second;
        }
        of << std::endl;

        of << "OPENQASM 2.0;" << std::endl;
        of << "include \"qelib1.inc\";" << std::endl;
        if (!qregs.empty()) {
            printSortedRegisters(qregs, "qreg", of);
        } else if (nqubits > 0) {
            of << "qreg q[" << nqubits << "];" << std::endl;
        }
        if (!cregs.empty()) {
            printSortedRegisters(cregs, "creg", of);
        } else if (nclassics > 0) {
            of << "creg c[" << nclassics << "];" << std::endl;
        }
        if (!ancregs.empty()) {
            printSortedRegisters(ancregs, "qreg", of);
        } else if (nancillae > 0) {
            of << "qreg anc[" << nancillae << "];" << std::endl;
        }

        RegisterNames qregnames{};
        RegisterNames cregnames{};
        RegisterNames ancregnames{};
        createRegisterArray(qregs, qregnames, nqubits, "q");
        createRegisterArray(cregs, cregnames, nclassics, "c");
        createRegisterArray(ancregs, ancregnames, nancillae, "anc");

        for (const auto& ancregname: ancregnames) {
            qregnames.push_back(ancregname);
        }

        for (const auto& op: ops) {
            op->dumpOpenQASM(of, qregnames, cregnames);
        }
    }

    void QuantumComputation::dump(const std::string& filename, Format format) {
        assert(std::count(filename.begin(), filename.end(), '.') == 1);
        auto of = std::ofstream(filename);
        if (!of.good()) {
            throw QFRException("[dump] Error opening file: " + filename);
        }
        dump(of, format);
    }

    void QuantumComputation::dump(std::ostream&& of, Format format) {
        switch (format) {
            case Format::OpenQASM:
                dumpOpenQASM(of);
                break;
            case Format::Real:
                std::cerr << "Dumping in real format currently not supported\n";
                break;
            case Format::GRCS:
                std::cerr << "Dumping in GRCS format currently not supported\n";
                break;
            case Format::TFC:
                std::cerr << "Dumping in TFC format currently not supported\n";
                break;
            case Format::QC:
                std::cerr << "Dumping in QC format currently not supported\n";
                break;
            default:
                throw QFRException("[dump] Format not recognized/supported for dumping.");
        }
    }

    bool QuantumComputation::isIdleQubit(const Qubit physicalQubit) const {
        return !std::any_of(ops.cbegin(), ops.cend(), [&physicalQubit](const auto& op) { return op->actsOn(physicalQubit); });
    }

    void QuantumComputation::stripIdleQubits(bool force, bool reduceIOpermutations) {
        auto layoutCopy = initialLayout;
        for (auto physicalQubitIt = layoutCopy.rbegin(); physicalQubitIt != layoutCopy.rend(); ++physicalQubitIt) {
            auto physicalQubitIndex = physicalQubitIt->first;
            if (isIdleQubit(physicalQubitIndex)) {
                if (auto it = outputPermutation.find(physicalQubitIndex); it != outputPermutation.end() && !force) {
                    continue;
                }

                auto logicalQubitIndex = initialLayout.at(physicalQubitIndex);
                removeQubit(logicalQubitIndex);

                if (reduceIOpermutations && (logicalQubitIndex < nqubits + nancillae)) {
                    for (auto& q: initialLayout) {
                        if (q.second > logicalQubitIndex) {
                            --q.second;
                        }
                    }

                    for (auto& q: outputPermutation) {
                        if (q.second > logicalQubitIndex) {
                            --q.second;
                        }
                    }
                }
            }
        }
        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }
    }

    std::string QuantumComputation::getQubitRegister(const Qubit physicalQubitIndex) const {
        for (const auto& reg: qregs) {
            auto startIdx = reg.second.first;
            auto count    = reg.second.second;
            if (physicalQubitIndex < startIdx) {
                continue;
            }
            if (physicalQubitIndex >= startIdx + count) {
                continue;
            }
            return reg.first;
        }
        for (const auto& reg: ancregs) {
            auto startIdx = reg.second.first;
            auto count    = reg.second.second;
            if (physicalQubitIndex < startIdx) {
                continue;
            }
            if (physicalQubitIndex >= startIdx + count) {
                continue;
            }
            return reg.first;
        }

        throw QFRException("[getQubitRegister] Qubit index " + std::to_string(physicalQubitIndex) + " not found in any register");
    }

    std::pair<std::string, Qubit> QuantumComputation::getQubitRegisterAndIndex(const Qubit physicalQubitIndex) const {
        const std::string regName = getQubitRegister(physicalQubitIndex);
        Qubit             index   = 0;
        auto              it      = qregs.find(regName);
        if (it != qregs.end()) {
            index = physicalQubitIndex - it->second.first;
        } else {
            auto itAnc = ancregs.find(regName);
            if (itAnc != ancregs.end()) {
                index = physicalQubitIndex - itAnc->second.first;
            }
            // no else branch needed here, since error would have already shown in getQubitRegister(physicalQubitIndex)
        }
        return {regName, index};
    }

    std::string QuantumComputation::getClassicalRegister(const Bit classicalIndex) const {
        for (const auto& reg: cregs) {
            auto startIdx = reg.second.first;
            auto count    = reg.second.second;
            if (classicalIndex < startIdx) {
                continue;
            }
            if (classicalIndex >= startIdx + count) {
                continue;
            }
            return reg.first;
        }

        throw QFRException("[getClassicalRegister] Classical index " + std::to_string(classicalIndex) + " not found in any register");
    }

    std::pair<std::string, Bit> QuantumComputation::getClassicalRegisterAndIndex(const Bit classicalIndex) const {
        const std::string regName = getClassicalRegister(classicalIndex);
        std::size_t       index   = 0;
        auto              it      = cregs.find(regName);
        if (it != cregs.end()) {
            index = classicalIndex - it->second.first;
        } // else branch not needed since getClassicalRegister already covers this case
        return {regName, index};
    }

    Qubit QuantumComputation::getIndexFromQubitRegister(const std::pair<std::string, Qubit>& qubit) const {
        // no range check is performed here!
        return qregs.at(qubit.first).first + qubit.second;
    }
    Bit QuantumComputation::getIndexFromClassicalRegister(const std::pair<std::string, std::size_t>& clbit) const {
        // no range check is performed here!
        return cregs.at(clbit.first).first + clbit.second;
    }

    std::ostream& QuantumComputation::printPermutation(const Permutation& permutation, std::ostream& os) {
        for (const auto& [physical, logical]: permutation) {
            os << "\t" << physical << ": " << logical << std::endl;
        }
        return os;
    }

    std::ostream& QuantumComputation::printRegisters(std::ostream& os) const {
        os << "qregs:";
        for (const auto& qreg: qregs) {
            os << " {" << qreg.first << ", {" << qreg.second.first << ", " << qreg.second.second << "}}";
        }
        os << std::endl;
        if (!ancregs.empty()) {
            os << "ancregs:";
            for (const auto& ancreg: ancregs) {
                os << " {" << ancreg.first << ", {" << ancreg.second.first << ", " << ancreg.second.second << "}}";
            }
            os << std::endl;
        }
        os << "cregs:";
        for (const auto& creg: cregs) {
            os << " {" << creg.first << ", {" << creg.second.first << ", " << creg.second.second << "}}";
        }
        os << std::endl;
        return os;
    }

    Qubit QuantumComputation::getHighestLogicalQubitIndex(const Permutation& permutation) {
        Qubit maxIndex = 0;
        for (const auto& [physical, logical]: permutation) {
            maxIndex = std::max(maxIndex, logical);
        }
        return maxIndex;
    }

    bool QuantumComputation::physicalQubitIsAncillary(const Qubit physicalQubitIndex) const {
        return std::any_of(ancregs.cbegin(), ancregs.cend(), [&physicalQubitIndex](const auto& ancreg) { return ancreg.second.first <= physicalQubitIndex && physicalQubitIndex < ancreg.second.first + ancreg.second.second; });
    }

    void QuantumComputation::setLogicalQubitGarbage(const Qubit logicalQubitIndex) {
        garbage[logicalQubitIndex] = true;
        // setting a logical qubit garbage also means removing it from the output permutation if it was present before
        for (auto it = outputPermutation.begin(); it != outputPermutation.end(); ++it) {
            if (it->second == logicalQubitIndex) {
                outputPermutation.erase(it);
                break;
            }
        }
    }

    [[nodiscard]] std::pair<bool, std::optional<Qubit>> QuantumComputation::containsLogicalQubit(const Qubit logicalQubitIndex) const {
        if (const auto it = std::find_if(
                    initialLayout.cbegin(),
                    initialLayout.cend(),
                    [&logicalQubitIndex](const auto& mapping) {
                        return mapping.second == logicalQubitIndex;
                    });
            it != initialLayout.cend()) {
            return {true, it->first};
        }
        return {false, std::nullopt};
    }

    bool QuantumComputation::isLastOperationOnQubit(const const_iterator& opIt, const const_iterator& end) const {
        if (opIt == end) {
            return true;
        }

        // determine which qubits the gate acts on
        std::vector<bool> actson(nqubits + nancillae);
        for (std::size_t i = 0; i < actson.size(); ++i) {
            if ((*opIt)->actsOn(static_cast<Qubit>(i))) {
                actson[i] = true;
            }
        }

        // iterate over remaining gates and check if any act on qubits overlapping with the target gate
        auto atEnd = opIt;
        std::advance(atEnd, 1);
        while (atEnd != end) {
            for (std::size_t i = 0; i < actson.size(); ++i) {
                if (actson[i] && (*atEnd)->actsOn(static_cast<Qubit>(i))) {
                    return false;
                }
            }
            ++atEnd;
        }
        return true;
    }

    void QuantumComputation::unifyQuantumRegisters(const std::string& regName) {
        ancregs.clear();
        qregs.clear();
        qregs[regName] = {0, getNqubits()};
        nancillae      = 0;
    }

    void QuantumComputation::appendMeasurementsAccordingToOutputPermutation(const std::string& registerName) {
        // ensure that the circuit contains enough classical registers
        if (cregs.empty()) {
            // in case there are no registers, create a new one
            addClassicalRegister(outputPermutation.size(), registerName);
        } else if (nclassics < outputPermutation.size()) {
            if (cregs.find(registerName) == cregs.end()) {
                // in case there are registers but not enough, add a new one
                addClassicalRegister(outputPermutation.size() - nclassics, registerName);
            } else {
                // in case the register already exists, augment it
                nclassics += outputPermutation.size() - nclassics;
                cregs[registerName].second = outputPermutation.size();
            }
        }
        auto targets = std::vector<qc::Qubit>{};
        for (std::size_t q = 0; q < getNqubits(); ++q) {
            targets.emplace_back(q);
        }
        barrier(targets);
        // append measurements according to output permutation
        for (const auto& [qubit, clbit]: outputPermutation) {
            measure(qubit, clbit);
        }
    }

    void QuantumComputation::checkQubitRange(const Qubit qubit) const {
        if (const auto it = initialLayout.find(qubit); it == initialLayout.end() || it->second >= getNqubits()) {
            throw QFRException("Qubit index out of range: " + std::to_string(qubit));
        }
    }
    void QuantumComputation::checkQubitRange(const Qubit qubit0, const Qubit qubit1) const {
        checkQubitRange(qubit0);
        checkQubitRange(qubit1);
    }
    void QuantumComputation::checkQubitRange(const Qubit qubit, const Control& control) const {
        checkQubitRange(qubit);
        checkQubitRange(control.qubit);
    }
    void QuantumComputation::checkQubitRange(const Qubit qubit0, const Qubit qubit1, const Control& control) const {
        checkQubitRange(qubit0, qubit1);
        checkQubitRange(control.qubit);
    }
    void QuantumComputation::checkQubitRange(const Qubit qubit, const Controls& controls) const {
        checkQubitRange(qubit);
        for (const auto& [ctrl, _]: controls) {
            checkQubitRange(ctrl);
        }
    }

    void QuantumComputation::checkQubitRange(const Qubit qubit0, const Qubit qubit1, const Controls& controls) const {
        checkQubitRange(qubit0, controls);
        checkQubitRange(qubit1);
    }

    void QuantumComputation::checkQubitRange(const std::vector<Qubit>& qubits) const {
        for (const auto& qubit: qubits) {
            checkQubitRange(qubit);
        }
    }

    void QuantumComputation::addVariable(const SymbolOrNumber& expr) {
        if (std::holds_alternative<Symbolic>(expr)) {
            const auto& sym = std::get<Symbolic>(expr);
            for (const auto& term: sym) {
                occuringVariables.insert(term.getVar());
            }
        }
    }

    void QuantumComputation::u3(Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta) {
        checkQubitRange(target);
        addVariables(lambda, phi, theta);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::U3, lambda, phi, theta);
    }
    void QuantumComputation::u3(Qubit target, const Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta) {
        checkQubitRange(target, control);
        addVariables(lambda, phi, theta);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::U3, lambda, phi, theta);
    }
    void QuantumComputation::u3(Qubit target, const Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta) {
        checkQubitRange(target, controls);
        addVariables(lambda, phi, theta);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::U3, lambda, phi, theta);
    }

    void QuantumComputation::u2(Qubit target, const SymbolOrNumber& lambda, const SymbolOrNumber& phi) {
        checkQubitRange(target);
        addVariables(lambda, phi);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::U2, lambda, phi);
    }
    void QuantumComputation::u2(Qubit target, const Control& control, const SymbolOrNumber& lambda, const SymbolOrNumber& phi) {
        checkQubitRange(target, control);
        addVariables(lambda, phi);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::U2, lambda, phi);
    }
    void QuantumComputation::u2(Qubit target, const Controls& controls, const SymbolOrNumber& lambda, const SymbolOrNumber& phi) {
        checkQubitRange(target, controls);
        addVariables(lambda, phi);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::U2, lambda, phi);
    }

    void QuantumComputation::phase(Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::Phase, lambda);
    }
    void QuantumComputation::phase(Qubit target, const Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::Phase, lambda);
    }
    void QuantumComputation::phase(Qubit target, const Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::Phase, lambda);
    }

    void QuantumComputation::rx(Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::RX, lambda);
    }
    void QuantumComputation::rx(Qubit target, const Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::RX, lambda);
    }
    void QuantumComputation::rx(Qubit target, const Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::RX, lambda);
    }

    void QuantumComputation::ry(Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::RY, lambda);
    }
    void QuantumComputation::ry(Qubit target, const Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::RY, lambda);
    }
    void QuantumComputation::ry(Qubit target, const Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::RY, lambda);
    }

    void QuantumComputation::rz(Qubit target, const SymbolOrNumber& lambda) {
        checkQubitRange(target);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), target, qc::RZ, lambda);
    }
    void QuantumComputation::rz(Qubit target, const Control& control, const SymbolOrNumber& lambda) {
        checkQubitRange(target, control);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), control, target, qc::RZ, lambda);
    }
    void QuantumComputation::rz(Qubit target, const Controls& controls, const SymbolOrNumber& lambda) {
        checkQubitRange(target, controls);
        addVariables(lambda);
        emplace_back<SymbolicOperation>(getNqubits(), controls, target, qc::RZ, lambda);
    }

    // Instantiates this computation
    void QuantumComputation::instantiate(const VariableAssignment& assignment) {
        for (auto& op: ops) {
            if (auto* symOp = dynamic_cast<SymbolicOperation*>(op.get())) {
                symOp->instantiate(assignment);
            }
        }
    }
} // namespace qc
