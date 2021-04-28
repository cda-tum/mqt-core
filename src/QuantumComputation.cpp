/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

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

    void QuantumComputation::import(const std::string& filename) {
        size_t      dot       = filename.find_last_of('.');
        std::string extension = filename.substr(dot + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) { return ::tolower(ch); });
        if (extension == "real") {
            import(filename, Real);
        } else if (extension == "qasm") {
            import(filename, OpenQASM);
        } else if (extension == "txt") {
            import(filename, GRCS);
        } else if (extension == "tfc") {
            import(filename, TFC);
        } else if (extension == "qc") {
            import(filename, QC);
        } else {
            throw QFRException("[import] extension " + extension + " not recognized");
        }
    }

    void QuantumComputation::import(const std::string& filename, Format format) {
        size_t slash = filename.find_last_of('/');
        size_t dot   = filename.find_last_of('.');
        name         = filename.substr(slash + 1, dot - slash - 1);

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
            case Real:
                importReal(is);
                break;
            case OpenQASM:
                updateMaxControls(2);
                importOpenQASM(is);
                break;
            case GRCS:
                importGRCS(is);
                break;
            case TFC:
                importTFC(is);
                break;
            case QC:
                importQC(is);
                break;
            default:
                throw QFRException("[import] Format " + std::to_string(format) + " not yet supported");
        }

        // initialize the initial layout and output permutation
        initializeIOMapping();
    }

    void QuantumComputation::initializeIOMapping() {
        // if no initial layout was found during parsing the identity mapping is assumed
        if (initialLayout.empty()) {
            for (dd::QubitCount i = 0; i < nqubits; ++i)
                initialLayout.insert({static_cast<dd::Qubit>(i), static_cast<dd::Qubit>(i)});
        }

        // try gathering (additional) output permutation information from measurements, e.g., a measurement
        //      `measure q[i] -> c[j];`
        // implies that the j-th (logical) output is obtained from measuring the i-th physical qubit.
        bool outputPermutationFound = !outputPermutation.empty();
        for (auto opIt = ops.begin(); opIt != ops.end(); ++opIt) {
            if ((*opIt)->getType() == qc::Measure) {
                if (!isLastOperationOnQubit(opIt))
                    continue;

                auto op = dynamic_cast<NonUnitaryOperation*>(opIt->get());
                assert(op->getTargets().size() == op->getClassics().size());
                auto classicIt = op->getClassics().cbegin();
                for (const auto& q: op->getTargets()) {
                    auto qubitidx = q;
                    auto bitidx   = *classicIt;

                    if (outputPermutationFound) {
                        // output permutation was already set before -> permute existing values
                        auto current = outputPermutation.at(qubitidx);
                        if (static_cast<std::size_t>(qubitidx) != bitidx && static_cast<std::size_t>(current) != bitidx) {
                            for (auto& p: outputPermutation) {
                                if (static_cast<std::size_t>(p.second) == bitidx) {
                                    p.second = current;
                                    break;
                                }
                            }
                            outputPermutation.at(qubitidx) = static_cast<dd::Qubit>(bitidx);
                        }
                    } else {
                        // directly set permutation if none was set beforehand
                        outputPermutation[qubitidx] = static_cast<dd::Qubit>(bitidx);
                    }
                    ++classicIt;
                }
            }
        }

        // if the output permutation is still empty, we assume the identity (i.e., it is equal to the initial layout)
        if (outputPermutation.empty()) {
            for (dd::QubitCount i = 0; i < nqubits; ++i) {
                // only add to output permutation if the qubit is actually acted upon
                if (!isIdleQubit(static_cast<dd::Qubit>(i)))
                    outputPermutation.insert({static_cast<dd::Qubit>(i), initialLayout.at(i)});
            }
        }

        // allow for incomplete output permutation -> mark rest as garbage
        for (const auto& in: initialLayout) {
            bool isOutput = false;
            for (const auto& out: outputPermutation) {
                if (in.second == out.second) {
                    isOutput = true;
                    break;
                }
            }
            if (!isOutput) {
                setLogicalQubitGarbage(in.second);
            }
        }
    }

    void QuantumComputation::addQubitRegister(std::size_t nq, const char* reg_name) {
        if (static_cast<std::size_t>(nqubits + nancillae + nq) > dd::Package::maxPossibleQubits) {
            throw QFRException("Requested too many qubits to be handled by the DD package. Qubit datatype only allows up to " +
                               std::to_string(dd::Package::maxPossibleQubits) + " qubits, while " +
                               std::to_string(nqubits + nancillae + nq) + " were requested. If you want to use more than " +
                               std::to_string(dd::Package::maxPossibleQubits) + " qubits, you have to recompile the package with a wider Qubit type in `export/dd_package/include/dd/Definitions.hpp!`");
        }

        if (qregs.count(reg_name)) {
            auto& reg = qregs.at(reg_name);
            if (reg.first + reg.second == nqubits + nancillae) {
                reg.second += nq;
            } else {
                throw QFRException("[addQubitRegister] Augmenting existing qubit registers is only supported for the last register in a circuit");
            }
        } else {
            qregs.insert({reg_name, {nqubits, static_cast<dd::QubitCount>(nq)}});
        }
        assert(nancillae == 0); // should only reach this point if no ancillae are present

        for (std::size_t i = 0; i < nq; ++i) {
            auto j = static_cast<dd::Qubit>(nqubits + i);
            initialLayout.insert({j, j});
            outputPermutation.insert({j, j});
        }
        nqubits += static_cast<dd::QubitCount>(nq);

        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }

        ancillary.resize(nqubits + nancillae);
        garbage.resize(nqubits + nancillae);
    }

    void QuantumComputation::addClassicalRegister(std::size_t nc, const char* reg_name) {
        if (cregs.count(reg_name)) {
            throw QFRException("[addClassicalRegister] Augmenting existing classical registers is currently not supported");
        }

        cregs.insert({reg_name, {nclassics, nc}});
        nclassics += nc;
    }

    void QuantumComputation::addAncillaryRegister(std::size_t nq, const char* reg_name) {
        if (static_cast<std::size_t>(nqubits + nancillae + nq) > dd::Package::maxPossibleQubits) {
            throw QFRException("Requested too many qubits to be handled by the DD package. Qubit datatype only allows up to " +
                               std::to_string(dd::Package::maxPossibleQubits) + " qubits, while " +
                               std::to_string(nqubits + nancillae + nq) + " were requested. If you want to use more than " +
                               std::to_string(dd::Package::maxPossibleQubits) + " qubits, you have to recompile the package with a wider Qubit type in `export/dd_package/include/dd/Definitions.hpp!`");
        }

        dd::QubitCount totalqubits = nqubits + nancillae;
        if (ancregs.count(reg_name)) {
            auto& reg = ancregs.at(reg_name);
            if (reg.first + reg.second == totalqubits) {
                reg.second += nq;
            } else {
                throw QFRException("[addAncillaryRegister] Augmenting existing ancillary registers is only supported for the last register in a circuit");
            }
        } else {
            ancregs.insert({reg_name, {totalqubits, static_cast<dd::QubitCount>(nq)}});
        }

        ancillary.resize(totalqubits + nq);
        garbage.resize(totalqubits + nq);
        for (std::size_t i = 0; i < nq; ++i) {
            auto j = static_cast<dd::Qubit>(totalqubits + i);
            initialLayout.insert({j, j});
            outputPermutation.insert({j, j});
            ancillary[j] = true;
        }
        nancillae += static_cast<dd::QubitCount>(nq);

        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }
    }

    // removes the i-th logical qubit and returns the index j it was assigned to in the initial layout
    // i.e., initialLayout[j] = i
    std::pair<dd::Qubit, dd::Qubit> QuantumComputation::removeQubit(dd::Qubit logical_qubit_index) {
        // Find index of the physical qubit i is assigned to
        dd::Qubit physical_qubit_index = 0;
        for (const auto& Q: initialLayout) {
            if (Q.second == logical_qubit_index)
                physical_qubit_index = Q.first;
        }

        // get register and register-index of the corresponding qubit
        auto reg = getQubitRegisterAndIndex(physical_qubit_index);

        if (physicalQubitIsAncillary(physical_qubit_index)) {
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
                auto ancreg     = ancregs.at(reg.first);
                auto low_part   = reg.first + "_l";
                auto low_index  = ancreg.first;
                auto low_count  = reg.second;
                auto high_part  = reg.first + "_h";
                auto high_index = ancreg.first + reg.second + 1;
                auto high_count = ancreg.second - reg.second - 1;

                ancregs.erase(reg.first);
                ancregs.insert({low_part, {low_index, low_count}});
                ancregs.insert({high_part, {high_index, high_count}});
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
                auto qreg       = qregs.at(reg.first);
                auto low_part   = reg.first + "_l";
                auto low_index  = qreg.first;
                auto low_count  = reg.second;
                auto high_part  = reg.first + "_h";
                auto high_index = qreg.first + reg.second + 1;
                auto high_count = qreg.second - reg.second - 1;

                qregs.erase(reg.first);
                qregs.insert({low_part, {low_index, low_count}});
                qregs.insert({high_part, {high_index, high_count}});
            }
            // reduce qubit count
            nqubits--;
        }

        // adjust initial layout permutation
        initialLayout.erase(physical_qubit_index);

        // remove potential output permutation entry
        dd::Qubit output_qubit_index = -1;
        auto      it                 = outputPermutation.find(physical_qubit_index);
        if (it != outputPermutation.end()) {
            output_qubit_index = it->second;
            // erasing entry
            outputPermutation.erase(physical_qubit_index);
        }

        // update all operations
        auto totalQubits = static_cast<dd::QubitCount>(nqubits + nancillae);
        for (auto& op: ops) {
            op->setNqubits(totalQubits);
        }

        // update ancillary and garbage tracking
        for (dd::QubitCount i = logical_qubit_index; i < totalQubits; ++i) {
            ancillary[i] = ancillary[i + 1];
            garbage[i]   = garbage[i + 1];
        }
        // unset last entry
        ancillary[totalQubits] = false;
        garbage[totalQubits]   = false;

        return {physical_qubit_index, output_qubit_index};
    }

    // adds j-th physical qubit as ancilla to the end of reg or creates the register if necessary
    void QuantumComputation::addAncillaryQubit(dd::Qubit physical_qubit_index, dd::Qubit output_qubit_index) {
        if (initialLayout.count(physical_qubit_index) || outputPermutation.count(physical_qubit_index)) {
            throw QFRException("[addAncillaryQubit] Attempting to insert physical qubit that is already assigned");
        }

        bool fusionPossible = false;
        for (auto& ancreg: ancregs) {
            auto& anc_start_index = ancreg.second.first;
            auto& anc_count       = ancreg.second.second;
            // 1st case: can append to start of existing register
            if (anc_start_index == physical_qubit_index + 1) {
                anc_start_index--;
                anc_count++;
                fusionPossible = true;
                break;
            }
            // 2nd case: can append to end of existing register
            else if (anc_start_index + anc_count == physical_qubit_index) {
                anc_count++;
                fusionPossible = true;
                break;
            }
        }

        if (ancregs.empty()) {
            ancregs.insert({DEFAULT_ANCREG, {physical_qubit_index, 1}});
        } else if (!fusionPossible) {
            auto new_reg_name = std::string(DEFAULT_ANCREG) + "_" + std::to_string(physical_qubit_index);
            ancregs.insert({new_reg_name, {physical_qubit_index, 1}});
        }

        // index of logical qubit
        auto logical_qubit_index = static_cast<dd::Qubit>(nqubits + nancillae);

        // increase ancillae count and mark as ancillary
        nancillae++;
        ancillary[logical_qubit_index] = true;

        // adjust initial layout
        initialLayout.insert({physical_qubit_index, logical_qubit_index});

        // adjust output permutation
        if (output_qubit_index >= 0) {
            outputPermutation.insert({physical_qubit_index, output_qubit_index});
        }

        // update all operations
        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }
    }

    void QuantumComputation::addQubit(dd::Qubit logical_qubit_index, dd::Qubit physical_qubit_index, dd::Qubit output_qubit_index) {
        if (initialLayout.count(physical_qubit_index) || outputPermutation.count(physical_qubit_index)) {
            throw QFRException("[addQubit] Attempting to insert physical qubit that is already assigned");
        }

        if (logical_qubit_index > nqubits) {
            throw QFRException("[addQubit] There are currently only " + std::to_string(nqubits) +
                               " qubits in the circuit. Adding " + std::to_string(logical_qubit_index) +
                               " is therefore not possible at the moment.");
            // TODO: this does not necessarily have to lead to an error. A new qubit register could be created and all ancillaries shifted
        }

        // check if qubit fits in existing register
        bool fusionPossible = false;
        for (auto& qreg: qregs) {
            auto& q_start_index = qreg.second.first;
            auto& q_count       = qreg.second.second;
            // 1st case: can append to start of existing register
            if (q_start_index == physical_qubit_index + 1) {
                q_start_index--;
                q_count++;
                fusionPossible = true;
                break;
            }
            // 2nd case: can append to end of existing register
            else if (q_start_index + q_count == physical_qubit_index) {
                if (physical_qubit_index == nqubits) {
                    // need to shift ancillaries
                    for (auto& ancreg: ancregs) {
                        ancreg.second.first++;
                    }
                }
                q_count++;
                fusionPossible = true;
                break;
            }
        }

        consolidateRegister(qregs);

        if (qregs.empty()) {
            qregs.insert({DEFAULT_QREG, {physical_qubit_index, 1}});
        } else if (!fusionPossible) {
            auto new_reg_name = std::string(DEFAULT_QREG) + "_" + std::to_string(physical_qubit_index);
            qregs.insert({new_reg_name, {physical_qubit_index, 1}});
        }

        // increase qubit count
        nqubits++;
        // adjust initial layout
        initialLayout.insert({physical_qubit_index, logical_qubit_index});
        if (output_qubit_index >= 0) {
            // adjust output permutation
            outputPermutation.insert({physical_qubit_index, output_qubit_index});
        }
        // update all operations
        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }

        // update ancillary and garbage tracking
        for (auto i = static_cast<dd::Qubit>(nqubits + nancillae - 1); i > logical_qubit_index; --i) {
            ancillary[i] = ancillary[i - 1];
            garbage[i]   = garbage[i - 1];
        }
        // unset new entry
        ancillary[logical_qubit_index] = false;
        garbage[logical_qubit_index]   = false;
    }

    MatrixDD QuantumComputation::createInitialMatrix(std::unique_ptr<dd::Package>& dd) const {
        auto e = dd->makeIdent(nqubits + nancillae);
        dd->incRef(e);
        e = dd->reduceAncillae(e, ancillary);
        return e;
    }

    MatrixDD QuantumComputation::buildFunctionality(std::unique_ptr<dd::Package>& dd) const {
        if (nqubits + nancillae == 0)
            return MatrixDD::one;

        auto permutation = initialLayout;
        auto e           = createInitialMatrix(dd);

        for (auto& op: ops) {
            auto tmp = dd->multiply(op->getDD(dd, permutation), e);

            dd->incRef(tmp);
            dd->decRef(e);
            e = tmp;

            dd->garbageCollect();
        }
        // correct permutation if necessary
        changePermutation(e, permutation, outputPermutation, dd);
        e = dd->reduceAncillae(e, ancillary);
        e = dd->reduceGarbage(e, garbage);

        return e;
    }

    MatrixDD QuantumComputation::buildFunctionalityRecursive(std::unique_ptr<dd::Package>& dd) const {
        if (nqubits + nancillae == 0)
            return MatrixDD::one;

        auto permutation = initialLayout;

        if (ops.size() == 1) {
            auto e = ops.front()->getDD(dd, permutation);
            dd->incRef(e);
            return e;
        }

        std::stack<MatrixDD> s{};
        auto                 depth = static_cast<std::size_t>(std::ceil(std::log2(ops.size())));
        buildFunctionalityRecursive(depth, 0, s, permutation, dd);
        auto e = s.top();
        s.pop();

        // correct permutation if necessary
        changePermutation(e, permutation, outputPermutation, dd);
        e = dd->reduceAncillae(e, ancillary);
        e = dd->reduceGarbage(e, garbage);

        return e;
    }

    bool QuantumComputation::buildFunctionalityRecursive(std::size_t depth, std::size_t opIdx, std::stack<MatrixDD>& s, Permutation& permutation, std::unique_ptr<dd::Package>& dd) const {
        // base case
        if (depth == 1) {
            auto e = ops[opIdx]->getDD(dd, permutation);
            ++opIdx;
            if (opIdx == ops.size()) { // only one element was left
                s.push(e);
                dd->incRef(e);
                return false;
            }
            auto f = ops[opIdx]->getDD(dd, permutation);
            s.push(dd->multiply(f, e)); // ! reverse multiplication
            dd->incRef(s.top());
            return (opIdx != ops.size() - 1);
        }

        // in case no operations are left after the first recursive call nothing has to be done
        size_t leftIdx = opIdx & ~(1UL << (depth - 1));
        if (!buildFunctionalityRecursive(depth - 1, leftIdx, s, permutation, dd)) return false;

        size_t rightIdx = opIdx | (1UL << (depth - 1));
        auto   success  = buildFunctionalityRecursive(depth - 1, rightIdx, s, permutation, dd);

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

    VectorDD QuantumComputation::simulate(const VectorDD& in, std::unique_ptr<dd::Package>& dd) const {
        // measurements are currently not supported here
        auto permutation = initialLayout;
        auto e           = in;
        dd->incRef(e);

        for (auto& op: ops) {
            auto tmp = dd->multiply(op->getDD(dd, permutation), e);
            dd->incRef(tmp);
            dd->decRef(e);
            e = tmp;

            dd->garbageCollect();
        }

        // correct permutation if necessary
        changePermutation(e, permutation, outputPermutation, dd);
        e = dd->reduceGarbage(e, garbage);

        return e;
    }

    std::ostream& QuantumComputation::print(std::ostream& os) const {
        if (!ops.empty()) {
            os << std::setw((int)std::log10(ops.size()) + 1) << "i: \t\t\t";
        } else {
            os << "i: \t\t\t";
        }
        for (const auto& Q: initialLayout) {
            if (ancillary[Q.second])
                os << "\033[31m" << static_cast<std::size_t>(Q.second) << "\t\033[0m";
            else
                os << static_cast<std::size_t>(Q.second) << "\t";
        }
        os << std::endl;
        size_t i = 0;
        for (const auto& op: ops) {
            os << std::setw((int)std::log10(ops.size()) + 1) << ++i << ": \t";
            op->print(os, initialLayout);
            os << std::endl;
        }
        if (!ops.empty()) {
            os << std::setw((int)std::log10(ops.size()) + 1) << "o: \t\t\t";
        } else {
            os << "o: \t\t\t";
        }
        for (const auto& physical_qubit: initialLayout) {
            auto it = outputPermutation.find(physical_qubit.first);
            if (it == outputPermutation.end()) {
                if (garbage[physical_qubit.second])
                    os << "\033[31m|\t\033[0m";
                else
                    os << "|\t";
            } else {
                os << static_cast<std::size_t>(it->second) << "\t";
            }
        }
        os << std::endl;
        return os;
    }

    void QuantumComputation::printBin(std::size_t n, std::stringstream& ss) {
        if (n > 1)
            printBin(n / 2, ss);
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
        size_t      dot       = filename.find_last_of('.');
        std::string extension = filename.substr(dot + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return ::tolower(c); });
        if (extension == "real") {
            dump(filename, Real);
        } else if (extension == "qasm") {
            dump(filename, OpenQASM);
        } else if (extension == "py") {
            dump(filename, Qiskit);
        } else if (extension == "qc") {
            dump(filename, QC);
        } else if (extension == "tfc") {
            dump(filename, TFC);
        } else {
            throw QFRException("[dump] Extension " + extension + " not recognized/supported for dumping.");
        }
    }

    void QuantumComputation::dumpOpenQASM(std::ostream& of) {
        // Add missing physical qubits
        if (!qregs.empty()) {
            for (dd::QubitCount physical_qubit = 0; physical_qubit < initialLayout.rbegin()->first; ++physical_qubit) {
                if (!initialLayout.count(static_cast<dd::Qubit>(physical_qubit))) {
                    auto logicalQubit = static_cast<dd::Qubit>(getHighestLogicalQubitIndex() + 1);
                    addQubit(logicalQubit, static_cast<dd::Qubit>(physical_qubit), -1);
                }
            }
        }

        // dump initial layout and output permutation
        Permutation inverseInitialLayout{};
        for (const auto& q: initialLayout)
            inverseInitialLayout.insert({q.second, q.first});
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
            of << " " << static_cast<std::size_t>(q.second);
        }
        of << std::endl;

        of << "OPENQASM 2.0;" << std::endl;
        of << "include \"qelib1.inc\";" << std::endl;
        if (!qregs.empty()) {
            printSortedRegisters(qregs, "qreg", of);
        } else if (nqubits > 0) {
            of << "qreg " << DEFAULT_QREG << "[" << static_cast<std::size_t>(nqubits) << "];" << std::endl;
        }
        if (!cregs.empty()) {
            printSortedRegisters(cregs, "creg", of);
        } else if (nclassics > 0) {
            of << "creg " << DEFAULT_CREG << "[" << nclassics << "];" << std::endl;
        }
        if (!ancregs.empty()) {
            printSortedRegisters(ancregs, "qreg", of);
        } else if (nancillae > 0) {
            of << "qreg " << DEFAULT_ANCREG << "[" << static_cast<std::size_t>(nancillae) << "];" << std::endl;
        }

        RegisterNames qregnames{};
        RegisterNames cregnames{};
        RegisterNames ancregnames{};
        createRegisterArray(qregs, qregnames, nqubits, DEFAULT_QREG);
        createRegisterArray(cregs, cregnames, nclassics, DEFAULT_CREG);
        createRegisterArray(ancregs, ancregnames, nancillae, DEFAULT_ANCREG);

        for (const auto& ancregname: ancregnames)
            qregnames.push_back(ancregname);

        for (const auto& op: ops) {
            op->dumpOpenQASM(of, qregnames, cregnames);
        }
    }

    void QuantumComputation::dump(const std::string& filename, Format format) {
        auto of = std::ofstream(filename);
        if (!of.good()) {
            throw QFRException("[dump] Error opening file: " + filename);
        }
        dump(of, format);
    }

    void QuantumComputation::dump(std::ostream&& of, Format format) {
        switch (format) {
            case OpenQASM:
                dumpOpenQASM(of);
                break;
            case Real:
                std::cerr << "Dumping in real format currently not supported\n";
                break;
            case GRCS:
                std::cerr << "Dumping in GRCS format currently not supported\n";
                break;
            case TFC:
                std::cerr << "Dumping in TFC format currently not supported\n";
                break;
            case QC:
                std::cerr << "Dumping in QC format currently not supported\n";
                break;
            case Qiskit:
                // TODO: improve/modernize Qiskit dump
                dd::QubitCount totalQubits = nqubits + nancillae + (max_controls >= 2 ? max_controls - 2 : 0);
                if (totalQubits > 53) {
                    std::cerr << "No more than 53 total qubits are currently supported" << std::endl;
                    break;
                }

                // For the moment all registers are fused together into for simplicity
                // This may be adapted in the future
                of << "from qiskit import *" << std::endl;
                of << "from qiskit.test.mock import ";
                dd::QubitCount narchitecture = 0;
                if (totalQubits <= 5) {
                    of << "FakeBurlington";
                    narchitecture = 5;
                } else if (totalQubits <= 20) {
                    of << "FakeBoeblingen";
                    narchitecture = 20;
                } else {
                    of << "FakeRochester";
                    narchitecture = 53;
                }
                of << std::endl;
                of << "from qiskit.converters import circuit_to_dag, dag_to_circuit" << std::endl;
                of << "from qiskit.transpiler.passes import *" << std::endl;
                of << "from math import pi" << std::endl
                   << std::endl;

                of << DEFAULT_QREG << " = QuantumRegister(" << static_cast<std::size_t>(nqubits) << ", '" << DEFAULT_QREG << "')" << std::endl;
                if (nclassics > 0) {
                    of << DEFAULT_CREG << " = ClassicalRegister(" << nclassics << ", '" << DEFAULT_CREG << "')" << std::endl;
                }
                if (nancillae > 0) {
                    of << DEFAULT_ANCREG << " = QuantumRegister(" << static_cast<std::size_t>(nancillae) << ", '" << DEFAULT_ANCREG << "')" << std::endl;
                }
                if (max_controls > 2) {
                    of << DEFAULT_MCTREG << " = QuantumRegister(" << static_cast<std::size_t>(max_controls - 2) << ", '" << DEFAULT_MCTREG << "')" << std::endl;
                }
                of << "qc = QuantumCircuit(";
                of << DEFAULT_QREG;
                if (nclassics > 0) {
                    of << ", " << DEFAULT_CREG;
                }
                if (nancillae > 0) {
                    of << ", " << DEFAULT_ANCREG;
                }
                if (max_controls > 2) {
                    of << ", " << DEFAULT_MCTREG;
                }
                of << ")" << std::endl
                   << std::endl;

                RegisterNames qregnames{};
                RegisterNames cregnames{};
                RegisterNames ancregnames{};
                createRegisterArray<QuantumRegister>({}, qregnames, nqubits, DEFAULT_QREG);
                createRegisterArray<ClassicalRegister>({}, cregnames, nclassics, DEFAULT_CREG);
                createRegisterArray<QuantumRegister>({}, ancregnames, nancillae, DEFAULT_ANCREG);

                for (const auto& ancregname: ancregnames)
                    qregnames.push_back(ancregname);

                for (const auto& op: ops) {
                    op->dumpQiskit(of, qregnames, cregnames, DEFAULT_MCTREG);
                }
                // add measurement for determining output mapping
                of << "qc.measure_all()" << std::endl;

                of << "qc_transpiled = transpile(qc, backend=";
                if (totalQubits <= 5) {
                    of << "FakeBurlington";
                } else if (totalQubits <= 20) {
                    of << "FakeBoeblingen";
                } else {
                    of << "FakeRochester";
                }
                of << "(), optimization_level=1)" << std::endl
                   << std::endl;
                of << "layout = qc_transpiled._layout" << std::endl;
                of << "virtual_bits = layout.get_virtual_bits()" << std::endl;

                of << "f = open(\"circuit"
                   << R"(_transpiled.qasm", "w"))" << std::endl;
                of << R"(f.write("// i"))" << std::endl;
                of << "for qubit in " << DEFAULT_QREG << ":" << std::endl;
                of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
                if (nancillae > 0) {
                    of << "for qubit in " << DEFAULT_ANCREG << ":" << std::endl;
                    of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
                }
                if (max_controls > 2) {
                    of << "for qubit in " << DEFAULT_MCTREG << ":" << std::endl;
                    of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
                }
                if (totalQubits < narchitecture) {
                    of << "for reg in layout.get_registers():" << std::endl;
                    of << '\t' << "if reg.name is 'ancilla':" << std::endl;
                    of << "\t\t"
                       << "for qubit in reg:" << std::endl;
                    of << "\t\t\t"
                       << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
                }
                of << R"(f.write("\n"))" << std::endl;
                of << "dag = circuit_to_dag(qc_transpiled)" << std::endl;
                of << "out = [item for sublist in list(dag.layers())[-1]['partition'] for item in sublist]" << std::endl;
                of << R"(f.write("// o"))" << std::endl;
                of << "for qubit in out:" << std::endl;
                of << '\t' << R"(f.write(" " + str(qubit.index)))" << std::endl;
                of << R"(f.write("\n"))" << std::endl;
                // remove measurements again
                of << "qc_transpiled = dag_to_circuit(RemoveFinalMeasurements().run(dag))" << std::endl;
                of << "f.write(qc_transpiled.qasm())" << std::endl;
                of << "f.close()" << std::endl;
                break;
        }
    }

    bool QuantumComputation::isIdleQubit(dd::Qubit physical_qubit) const {
        return !std::any_of(ops.cbegin(), ops.cend(), [&physical_qubit](const auto& op) { return op->actsOn(physical_qubit); });
    }

    void QuantumComputation::stripIdleQubits(bool force, bool reduceIOpermutations) {
        auto layout_copy = initialLayout;
        for (auto physical_qubit_it = layout_copy.rbegin(); physical_qubit_it != layout_copy.rend(); ++physical_qubit_it) {
            auto physical_qubit_index = physical_qubit_it->first;
            if (isIdleQubit(physical_qubit_index)) {
                auto it = outputPermutation.find(physical_qubit_index);
                if (it != outputPermutation.end()) {
                    auto output_index = it->second;
                    if (!force && output_index >= 0) continue;
                }

                auto logical_qubit_index = initialLayout.at(physical_qubit_index);
                removeQubit(logical_qubit_index);

                if (reduceIOpermutations && (logical_qubit_index < nqubits + nancillae)) {
                    for (auto& q: initialLayout) {
                        if (q.second > logical_qubit_index)
                            q.second--;
                    }

                    for (auto& q: outputPermutation) {
                        if (q.second > logical_qubit_index)
                            q.second--;
                    }
                }
            }
        }
        for (auto& op: ops) {
            op->setNqubits(nqubits + nancillae);
        }
    }

    std::string QuantumComputation::getQubitRegister(dd::Qubit physical_qubit_index) const {
        for (const auto& reg: qregs) {
            auto start_idx = reg.second.first;
            auto count     = reg.second.second;
            if (physical_qubit_index < start_idx) continue;
            if (physical_qubit_index >= start_idx + count) continue;
            return reg.first;
        }
        for (const auto& reg: ancregs) {
            auto start_idx = reg.second.first;
            auto count     = reg.second.second;
            if (physical_qubit_index < start_idx) continue;
            if (physical_qubit_index >= start_idx + count) continue;
            return reg.first;
        }

        throw QFRException("[getQubitRegister] Qubit index " + std::to_string(physical_qubit_index) + " not found in any register");
    }

    std::pair<std::string, dd::Qubit> QuantumComputation::getQubitRegisterAndIndex(dd::Qubit physical_qubit_index) const {
        std::string reg_name = getQubitRegister(physical_qubit_index);
        dd::Qubit   index    = 0;
        auto        it       = qregs.find(reg_name);
        if (it != qregs.end()) {
            index = static_cast<dd::Qubit>(physical_qubit_index - it->second.first);
        } else {
            auto it_anc = ancregs.find(reg_name);
            if (it_anc != ancregs.end()) {
                index = static_cast<dd::Qubit>(physical_qubit_index - it_anc->second.first);
            }
            // no else branch needed here, since error would have already shown in getQubitRegister(physical_qubit_index)
        }
        return {reg_name, index};
    }

    std::string QuantumComputation::getClassicalRegister(std::size_t classical_index) const {
        for (const auto& reg: cregs) {
            auto start_idx = reg.second.first;
            auto count     = reg.second.second;
            if (classical_index < start_idx) continue;
            if (classical_index >= start_idx + count) continue;
            return reg.first;
        }

        throw QFRException("[getClassicalRegister] Classical index " + std::to_string(classical_index) + " not found in any register");
    }

    std::pair<std::string, std::size_t> QuantumComputation::getClassicalRegisterAndIndex(std::size_t classical_index) const {
        std::string reg_name = getClassicalRegister(classical_index);
        std::size_t index    = 0;
        auto        it       = cregs.find(reg_name);
        if (it != cregs.end()) {
            index = classical_index - it->second.first;
        } // else branch not needed since getClassicalRegister already covers this case
        return {reg_name, index};
    }

    dd::Qubit QuantumComputation::getIndexFromQubitRegister(const std::pair<std::string, dd::Qubit>& qubit) const {
        // no range check is performed here!
        return static_cast<dd::Qubit>(qregs.at(qubit.first).first + qubit.second);
    }
    std::size_t QuantumComputation::getIndexFromClassicalRegister(const std::pair<std::string, std::size_t>& clbit) const {
        // no range check is performed here!
        return static_cast<std::size_t>(cregs.at(clbit.first).first + clbit.second);
    }

    std::ostream& QuantumComputation::printPermutation(const Permutation& permutation, std::ostream& os) {
        for (const auto& Q: permutation) {
            os << "\t" << static_cast<std::size_t>(Q.first) << ": " << static_cast<std::size_t>(Q.second) << std::endl;
        }
        return os;
    }

    std::ostream& QuantumComputation::printRegisters(std::ostream& os) const {
        os << "qregs:";
        for (const auto& qreg: qregs) {
            os << " {" << qreg.first << ", {" << static_cast<std::size_t>(qreg.second.first) << ", " << static_cast<std::size_t>(qreg.second.second) << "}}";
        }
        os << std::endl;
        if (!ancregs.empty()) {
            os << "ancregs:";
            for (const auto& ancreg: ancregs) {
                os << " {" << ancreg.first << ", {" << static_cast<std::size_t>(ancreg.second.first) << ", " << static_cast<std::size_t>(ancreg.second.second) << "}}";
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

    dd::Qubit QuantumComputation::getHighestLogicalQubitIndex(const Permutation& permutation) {
        dd::Qubit max_index = 0;
        for (const auto& physical_qubit: permutation) {
            max_index = std::max(max_index, physical_qubit.second);
        }
        return max_index;
    }

    bool QuantumComputation::physicalQubitIsAncillary(dd::Qubit physical_qubit_index) const {
        return std::any_of(ancregs.cbegin(), ancregs.cend(), [&physical_qubit_index](const auto& ancreg) { return ancreg.second.first <= physical_qubit_index && physical_qubit_index < ancreg.second.first + ancreg.second.second; });
    }

    bool QuantumComputation::isLastOperationOnQubit(const decltype(ops.cbegin())& opIt, const decltype(ops.cend())& end) const {
        if (opIt == end)
            return true;

        // determine which qubits the gate acts on
        std::vector<bool> actson(nqubits + nancillae);
        for (std::size_t i = 0; i < actson.size(); ++i) {
            if ((*opIt)->actsOn(static_cast<dd::Qubit>(i)))
                actson[i] = true;
        }

        // iterate over remaining gates and check if any act on qubits overlapping with the target gate
        auto atEnd = opIt;
        std::advance(atEnd, 1);
        while (atEnd != end) {
            for (std::size_t i = 0; i < actson.size(); ++i) {
                if (actson[i] && (*atEnd)->actsOn(static_cast<dd::Qubit>(i))) return false;
            }
            ++atEnd;
        }
        return true;
    }
} // namespace qc
