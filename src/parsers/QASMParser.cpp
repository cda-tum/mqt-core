/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

void qc::QuantumComputation::importOpenQASM(std::istream& is) {
    using namespace qasm;
    // initialize parser
    Parser p(is, qregs, cregs);

    p.scan();
    p.check(Token::Kind::openqasm);
    p.check(Token::Kind::real);
    p.check(Token::Kind::semicolon);

    do {
        if (p.sym == Token::Kind::qreg) {
            // quantum register definition
            p.scan();
            p.check(Token::Kind::identifier);
            const std::string s = p.t.str;
            p.check(Token::Kind::lbrack);
            p.check(Token::Kind::nninteger);
            const auto n = static_cast<std::size_t>(p.t.val);
            p.check(Token::Kind::rbrack);
            p.check(Token::Kind::semicolon);
            addQubitRegister(n, s);
            p.nqubits = nqubits;
        } else if (p.sym == Token::Kind::creg) {
            // classical register definition
            p.scan();
            p.check(Token::Kind::identifier);
            const std::string s = p.t.str;
            p.check(Token::Kind::lbrack);
            p.check(Token::Kind::nninteger);
            const auto n = static_cast<std::size_t>(p.t.val);
            p.check(Token::Kind::rbrack);
            p.check(Token::Kind::semicolon);
            addClassicalRegister(n, s);
            p.nclassics = nclassics;
        } else if (p.sym == Token::Kind::ugate || p.sym == Token::Kind::cxgate || p.sym == Token::Kind::swap || p.sym == Token::Kind::identifier || p.sym == Token::Kind::measure || p.sym == Token::Kind::reset || p.sym == Token::Kind::mcx_gray || p.sym == Token::Kind::mcx_recursive || p.sym == Token::Kind::mcx_vchain || p.sym == Token::Kind::mcphase || p.sym == Token::Kind::sxgate || p.sym == Token::Kind::sxdggate) {
            // gate application
            ops.emplace_back(p.Qop());
        } else if (p.sym == Token::Kind::gate) {
            // gate definition
            p.GateDecl();
        } else if (p.sym == Token::Kind::include) {
            // include statement
            p.scan();
            p.check(Token::Kind::string);
            p.scanner->addFileInput(p.t.str);
            p.check(Token::Kind::semicolon);
        } else if (p.sym == Token::Kind::barrier) {
            // barrier statement
            p.scan();
            std::vector<qc::QuantumRegister> args;
            p.ArgList(args);
            p.check(Token::Kind::semicolon);

            std::vector<qc::Qubit> qubits{};
            for (auto& arg: args) {
                for (std::size_t q = 0; q < arg.second; ++q) {
                    qubits.emplace_back(arg.first + q);
                }
            }

            emplace_back<NonUnitaryOperation>(nqubits, qubits, Barrier);
        } else if (p.sym == Token::Kind::opaque) {
            // opaque gate definition
            p.OpaqueGateDecl();
        } else if (p.sym == Token::Kind::_if) {
            // classically-controlled operation
            p.scan();
            p.check(Token::Kind::lpar);
            p.check(Token::Kind::identifier);
            const std::string creg = p.t.str;
            p.check(Token::Kind::eq);
            p.check(Token::Kind::nninteger);
            const auto n = static_cast<std::size_t>(p.t.val);
            p.check(Token::Kind::rpar);

            auto it = p.cregs.find(creg);
            if (it == p.cregs.end()) {
                p.error("Error in if statement: " + creg + " is not a creg!");
            } else {
                emplace_back<ClassicControlledOperation>(p.Qop(), it->second, n);
            }
        } else if (p.sym == Token::Kind::snapshot) {
            // snapshot statement
            p.scan();
            p.check(Token::Kind::lpar);
            p.check(Token::Kind::nninteger);
            auto n = static_cast<std::size_t>(p.t.val);
            p.check(Token::Kind::rpar);

            std::vector<qc::QuantumRegister> arguments{};
            p.ArgList(arguments);

            p.check(Token::Kind::semicolon);

            for (auto& arg: arguments) {
                if (arg.second != 1) {
                    p.error("Error in snapshot: arguments must be qubits");
                }
            }

            Targets qubits{};
            qubits.reserve(arguments.size());
            for (auto& arg: arguments) {
                qubits.emplace_back(arg.first);
            }

            emplace_back<NonUnitaryOperation>(nqubits, qubits, n);
        } else if (p.sym == Token::Kind::probabilities) {
            // show probabilities statement
            emplace_back<NonUnitaryOperation>(nqubits);
            p.scan();
            p.check(Token::Kind::semicolon);
        } else if (p.sym == Token::Kind::comment) {
            // comment
            p.scan();
            p.handleComment();
        } else {
            p.error("Unexpected statement: started with " + KindNames[p.sym] + "!");
        }
    } while (p.sym != Token::Kind::eof);

    // if any I/O information was gathered during parsing, transfer it to the QuantumComputation
    if (!p.initialLayout.empty()) {
        initialLayout = std::move(p.initialLayout);
    }
    if (!p.outputPermutation.empty()) {
        outputPermutation = std::move(p.outputPermutation);
    }
}
