/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "parsers/qasm_parser/Parser.hpp"

namespace qasm {

    /***
     * Private Methods
     ***/
    std::shared_ptr<Parser::Expr> Parser::exponentiation() {
        if (sym == Token::Kind::Minus) {
            scan();
            auto x = exponentiation();
            if (x->kind == Expr::Kind::Number) {
                x->num = -x->num;
            } else {
                x = std::make_shared<Expr>(Expr::Kind::Sign, 0., x);
            }
            return x;
        }

        if (sym == Token::Kind::Real) {
            scan();
            return std::make_shared<Expr>(Expr::Kind::Number, t.valReal);
        }
        if (sym == Token::Kind::Nninteger) {
            scan();
            return std::make_shared<Expr>(Expr::Kind::Number, t.val);
        }
        if (sym == Token::Kind::Pi) {
            scan();
            return std::make_shared<Expr>(Expr::Kind::Number, qc::PI);
        }
        if (sym == Token::Kind::Identifier) {
            scan();
            return std::make_shared<Expr>(Expr::Kind::Id, 0., nullptr, nullptr, t.str);
        }
        if (sym == Token::Kind::Lpar) {
            scan();
            auto x = exp();
            check(Token::Kind::Rpar);
            return x;
        }
        if (unaryops.find(sym) != unaryops.end()) {
            auto op = sym;
            scan();
            check(Token::Kind::Lpar);
            auto x = exp();
            check(Token::Kind::Rpar);
            if (x->kind == Expr::Kind::Number) {
                if (op == Token::Kind::Sin) {
                    x->num = std::sin(x->num);
                } else if (op == Token::Kind::Cos) {
                    x->num = std::cos(x->num);
                } else if (op == Token::Kind::Tan) {
                    x->num = std::tan(x->num);
                } else if (op == Token::Kind::Exp) {
                    x->num = std::exp(x->num);
                } else if (op == Token::Kind::Ln) {
                    x->num = std::log(x->num);
                } else if (op == Token::Kind::Sqrt) {
                    x->num = std::sqrt(x->num);
                }
                return x;
            }
            if (op == Token::Kind::Sin) {
                return std::make_shared<Expr>(Expr::Kind::Sin, 0., x);
            }
            if (op == Token::Kind::Cos) {
                return std::make_shared<Expr>(Expr::Kind::Cos, 0., x);
            }
            if (op == Token::Kind::Tan) {
                return std::make_shared<Expr>(Expr::Kind::Tan, 0., x);
            }
            if (op == Token::Kind::Exp) {
                return std::make_shared<Expr>(Expr::Kind::Exp, 0., x);
            }
            if (op == Token::Kind::Ln) {
                return std::make_shared<Expr>(Expr::Kind::Ln, 0., x);
            }
            if (op == Token::Kind::Sqrt) {
                return std::make_shared<Expr>(Expr::Kind::Sqrt, 0., x);
            }
        } else {
            error("Invalid Expression");
        }

        return nullptr;
    }

    std::shared_ptr<Parser::Expr> Parser::factor() {
        auto x = exponentiation();
        while (sym == Token::Kind::Power) {
            scan();
            auto y = exponentiation();
            if (x->kind == Expr::Kind::Number && y->kind == Expr::Kind::Number) {
                x->num = std::pow(x->num, y->num);
            } else {
                x = std::make_shared<Expr>(Expr::Kind::Power, 0., x, y);
            }
        }
        return x;
    }

    std::shared_ptr<Parser::Expr> Parser::term() {
        auto x = factor();
        while (sym == Token::Kind::Times || sym == Token::Kind::Div) {
            auto op = sym;
            scan();
            auto y = factor();
            if (op == Token::Kind::Times) {
                if (x->kind == Expr::Kind::Number && y->kind == Expr::Kind::Number) {
                    x->num = x->num * y->num;
                } else {
                    x = std::make_shared<Expr>(Expr::Kind::Times, 0., x, y);
                }
            } else {
                if (x->kind == Expr::Kind::Number && y->kind == Expr::Kind::Number) {
                    x->num = x->num / y->num;
                } else {
                    x = std::make_shared<Expr>(Expr::Kind::Div, 0., x, y);
                }
            }
        }
        return x;
    }

    std::shared_ptr<Parser::Expr> Parser::exp() {
        std::shared_ptr<Expr> x{};
        if (sym == Token::Kind::Minus) {
            scan();
            x = term();
            if (x->kind == Expr::Kind::Number) {
                x->num = -x->num;
            } else {
                x = std::make_shared<Expr>(Expr::Kind::Sign, 0., x);
            }
        } else {
            x = term();
        }

        while (sym == Token::Kind::Plus || sym == Token::Kind::Minus) {
            auto op = sym;
            scan();
            auto y = term();
            if (op == Token::Kind::Plus) {
                if (x->kind == Expr::Kind::Number && y->kind == Expr::Kind::Number) {
                    x->num += y->num;
                } else {
                    x = std::make_shared<Expr>(Expr::Kind::Plus, 0., x, y);
                }
            } else {
                if (x->kind == Expr::Kind::Number && y->kind == Expr::Kind::Number) {
                    x->num -= y->num;
                } else {
                    x = std::make_shared<Expr>(Expr::Kind::Minus, 0., x, y);
                }
            }
        }
        return x;
    }

    std::shared_ptr<Parser::Expr> Parser::rewriteExpr(const std::shared_ptr<Expr>& expr, std::map<std::string, std::shared_ptr<Expr>>& exprMap) {
        if (expr == nullptr) {
            return nullptr;
        }
        auto op1 = rewriteExpr(expr->op1, exprMap);
        auto op2 = rewriteExpr(expr->op2, exprMap);

        if (expr->kind == Expr::Kind::Number) {
            return std::make_shared<Expr>(expr->kind, expr->num, op1, op2, expr->id);
        }
        if (expr->kind == Expr::Kind::Plus) {
            if (op1->kind == Expr::Kind::Number && op2->kind == Expr::Kind::Number) {
                op1->num = op1->num + op2->num;
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Minus) {
            if (op1->kind == Expr::Kind::Number && op2->kind == Expr::Kind::Number) {
                op1->num = op1->num - op2->num;
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Sign) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = -op1->num;
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Times) {
            if (op1->kind == Expr::Kind::Number && op2->kind == Expr::Kind::Number) {
                op1->num = op1->num * op2->num;
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Div) {
            if (op1->kind == Expr::Kind::Number && op2->kind == Expr::Kind::Number) {
                op1->num = op1->num / op2->num;
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Power) {
            if (op1->kind == Expr::Kind::Number && op2->kind == Expr::Kind::Number) {
                op1->num = std::pow(op1->num, op2->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Sin) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = std::sin(op1->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Cos) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = std::cos(op1->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Tan) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = std::tan(op1->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Exp) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = std::exp(op1->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Ln) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = std::log(op1->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Sqrt) {
            if (op1->kind == Expr::Kind::Number) {
                op1->num = std::sqrt(op1->num);
                return op1;
            }
        } else if (expr->kind == Expr::Kind::Id) {
            return exprMap[expr->id];
        }

        return std::make_shared<Expr>(expr->kind, expr->num, op1, op2, expr->id);
    }

    void Parser::handleComment() {
        // check if this comment provides any I/O mapping information
        auto&& initial = checkForInitialLayout(t.str);
        if (!initial.empty()) {
            if (!initialLayout.empty()) {
                error("Multiple initial layout specifications found.");
            } else {
                initialLayout = initial;
            }
        }
        auto&& output = checkForOutputPermutation(t.str);
        if (!output.empty()) {
            if (!outputPermutation.empty()) {
                error("Multiple output permutation specifications found.");
            } else {
                outputPermutation = output;
            }
        }
    }

    qc::Permutation Parser::checkForInitialLayout(std::string comment) {
        static auto     initialLayoutRegex = std::regex("i (\\d+ )*(\\d+)");
        static auto     qubitRegex         = std::regex("\\d+");
        qc::Permutation initial{};
        if (std::regex_search(comment, initialLayoutRegex)) {
            qc::Qubit logicalQubit = 0;
            for (std::smatch m; std::regex_search(comment, m, qubitRegex); comment = m.suffix()) {
                auto physicalQubit = static_cast<qc::Qubit>(std::stoul(m.str()));
                initial.insert({physicalQubit, logicalQubit});
                ++logicalQubit;
            }
        }
        return initial;
    }

    qc::Permutation Parser::checkForOutputPermutation(std::string comment) {
        static auto     outputPermutationRegex = std::regex("o (\\d+ )*(\\d+)");
        static auto     qubitRegex             = std::regex("\\d+");
        qc::Permutation output{};
        if (std::regex_search(comment, outputPermutationRegex)) {
            qc::Qubit logicalQubit = 0;
            for (std::smatch m; std::regex_search(comment, m, qubitRegex); comment = m.suffix()) {
                auto physicalQubit = static_cast<qc::Qubit>(std::stoul(m.str()));
                output.insert({physicalQubit, logicalQubit});
                ++logicalQubit;
            }
        }
        return output;
    }

    /***
     * Public Methods
     ***/
    void Parser::scan() {
        t   = la;
        la  = scanner->next();
        sym = la.kind;
    }

    void Parser::check(Token::Kind expected) {
        while (sym == Token::Kind::Comment) {
            scan();
            handleComment();
        }

        if (sym == expected) {
            scan();
        } else {
            error("Expected '" + qasm::KIND_NAMES.at(expected) + "' but found '" + qasm::KIND_NAMES.at(sym) + "' in line " + std::to_string(la.line) + ", column " + std::to_string(la.col));
        }
    }

    qc::QuantumRegister Parser::argumentQreg() {
        check(Token::Kind::Identifier);
        const std::string s = t.str;
        if (qregs.find(s) == qregs.end()) {
            error("Argument is not a qreg: " + s);
        }

        if (sym == Token::Kind::Lbrack) {
            scan();
            check(Token::Kind::Nninteger);
            auto offset = static_cast<std::size_t>(t.val);
            check(Token::Kind::Rbrack);
            return std::make_pair(qregs[s].first + offset, 1);
        }
        return std::make_pair(qregs[s].first, qregs[s].second);
    }

    qc::ClassicalRegister Parser::argumentCreg() {
        check(Token::Kind::Identifier);
        const std::string s = t.str;
        if (cregs.find(s) == cregs.end()) {
            error("Argument is not a creg: " + s);
        }

        if (sym == Token::Kind::Lbrack) {
            scan();
            check(Token::Kind::Nninteger);
            auto offset = static_cast<std::size_t>(t.val);
            check(Token::Kind::Rbrack);
            return std::make_pair(cregs[s].first + offset, 1);
        }

        return std::make_pair(cregs[s].first, cregs[s].second);
    }

    void Parser::expList(std::vector<std::shared_ptr<Parser::Expr>>& expressions) {
        expressions.emplace_back(exp());
        while (sym == Token::Kind::Comma) {
            scan();
            expressions.emplace_back(exp());
        }
    }

    void Parser::argList(std::vector<qc::QuantumRegister>& arguments) {
        arguments.emplace_back(argumentQreg());
        while (sym == Token::Kind::Comma) {
            scan();
            arguments.emplace_back(argumentQreg());
        }
    }

    void Parser::idList(std::vector<std::string>& identifiers) {
        check(Token::Kind::Identifier);
        identifiers.emplace_back(t.str);
        while (sym == Token::Kind::Comma) {
            scan();
            check(Token::Kind::Identifier);
            identifiers.emplace_back(t.str);
        }
    }

    std::unique_ptr<qc::Operation> Parser::gate() {
        if (sym == Token::Kind::Ugate) {
            scan();
            check(Token::Kind::Lpar);
            const auto theta = exp();
            check(Token::Kind::Comma);
            const auto phi = exp();
            check(Token::Kind::Comma);
            const auto lambda = exp();
            check(Token::Kind::Rpar);
            auto target = argumentQreg();
            check(Token::Kind::Semicolon);

            if (target.second == 1) {
                return std::make_unique<qc::StandardOperation>(nqubits, target.first, qc::U3, lambda->num, phi->num, theta->num);
            }

            // TODO: multiple targets could be useful here
            auto gate = qc::CompoundOperation(nqubits);
            for (std::size_t i = 0; i < target.second; ++i) {
                gate.emplace_back<qc::StandardOperation>(nqubits, target.first + i, qc::U3, lambda->num, phi->num, theta->num);
            }
            return std::make_unique<qc::CompoundOperation>(std::move(gate));
        }
        if (sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive || sym == Token::Kind::McxVchain) {
            auto type = sym;
            scan();
            std::vector<qc::QuantumRegister> registers{};
            registers.emplace_back(argumentQreg());
            while (sym != Token::Kind::Semicolon) {
                check(Token::Kind::Comma);
                registers.emplace_back(argumentQreg());
            }
            scan();

            std::vector<qc::Control> qubits{};
            for (const auto& reg: registers) {
                if (reg.second != 1) {
                    error("MCX for whole qubit registers not yet implemented");
                }

                if (std::count(registers.begin(), registers.end(), reg) > 1) {
                    std::ostringstream oss{};
                    oss << "Duplicate qubit " << reg.first << " in mcx definition";
                    error(oss.str());
                }

                qubits.emplace_back(qc::Control{reg.first});
            }

            // drop ancillaries since our library can natively work with MCTs
            if (type == Token::Kind::McxVchain) {
                // n controls, 1 target, n-2 ancillaries = 2n-1 qubits
                const auto ancillaries = (qubits.size() + 1) / 2 - 2;
                for (std::size_t i = 0; i < ancillaries; ++i) {
                    qubits.pop_back();
                }
            } else if (type == Token::Kind::McxRecursive) {
                // 1 ancillary if more than 4 controls
                if (qubits.size() > 5) {
                    qubits.pop_back();
                }
            }
            auto target = qubits.back().qubit;
            qubits.pop_back();
            return std::make_unique<qc::StandardOperation>(nqubits, qc::Controls{qubits.cbegin(), qubits.cend()}, target);
        }
        if (sym == Token::Kind::Mcphase) {
            scan();
            check(Token::Kind::Lpar);
            const auto lambda = exp();
            check(Token::Kind::Rpar);

            std::vector<qc::QuantumRegister> registers{};
            registers.emplace_back(argumentQreg());
            while (sym != Token::Kind::Semicolon) {
                check(Token::Kind::Comma);
                registers.emplace_back(argumentQreg());
            }
            scan();

            std::vector<qc::Control> qubits{};
            for (const auto& reg: registers) {
                if (reg.second != 1) {
                    error("Mcphase for whole qubit registers not yet implemented");
                }

                if (std::count(registers.begin(), registers.end(), reg) > 1) {
                    std::ostringstream oss{};
                    oss << "Duplicate qubit " << reg.first << " in Mcphase definition";
                    error(oss.str());
                }

                qubits.emplace_back(qc::Control{reg.first});
            }
            auto target = qubits.back().qubit;
            qubits.pop_back();
            return std::make_unique<qc::StandardOperation>(nqubits, qc::Controls{qubits.cbegin(), qubits.cend()}, target, qc::Phase, lambda->num);
        }
        if (sym == Token::Kind::Swap) {
            scan();
            auto firstTarget = argumentQreg();
            check(Token::Kind::Comma);
            auto secondTarget = argumentQreg();
            check(Token::Kind::Semicolon);

            // return corresponding operation
            if (firstTarget.second == 1 && secondTarget.second == 1) {
                if (firstTarget.first == secondTarget.first) {
                    error("SWAP with two identical targets");
                }
                return std::make_unique<qc::StandardOperation>(nqubits, qc::Controls{}, firstTarget.first, secondTarget.first, qc::SWAP);
            }
            error("SWAP for whole qubit registers not yet implemented");
        }
        if (sym == Token::Kind::Cxgate) {
            scan();
            auto control = argumentQreg();
            check(Token::Kind::Comma);
            auto target = argumentQreg();
            check(Token::Kind::Semicolon);

            // valid check
            for (std::size_t i = 0; i < control.second; ++i) {
                for (std::size_t j = 0; j < target.second; ++j) {
                    if (control.first + i == target.first + j) {
                        std::ostringstream oss{};
                        oss << "Qubit " << control.first + i << " cannot be control and target at the same time";
                        error(oss.str());
                    }
                }
            }

            // return corresponding operation
            if (control.second == 1 && target.second == 1) {
                return std::make_unique<qc::StandardOperation>(nqubits, qc::Control{control.first}, target.first, qc::X);
            }
            auto gate = qc::CompoundOperation(nqubits);
            if (control.second == target.second) {
                for (std::size_t i = 0; i < target.second; ++i) {
                    gate.emplace_back<qc::StandardOperation>(nqubits, qc::Control{static_cast<qc::Qubit>(control.first + i)}, target.first + i, qc::X);
                }
            } else if (control.second == 1) {
                // TODO: multiple targets could be useful here
                for (std::size_t i = 0; i < target.second; ++i) {
                    gate.emplace_back<qc::StandardOperation>(nqubits, qc::Control{control.first}, target.first + i, qc::X);
                }
            } else if (target.second == 1) {
                for (std::size_t i = 0; i < control.second; ++i) {
                    gate.emplace_back<qc::StandardOperation>(nqubits, qc::Control{static_cast<qc::Qubit>(control.first + i)}, target.first, qc::X);
                }
            } else {
                error("Register size does not match for CX gate!");
            }
            return std::make_unique<qc::CompoundOperation>(std::move(gate));
        }
        if (sym == Token::Kind::Sxgate || sym == Token::Kind::Sxdggate) {
            const auto type = (sym == Token::Kind::Sxgate) ? qc::SX : qc::SXdag;
            scan();

            auto target = argumentQreg();
            check(Token::Kind::Semicolon);

            if (target.second == 1) {
                return std::make_unique<qc::StandardOperation>(nqubits, target.first, type);
            }

            // TODO: multiple targets could be useful here
            auto gate = qc::CompoundOperation(nqubits);
            for (std::size_t i = 0; i < target.second; ++i) {
                gate.emplace_back<qc::StandardOperation>(nqubits, target.first + i, type);
            }
            return std::make_unique<qc::CompoundOperation>(std::move(gate));
        }
        if (sym == Token::Kind::Identifier) {
            scan();
            auto        gateName  = t.str;
            auto        cGateName = gateName;
            std::size_t ncontrols = 0;
            while (cGateName.front() == 'c') {
                cGateName = cGateName.substr(1);
                ncontrols++;
            }

            // special treatment for controlled swap
            if (cGateName == "swap") {
                std::vector<qc::QuantumRegister> arguments;
                argList(arguments);
                check(Token::Kind::Semicolon);
                qc::QuantumRegisterMap argMap;
                if (arguments.size() != ncontrols + 2) {
                    std::ostringstream oss{};
                    if (arguments.size() > ncontrols + 2) {
                        oss << "Too many arguments for ";
                    } else {
                        oss << "Too few arguments for ";
                    }
                    if (ncontrols > 1) {
                        oss << ncontrols << "-";
                    }
                    oss << "controlled swap-gate! Expected " << ncontrols << "+" << 2 << ", but got " << arguments.size();
                    error(oss.str());
                }

                for (size_t i = 0; i < arguments.size(); ++i) {
                    argMap["q" + std::to_string(i)] = arguments[i];
                    if (arguments[i].second > 1) {
                        error("cSWAP with whole qubit registers not yet implemented");
                    }
                }

                qc::Controls controls{};
                for (std::size_t j = 0; j < ncontrols; ++j) {
                    const auto arg = "q" + std::to_string(j);
                    controls.emplace(qc::Control{argMap.at(arg).first});
                }

                const auto targ  = "q" + std::to_string(ncontrols);
                const auto targ2 = "q" + std::to_string(ncontrols + 1);
                return std::make_unique<qc::StandardOperation>(nqubits, controls,
                                                               argMap.at(targ).first,
                                                               argMap.at(targ2).first,
                                                               qc::SWAP);
            }

            auto gateIt  = compoundGates.find(gateName);
            auto cGateIt = compoundGates.find(cGateName);
            if (gateIt != compoundGates.end() || cGateIt != compoundGates.end()) {
                std::vector<std::shared_ptr<Parser::Expr>> parameters;
                std::vector<qc::QuantumRegister>           arguments;
                if (sym == Token::Kind::Lpar) {
                    scan();
                    if (sym != Token::Kind::Rpar) {
                        expList(parameters);
                    }
                    check(Token::Kind::Rpar);
                }
                argList(arguments);
                check(Token::Kind::Semicolon);

                // return corresponding operation
                qc::QuantumRegisterMap                               argMap;
                std::map<std::string, std::shared_ptr<Parser::Expr>> paramMap;
                std::size_t                                          size = 1;
                if (gateIt != compoundGates.end()) {
                    if ((*gateIt).second.argumentNames.size() != arguments.size()) {
                        std::ostringstream oss{};
                        if ((*gateIt).second.argumentNames.size() < arguments.size()) {
                            oss << "Too many arguments for ";
                        } else {
                            oss << "Too few arguments for ";
                        }
                        oss << (*gateIt).first << " gate! Expected " << (*gateIt).second.argumentNames.size() << ", but got " << arguments.size();
                        error(oss.str());
                    }

                    for (size_t i = 0; i < arguments.size(); ++i) {
                        argMap[gateIt->second.argumentNames[i]] = arguments[i];
                        if (arguments[i].second > 1 && size != 1 && arguments[i].second != size) {
                            error("Register sizes do not match!");
                        }

                        if (arguments[i].second > 1) {
                            size = arguments[i].second;
                        }
                    }

                    for (size_t i = 0; i < parameters.size(); ++i) {
                        paramMap[gateIt->second.parameterNames[i]] = parameters[i];
                    }
                } else { // controlled Gate treatment
                    if (cGateIt->second.gates.size() > 1) {
                        std::ostringstream oss{};
                        oss << "Controlled operation '" << gateName << "' for which no definition was found, but a definition of a non-controlled gate '" << cGateName << "' was found. Arbitrary controlled gates without definition are currently not supported.";
                        error(oss.str());
                    }

                    if (arguments.size() != ncontrols + cGateIt->second.argumentNames.size()) {
                        std::ostringstream oss{};
                        if (arguments.size() > ncontrols + cGateIt->second.argumentNames.size()) {
                            oss << "Too many arguments for ";
                        } else {
                            oss << "Too few arguments for ";
                        }
                        if (ncontrols > 1) {
                            oss << ncontrols << "-";
                        }
                        oss << "controlled ";
                        oss << (*cGateIt).first << "-";
                        oss << "gate! Expected " << ncontrols << "+" << cGateIt->second.argumentNames.size() << ", but got " << arguments.size();

                        error(oss.str());
                    }

                    for (size_t i = 0; i < arguments.size(); ++i) {
                        argMap["q" + std::to_string(i)] = arguments[i];
                        if (arguments[i].second > 1 && size != 1 && arguments[i].second != size) {
                            error("Register sizes do not match!");
                        }

                        if (arguments[i].second > 1) {
                            size = arguments[i].second;
                        }
                    }

                    for (size_t i = 0; i < parameters.size(); ++i) {
                        paramMap[cGateIt->second.parameterNames[i]] = parameters[i];
                    }
                }

                // check if single controlled gate
                if (ncontrols > 0 && size == 1) {
                    // TODO: this could be enhanced for the case that any argument is a register
                    if (cGateIt->second.gates.size() == 1) {
                        qc::Controls controls{};
                        for (std::size_t j = 0; j < ncontrols; ++j) {
                            auto arg = (gateIt != compoundGates.end()) ? gateIt->second.argumentNames[j] : ("q" + std::to_string(j));
                            controls.emplace(qc::Control{argMap.at(arg).first});
                        }

                        auto targ = (gateIt != compoundGates.end()) ? gateIt->second.argumentNames.back() : ("q" + std::to_string(ncontrols));

                        // special treatment for Toffoli
                        if (cGateName == "x" && ncontrols > 1) {
                            return std::make_unique<qc::StandardOperation>(nqubits, controls, argMap.at(targ).first);
                        }

                        const auto cGate = cGateIt->second.gates.front();
                        for (size_t j = 0; j < parameters.size(); ++j) {
                            paramMap[cGateIt->second.parameterNames[j]] = parameters[j];
                        }

                        if (auto* cu = dynamic_cast<SingleQubitGate*>(cGate.get())) {
                            const auto theta  = rewriteExpr(cu->theta, paramMap);
                            const auto phi    = rewriteExpr(cu->phi, paramMap);
                            const auto lambda = rewriteExpr(cu->lambda, paramMap);

                            return std::make_unique<qc::StandardOperation>(nqubits, controls, argMap.at(targ).first, qc::U3, lambda->num, phi->num, theta->num);
                        }
                        error("Cast to u-Gate not possible for controlled operation.");
                    }
                }
                if (gateIt == compoundGates.end()) {
                    error("Controlled operation for which no definition could be found or which acts on whole qubit register.");
                }

                // identifier specifies just a single operation (U3 or CX)
                if (gateIt != compoundGates.end() && gateIt->second.gates.size() == 1) {
                    const auto gate = gateIt->second.gates.front();
                    if (auto* u = dynamic_cast<SingleQubitGate*>(gate.get())) {
                        const auto theta  = rewriteExpr(u->theta, paramMap);
                        const auto phi    = rewriteExpr(u->phi, paramMap);
                        const auto lambda = rewriteExpr(u->lambda, paramMap);

                        if (argMap.at(u->target).second == 1) {
                            return std::make_unique<qc::StandardOperation>(nqubits, argMap.at(u->target).first, qc::U3, lambda->num, phi->num, theta->num);
                        }
                    } else if (auto* cx = dynamic_cast<CXgate*>(gate.get())) {
                        if (argMap.at(cx->control).second == 1 && argMap.at(cx->target).second == 1) {
                            return std::make_unique<qc::StandardOperation>(nqubits, qc::Control{argMap.at(cx->control).first}, argMap.at(cx->target).first, qc::X);
                        }
                    }
                }

                qc::CompoundOperation op(nqubits);
                for (auto& gate: gateIt->second.gates) {
                    if (auto* u = dynamic_cast<SingleQubitGate*>(gate.get())) {
                        const auto theta  = rewriteExpr(u->theta, paramMap);
                        const auto phi    = rewriteExpr(u->phi, paramMap);
                        const auto lambda = rewriteExpr(u->lambda, paramMap);

                        if (argMap.at(u->target).second == 1) {
                            op.emplace_back<qc::StandardOperation>(nqubits, argMap.at(u->target).first, u->type,
                                                                   lambda ? lambda->num : 0.,
                                                                   phi ? phi->num : 0.,
                                                                   theta ? theta->num : 0.);
                        } else {
                            // TODO: multiple targets could be useful here
                            for (std::size_t j = 0; j < argMap.at(u->target).second; ++j) {
                                op.emplace_back<qc::StandardOperation>(nqubits, argMap.at(u->target).first + j, qc::U3, lambda->num, phi->num, theta->num);
                            }
                        }
                    } else if (auto* cx = dynamic_cast<CXgate*>(gate.get())) {
                        // valid check
                        for (std::size_t i = 0; i < argMap.at(cx->control).second; ++i) {
                            for (std::size_t j = 0; j < argMap.at(cx->target).second; ++j) {
                                if (argMap.at(cx->control).first + i == argMap.at(cx->target).first + j) {
                                    std::ostringstream oss{};
                                    oss << "Qubit " << argMap.at(cx->control).first + i << " cannot be control and target at the same time";
                                    error(oss.str());
                                }
                            }
                        }
                        if (argMap.at(cx->control).second == 1 && argMap.at(cx->target).second == 1) {
                            op.emplace_back<qc::StandardOperation>(nqubits, qc::Control{argMap.at(cx->control).first}, argMap.at(cx->target).first, qc::X);
                        } else if (argMap.at(cx->control).second == argMap.at(cx->target).second) {
                            for (std::size_t j = 0; j < argMap.at(cx->target).second; ++j) {
                                op.emplace_back<qc::StandardOperation>(nqubits, qc::Control{static_cast<qc::Qubit>(argMap.at(cx->control).first + j)}, argMap.at(cx->target).first + j, qc::X);
                            }
                        } else if (argMap.at(cx->control).second == 1) {
                            // TODO: multiple targets could be useful here
                            for (std::size_t k = 0; k < argMap.at(cx->target).second; ++k) {
                                op.emplace_back<qc::StandardOperation>(nqubits, qc::Control{argMap.at(cx->control).first}, argMap.at(cx->target).first + k, qc::X);
                            }
                        } else if (argMap.at(cx->target).second == 1) {
                            for (std::size_t l = 0; l < argMap.at(cx->control).second; ++l) {
                                op.emplace_back<qc::StandardOperation>(nqubits, qc::Control{static_cast<qc::Qubit>(argMap.at(cx->control).first + l)}, argMap.at(cx->target).first, qc::X);
                            }
                        } else {
                            error("Register size does not match for CX gate!");
                        }
                    } else if (auto* mcx = dynamic_cast<MCXgate*>(gate.get())) {
                        // valid check
                        for (const auto& control: mcx->controls) {
                            if (argMap.at(control).second != 1) {
                                error("Multi-controlled gates with whole qubit registers not supported");
                            }
                            if (argMap.at(control) == argMap.at(mcx->target)) {
                                std::ostringstream oss{};
                                oss << "Qubit " << argMap.at(mcx->target).first << " cannot be control and target at the same time";
                                error(oss.str());
                            }
                            if (std::count(mcx->controls.begin(), mcx->controls.end(), control) > 1) {
                                std::ostringstream oss{};
                                oss << "Qubit " << argMap.at(control).first << " cannot be control more than once";
                                error(oss.str());
                            }
                        }
                        if (argMap.at(mcx->target).second != 1) {
                            error("Multi-controlled gates with whole qubit registers not supported");
                        }

                        qc::Controls controls{};
                        for (const auto& control: mcx->controls) {
                            controls.emplace(qc::Control{argMap.at(control).first});
                        }
                        op.emplace_back<qc::StandardOperation>(nqubits, controls, argMap.at(mcx->target).first);
                    } else if (auto* cu = dynamic_cast<CUgate*>(gate.get())) {
                        // valid check
                        for (const auto& control: cu->controls) {
                            if (argMap.at(control).second != 1) {
                                error("Multi-controlled gates with whole qubit registers not supported");
                            }
                            if (argMap.at(control) == argMap.at(cu->target)) {
                                std::ostringstream oss{};
                                oss << "Qubit " << argMap.at(cu->target).first << " cannot be control and target at the same time";
                                error(oss.str());
                            }
                            if (std::count(cu->controls.begin(), cu->controls.end(), control) > 1) {
                                std::ostringstream oss{};
                                oss << "Qubit " << argMap.at(control).first << " cannot be control more than once";
                                error(oss.str());
                            }
                        }

                        const auto theta  = rewriteExpr(cu->theta, paramMap);
                        const auto phi    = rewriteExpr(cu->phi, paramMap);
                        const auto lambda = rewriteExpr(cu->lambda, paramMap);

                        qc::Controls controls{};
                        for (const auto& control: cu->controls) {
                            controls.emplace(qc::Control{argMap.at(control).first});
                        }

                        if (argMap.at(cu->target).second == 1) {
                            op.emplace_back<qc::StandardOperation>(nqubits, controls, argMap.at(cu->target).first, qc::U3, lambda->num, phi->num, theta->num);
                        } else if (auto* sw = dynamic_cast<SWAPgate*>(gate.get())) {
                            // valid check
                            for (std::size_t i = 0; i < argMap.at(sw->target0).second; ++i) {
                                for (std::size_t j = 0; j < argMap.at(sw->target1).second; ++j) {
                                    if (argMap.at(sw->target0).first + i == argMap.at(sw->target1).first + j) {
                                        std::ostringstream oss{};
                                        oss << "Qubit " << argMap.at(sw->target0).first + i << " cannot be swap target twice";
                                        error(oss.str());
                                    }
                                }
                            }
                            if (argMap.at(sw->target0).second == 1 && argMap.at(sw->target1).second == 1) {
                                op.emplace_back<qc::StandardOperation>(nqubits, qc::Controls{}, argMap.at(sw->target1).first, argMap.at(sw->target1).first, qc::SWAP);
                            } else if (argMap.at(sw->target0).second == argMap.at(sw->target1).second) {
                                for (std::size_t j = 0; j < argMap.at(sw->target1).second; ++j) {
                                    op.emplace_back<qc::StandardOperation>(nqubits, qc::Controls{}, argMap.at(sw->target0).first + j, argMap.at(sw->target1).first + j, qc::SWAP);
                                }
                            } else if (argMap.at(sw->target0).second == 1) {
                                // TODO: multiple targets could be useful here
                                for (std::size_t k = 0; k < argMap.at(sw->target1).second; ++k) {
                                    op.emplace_back<qc::StandardOperation>(nqubits, qc::Controls{}, argMap.at(sw->target0).first, argMap.at(sw->target1).first + k, qc::SWAP);
                                }
                            } else if (argMap.at(sw->target1).second == 1) {
                                for (std::size_t l = 0; l < argMap.at(sw->target0).second; ++l) {
                                    op.emplace_back<qc::StandardOperation>(nqubits, qc::Controls{}, argMap.at(sw->target0).first + l, argMap.at(sw->target1).first, qc::SWAP);
                                }
                            } else {
                                error("Register size does not match for SWAP gate!");
                            }
                        } else {
                            error("Multi-controlled gates with whole qubit registers not supported");
                        }
                    } else {
                        error("Could not cast to any known gate type");
                    }
                }
                return std::make_unique<qc::CompoundOperation>(std::move(op));
            }
            error("Undefined gate " + t.str);
        } else {
            error("Symbol " + qasm::KIND_NAMES.at(sym) + " not expected in Gate() routine!");
        }
    }

    void Parser::opaqueGateDecl() {
        check(Token::Kind::Opaque);
        check(Token::Kind::Identifier);

        CompoundGate gate;
        auto         gateName = t.str;
        if (sym == Token::Kind::Lpar) {
            scan();
            if (sym != Token::Kind::Rpar) {
                idList(gate.argumentNames);
            }
            check(Token::Kind::Rpar);
        }
        idList(gate.argumentNames);
        compoundGates[gateName] = gate;
        check(Token::Kind::Semicolon);
    }

    void Parser::gateDecl() {
        check(Token::Kind::Gate);
        // skip declarations of known gates
        if (sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive || sym == Token::Kind::McxVchain || sym == Token::Kind::Mcphase || sym == Token::Kind::Swap || sym == Token::Kind::Sxgate || sym == Token::Kind::Sxdggate) {
            while (sym != Token::Kind::Rbrace) {
                scan();
            }

            check(Token::Kind::Rbrace);
            return;
        }
        check(Token::Kind::Identifier);

        CompoundGate      gate;
        const std::string gateName = t.str;
        if (sym == Token::Kind::Lpar) {
            scan();
            if (sym != Token::Kind::Rpar) {
                idList(gate.parameterNames);
            }
            check(Token::Kind::Rpar);
        }
        idList(gate.argumentNames);
        check(Token::Kind::Lbrace);

        auto        cGateName = gateName;
        std::size_t ncontrols = 0;
        while (cGateName.front() == 'c') {
            cGateName = cGateName.substr(1);
            ncontrols++;
        }
        // see if non-controlled version (consisting of a single gate) already available
        auto controlledGateIt = compoundGates.find(cGateName);
        if (controlledGateIt != compoundGates.end() && controlledGateIt->second.gates.size() <= 1) {
            // skip over gate declaration
            while (sym != Token::Kind::Rbrace) {
                scan();
            }
            scan();
            return;
        }

        while (sym != Token::Kind::Rbrace) {
            if (sym == Token::Kind::Ugate) {
                scan();
                check(Token::Kind::Lpar);
                auto theta = exp();
                check(Token::Kind::Comma);
                auto phi = exp();
                check(Token::Kind::Comma);
                auto lambda = exp();
                check(Token::Kind::Rpar);
                check(Token::Kind::Identifier);
                gate.gates.push_back(std::make_shared<SingleQubitGate>(t.str, qc::U3, lambda, phi, theta));
                check(Token::Kind::Semicolon);
            } else if (sym == Token::Kind::Sxgate || sym == Token::Kind::Sxdggate) {
                const auto gateType = sym == Token::Kind::Sxgate ? qc::SX : qc::SXdag;
                scan();
                check(Token::Kind::Identifier);
                gate.gates.push_back(std::make_shared<SingleQubitGate>(t.str, gateType));
                check(Token::Kind::Semicolon);
            } else if (sym == Token::Kind::Cxgate) {
                scan();
                check(Token::Kind::Identifier);
                const std::string control = t.str;
                check(Token::Kind::Comma);
                check(Token::Kind::Identifier);
                gate.gates.push_back(std::make_shared<CXgate>(control, t.str));
                check(Token::Kind::Semicolon);
            } else if (sym == Token::Kind::Swap) {
                scan();
                check(Token::Kind::Identifier);
                auto target0 = t.str;
                check(Token::Kind::Comma);
                check(Token::Kind::Identifier);
                auto target1 = t.str;
                gate.gates.push_back(std::make_shared<SWAPgate>(target0, target1));
                check(Token::Kind::Semicolon);
            } else if (sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive || sym == Token::Kind::McxVchain) {
                auto type = sym;
                scan();
                std::vector<std::string> arguments{};
                check(Token::Kind::Identifier);
                arguments.emplace_back(t.str);
                while (sym != Token::Kind::Semicolon) {
                    check(Token::Kind::Comma);
                    check(Token::Kind::Identifier);
                    arguments.emplace_back(t.str);
                }
                scan();

                // drop ancillaries since our library can natively work with MCTs
                if (type == Token::Kind::McxVchain) {
                    const auto ancillaries = (arguments.size() + 1) / 2 - 2;
                    for (std::size_t i = 0; i < ancillaries; ++i) {
                        arguments.pop_back();
                    }
                } else if (type == Token::Kind::McxRecursive) {
                    // 1 ancillary if more than 4 controls
                    if (arguments.size() > 5) {
                        arguments.pop_back();
                    }
                }

                auto target = arguments.back();
                arguments.pop_back();
                gate.gates.push_back(std::make_shared<MCXgate>(arguments, target));
            } else if (sym == Token::Kind::Mcphase) {
                scan();
                check(Token::Kind::Lpar);
                auto lambda = exp();
                check(Token::Kind::Rpar);
                std::vector<std::string> arguments{};
                check(Token::Kind::Identifier);
                arguments.emplace_back(t.str);
                while (sym != Token::Kind::Semicolon) {
                    check(Token::Kind::Comma);
                    check(Token::Kind::Identifier);
                    arguments.emplace_back(t.str);
                }
                scan();
                auto target = arguments.back();
                arguments.pop_back();
                auto theta = std::make_shared<Expr>(Expr::Kind::Number);
                auto phi   = std::make_shared<Expr>(Expr::Kind::Number);
                gate.gates.push_back(std::make_shared<CUgate>(theta, phi, lambda, arguments, target));
            } else if (sym == Token::Kind::Identifier) {
                scan();
                const std::string name = t.str;

                cGateName = name;
                ncontrols = 0;
                while (cGateName.front() == 'c') {
                    cGateName = cGateName.substr(1);
                    ncontrols++;
                }

                // see if non-controlled version already available
                auto gateIt  = compoundGates.find(name);
                auto cGateIt = compoundGates.find(cGateName);
                if (gateIt != compoundGates.end() || cGateIt != compoundGates.end()) {
                    std::vector<std::shared_ptr<Parser::Expr>> parameters;
                    std::vector<std::string>                   arguments;
                    if (sym == Token::Kind::Lpar) {
                        scan();
                        if (sym != Token::Kind::Rpar) {
                            expList(parameters);
                        }
                        check(Token::Kind::Rpar);
                    }
                    idList(arguments);
                    check(Token::Kind::Semicolon);

                    std::map<std::string, std::string>                   argMap;
                    std::map<std::string, std::shared_ptr<Parser::Expr>> paramMap;
                    if (gateIt != compoundGates.end()) {
                        if ((*gateIt).second.argumentNames.size() != arguments.size()) {
                            std::ostringstream oss{};
                            if ((*gateIt).second.argumentNames.size() < arguments.size()) {
                                oss << "Too many arguments for ";
                            } else {
                                oss << "Too few arguments for ";
                            }
                            oss << (*gateIt).first << " gate! Expected " << (*gateIt).second.argumentNames.size() << ", but got " << arguments.size();
                            error(oss.str());
                        }

                        for (size_t i = 0; i < arguments.size(); ++i) {
                            argMap[gateIt->second.argumentNames[i]] = arguments[i];
                        }
                        for (size_t i = 0; i < parameters.size(); ++i) {
                            paramMap[gateIt->second.parameterNames[i]] = parameters[i];
                        }

                        for (auto& it: gateIt->second.gates) {
                            if (auto* u = dynamic_cast<SingleQubitGate*>(it.get())) {
                                gate.gates.push_back(std::make_shared<SingleQubitGate>(argMap.at(u->target), u->type, rewriteExpr(u->lambda, paramMap), rewriteExpr(u->phi, paramMap), rewriteExpr(u->theta, paramMap)));
                            } else if (auto* cx = dynamic_cast<CXgate*>(it.get())) {
                                gate.gates.push_back(std::make_shared<CXgate>(argMap.at(cx->control), argMap.at(cx->target)));
                            } else if (auto* cu = dynamic_cast<CUgate*>(it.get())) {
                                std::vector<std::string> controls{};
                                for (const auto& control: cu->controls) {
                                    controls.emplace_back(argMap.at(control));
                                }
                                gate.gates.push_back(std::make_shared<CUgate>(rewriteExpr(cu->theta, paramMap), rewriteExpr(cu->phi, paramMap), rewriteExpr(cu->lambda, paramMap), controls, argMap.at(cu->target)));
                            } else if (auto* mcx = dynamic_cast<MCXgate*>(it.get())) {
                                std::vector<std::string> controls{};
                                for (const auto& control: mcx->controls) {
                                    controls.emplace_back(argMap.at(control));
                                }
                                gate.gates.push_back(std::make_shared<MCXgate>(controls, argMap.at(mcx->target)));
                            } else {
                                error("Unexpected gate in GateDecl!");
                            }
                        }
                    } else {
                        if (cGateIt->second.gates.size() != 1) {
                            throw QASMParserException("Gate declaration with controlled gates inferred from internal qelib1.inc not yet implemented.");
                        }

                        if (arguments.size() != ncontrols + 1) {
                            std::ostringstream oss{};
                            if (arguments.size() > ncontrols + 1) {
                                oss << "Too many arguments for ";
                            } else {
                                oss << "Too few arguments for ";
                            }
                            if (ncontrols > 1) {
                                oss << ncontrols << "-";
                            }
                            oss << "controlled ";
                            oss << (*cGateIt).first << "-";
                            oss << "gate! Expected " << ncontrols << "+1, but got " << arguments.size();

                            error(oss.str());
                        }

                        for (size_t i = 0; i < arguments.size(); ++i) {
                            argMap["q" + std::to_string(i)] = arguments[i];
                        }

                        for (size_t i = 0; i < parameters.size(); ++i) {
                            paramMap[cGateIt->second.parameterNames[i]] = parameters[i];
                        }

                        if (cGateName == "x" || cGateName == "X") {
                            std::vector<std::string> controls{};
                            for (size_t i = 0; i < arguments.size() - 1; ++i) {
                                controls.emplace_back(arguments[i]);
                            }
                            gate.gates.push_back(std::make_shared<MCXgate>(controls, arguments.back()));
                        } else {
                            std::vector<std::string> controls{};
                            for (size_t i = 0; i < arguments.size() - 1; ++i) {
                                controls.emplace_back(arguments[i]);
                            }
                            if (auto* u = dynamic_cast<SingleQubitGate*>(cGateIt->second.gates.at(0).get())) {
                                gate.gates.push_back(std::make_shared<CUgate>(rewriteExpr(u->theta, paramMap), rewriteExpr(u->phi, paramMap), rewriteExpr(u->lambda, paramMap), controls, arguments.back()));
                            } else {
                                throw QASMParserException("Could not cast to UGate in gate declaration.");
                            }
                        }
                    }
                } else {
                    error("Undefined gate " + t.str);
                }
            } else if (sym == Token::Kind::Barrier) {
                scan();
                std::vector<std::string> arguments;
                idList(arguments);
                check(Token::Kind::Semicolon);
                //Nothing to do here for the simulator
            } else if (sym == Token::Kind::Comment) {
                scan();
                handleComment();
            } else {
                error("Error in gate declaration!");
            }
        }
        compoundGates[gateName] = gate;
        check(Token::Kind::Rbrace);
    }

    std::unique_ptr<qc::Operation> Parser::qop() {
        if (sym == Token::Kind::Ugate || sym == Token::Kind::Cxgate ||
            sym == Token::Kind::Swap || sym == Token::Kind::Identifier ||
            sym == Token::Kind::Sxgate || sym == Token::Kind::Sxdggate ||
            sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive || sym == Token::Kind::McxVchain || sym == Token::Kind::Mcphase) {
            return gate();
        }
        if (sym == Token::Kind::Measure) {
            scan();
            auto qreg = argumentQreg();
            check(Token::Kind::Minus);
            check(Token::Kind::Gt);
            auto creg = argumentCreg();
            check(Token::Kind::Semicolon);

            if (qreg.second == creg.second) {
                std::vector<qc::Qubit> qubits{};
                std::vector<qc::Bit>   classics{};
                for (std::size_t i = 0; i < qreg.second; ++i) {
                    const auto qubit = qreg.first + i;
                    const auto clbit = creg.first + i;
                    if (qubit >= nqubits) {
                        std::stringstream ss{};
                        ss << "Qubit " << qubit << " cannot be measured since the circuit only contains " << nqubits << " qubits";
                        error(ss.str());
                    }
                    if (clbit >= nclassics) {
                        std::stringstream ss{};
                        ss << "Bit " << clbit << " cannot be target of a measurement since the circuit only contains " << nclassics << " classical bits";
                        error(ss.str());
                    }
                    qubits.emplace_back(qubit);
                    classics.emplace_back(clbit);
                }
                return std::make_unique<qc::NonUnitaryOperation>(nqubits, qubits, classics);
            }
            error("Mismatch of qreg and creg size in measurement");
        }
        if (sym == Token::Kind::Reset) {
            scan();
            auto qreg = argumentQreg();
            check(Token::Kind::Semicolon);

            std::vector<qc::Qubit> qubits;
            for (std::size_t i = 0; i < qreg.second; ++i) {
                auto qubit = qreg.first + i;
                if (qubit >= nqubits) {
                    std::stringstream ss{};
                    ss << "Qubit " << qubit << " cannot be reset since the circuit only contains " << nqubits << " qubits";
                    error(ss.str());
                }
                qubits.emplace_back(qubit);
            }
            return std::make_unique<qc::NonUnitaryOperation>(nqubits, qubits);
        }
        error("No valid Qop: " + t.str);
    }
} // namespace qasm
