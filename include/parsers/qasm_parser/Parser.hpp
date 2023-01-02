/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Definitions.hpp"
#include "Scanner.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/StandardOperation.hpp"

#include <cmath>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qasm {
    class QASMParserException: public std::invalid_argument {
        std::string msg;

    public:
        explicit QASMParserException(const std::string& m):
            std::invalid_argument("QASM Parser Exception") {
            std::stringstream ss{};
            ss << "[qasm parser] " << m;
            msg = ss.str();
        }

        [[nodiscard]] const char* what() const noexcept override {
            return msg.c_str();
        }
    };

    class Parser {
        struct Expr {
            enum class Kind {
                Number,
                Plus,
                Minus,
                Sign,
                Times,
                Sin,
                Cos,
                Tan,
                Exp,
                Ln,
                Sqrt,
                Div,
                Power,
                Id
            };
            qc::fp                num;
            Kind                  kind;
            std::shared_ptr<Expr> op1 = nullptr;
            std::shared_ptr<Expr> op2 = nullptr;
            std::string           id;

            explicit Expr(Kind k, qc::fp n = 0., std::shared_ptr<Expr> operation1 = nullptr, std::shared_ptr<Expr> operation2 = nullptr, std::string identifier = ""):
                num(n), kind(k), op1(std::move(operation1)), op2(std::move(operation2)), id(std::move(identifier)) {}
            Expr(const Expr& expr):
                num(expr.num), kind(expr.kind), id(expr.id) {
                if (expr.op1 != nullptr) {
                    op1 = expr.op1;
                }
                if (expr.op2 != nullptr) {
                    op2 = expr.op2;
                }
            }
            Expr& operator=(const Expr& expr) {
                if (&expr == this) {
                    return *this;
                }

                num  = expr.num;
                kind = expr.kind;
                id   = expr.id;

                op1 = expr.op1;
                op2 = expr.op2;

                return *this;
            }

            virtual ~Expr() = default;
        };

        struct BasisGate {
            virtual ~BasisGate() = default;
        };

        struct CUgate: public BasisGate {
            std::shared_ptr<Expr>    theta  = nullptr;
            std::shared_ptr<Expr>    phi    = nullptr;
            std::shared_ptr<Expr>    lambda = nullptr;
            std::vector<std::string> controls;
            std::string              target;

            CUgate(std::shared_ptr<Expr> t, std::shared_ptr<Expr> p, std::shared_ptr<Expr> l, std::vector<std::string> c, std::string targ):
                theta(std::move(t)), phi(std::move(p)), lambda(std::move(l)), controls(std::move(c)), target(std::move(targ)) {}
        };

        struct CXgate: public BasisGate {
            std::string control;
            std::string target;

            CXgate(std::string c, std::string t):
                control(std::move(c)), target(std::move(t)) {}
        };

        struct SingleQubitGate: public BasisGate {
            std::string           target;
            qc::OpType            type;
            std::shared_ptr<Expr> lambda;
            std::shared_ptr<Expr> phi;
            std::shared_ptr<Expr> theta;

            explicit SingleQubitGate(std::string targ, qc::OpType typ = qc::U3, std::shared_ptr<Expr> l = nullptr, std::shared_ptr<Expr> p = nullptr, std::shared_ptr<Expr> t = nullptr):
                target(std::move(targ)), type(typ), lambda(std::move(l)), phi(std::move(p)), theta(std::move(t)) {}
        };

        struct SWAPgate: public BasisGate {
            std::string target0;
            std::string target1;

            SWAPgate(std::string t0, std::string t1):
                target0(std::move(t0)), target1(std::move(t1)) {}
        };

        struct MCXgate: public BasisGate {
            std::vector<std::string> controls;
            std::string              target;

            MCXgate(std::vector<std::string> c, std::string t):
                controls(std::move(c)), target(std::move(t)) {}
        };

        struct CompoundGate {
            std::vector<std::string>                parameterNames;
            std::vector<std::string>                argumentNames;
            std::vector<std::shared_ptr<BasisGate>> gates;
        };

        std::istream&                       in;
        std::set<Token::Kind>               unaryops{Token::Kind::Sin, Token::Kind::Cos, Token::Kind::Tan, Token::Kind::Exp, Token::Kind::Ln, Token::Kind::Sqrt};
        std::map<std::string, CompoundGate> compoundGates;

        std::shared_ptr<Expr> exponentiation();
        std::shared_ptr<Expr> factor();
        std::shared_ptr<Expr> term();
        std::shared_ptr<Expr> exp();

        static std::shared_ptr<Expr> rewriteExpr(const std::shared_ptr<Expr>& expr, std::map<std::string, std::shared_ptr<Expr>>& exprMap);

    public:
        Token                     la, t;
        Token::Kind               sym = Token::Kind::None;
        std::shared_ptr<Scanner>  scanner;
        qc::QuantumRegisterMap&   qregs;
        qc::ClassicalRegisterMap& cregs;
        std::size_t               nqubits   = 0;
        std::size_t               nclassics = 0;
        qc::Permutation           initialLayout{};
        qc::Permutation           outputPermutation{};

        explicit Parser(std::istream& is, qc::QuantumRegisterMap& q, qc::ClassicalRegisterMap& c):
            in(is), qregs(q), cregs(c) {
            scanner = std::make_shared<Scanner>(in);
        }

        virtual ~Parser() = default;

        void scan();

        void check(Token::Kind expected);

        qc::QuantumRegister argumentQreg();

        qc::ClassicalRegister argumentCreg();

        void expList(std::vector<std::shared_ptr<Expr>>& expressions);

        void argList(std::vector<qc::QuantumRegister>& arguments);

        void idList(std::vector<std::string>& identifiers);

        std::unique_ptr<qc::Operation> gate();

        void opaqueGateDecl();

        void gateDecl();

        std::unique_ptr<qc::Operation> qop();

        void error [[noreturn]] (const std::string& msg) const {
            std::ostringstream oss{};
            oss << "l:" << t.line << " c:" << t.col << " msg: " << msg;
            throw QASMParserException(oss.str());
        }

        void handleComment();
        // check string for I/O layout information of the form
        //      'i Q_i Q_j ... Q_k' meaning, e.g. q_0 is mapped to Q_i, q_1 to Q_j, etc.
        //      'o Q_i Q_j ... Q_k' meaning, e.g. q_0 is found at Q_i, q_1 at Q_j, etc.
        // where i describes the initial layout, e.g. 'i 2 1 0' means q0 -> Q2, q1 -> Q1, q2 -> Q0
        // and o describes the output permutation, e.g. 'o 2 1 0' means  q0 is expected at Q2, q1 at Q1, and q2 at Q0
        static qc::Permutation checkForInitialLayout(std::string comment);
        static qc::Permutation checkForOutputPermutation(std::string comment);
    };

} // namespace qasm
