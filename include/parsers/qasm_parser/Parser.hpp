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
        explicit QASMParserException(const std::string& msg):
            std::invalid_argument("QASM Parser Exception") {
            std::stringstream ss{};
            ss << "[qasm parser] " << msg;
            this->msg = ss.str();
        }

        [[nodiscard]] const char* what() const noexcept override {
            return msg.c_str();
        }
    };

    class Parser {
        struct Expr {
            enum class Kind {
                number,
                plus,
                minus,
                sign,
                times,
                sin,
                cos,
                tan,
                exp,
                ln,
                sqrt,
                div,
                power,
                id
            };
            dd::fp      num;
            Kind        kind;
            Expr*       op1 = nullptr;
            Expr*       op2 = nullptr;
            std::string id;

            explicit Expr(Kind kind, dd::fp num = 0., Expr* op1 = nullptr, Expr* op2 = nullptr, std::string id = ""):
                num(num), kind(kind), op1(op1), op2(op2), id(std::move(id)) {}
            Expr(const Expr& expr):
                num(expr.num), kind(expr.kind), id(expr.id) {
                if (expr.op1 != nullptr)
                    op1 = new Expr(*expr.op1);
                if (expr.op2 != nullptr)
                    op2 = new Expr(*expr.op2);
            }
            Expr& operator=(const Expr& expr) {
                if (&expr == this)
                    return *this;

                num  = expr.num;
                kind = expr.kind;
                id   = expr.id;
                delete op1;
                delete op2;

                if (expr.op1 != nullptr)
                    op1 = new Expr(*expr.op1);
                else
                    op1 = nullptr;

                if (expr.op2 != nullptr)
                    op2 = new Expr(*expr.op2);
                else
                    op2 = nullptr;

                return *this;
            }

            virtual ~Expr() {
                delete op1;
                delete op2;
            }
        };

        struct BasisGate {
            virtual ~BasisGate() = default;
        };

        struct CUgate: public BasisGate {
            Expr*                    theta  = nullptr;
            Expr*                    phi    = nullptr;
            Expr*                    lambda = nullptr;
            std::vector<std::string> controls;
            std::string              target;

            CUgate(Expr* theta, Expr* phi, Expr* lambda, std::vector<std::string> controls, std::string target):
                theta(theta), phi(phi), lambda(lambda), controls(std::move(controls)), target(std::move(target)) {}

            ~CUgate() override {
                delete theta;
                delete phi;
                delete lambda;
            }
        };

        struct CXgate: public BasisGate {
            std::string control;
            std::string target;

            CXgate(std::string control, std::string target):
                control(std::move(control)), target(std::move(target)) {}
        };

        struct SingleQubitGate: public BasisGate {
            std::string target;
            qc::OpType  type;
            Expr*       lambda;
            Expr*       phi;
            Expr*       theta;

            explicit SingleQubitGate(std::string target, qc::OpType type = qc::U3, Expr* lambda = nullptr, Expr* phi = nullptr, Expr* theta = nullptr):
                target(std::move(target)), type(type), lambda(lambda), phi(phi), theta(theta) {}
            ~SingleQubitGate() override {
                delete lambda;
                delete phi;
                delete theta;
            }
        };

        struct SWAPgate: public BasisGate {
            std::string target0;
            std::string target1;

            SWAPgate(std::string target0, std::string target1):
                target0(std::move(target0)), target1(std::move(target1)) {}
        };

        struct MCXgate: public BasisGate {
            std::vector<std::string> controls;
            std::string              target;

            MCXgate(std::vector<std::string> controls, std::string target):
                controls(std::move(controls)), target(std::move(target)) {}
        };

        struct CompoundGate {
            std::vector<std::string> parameterNames;
            std::vector<std::string> argumentNames;
            std::vector<BasisGate*>  gates;
        };

        std::istream&                       in;
        std::set<Token::Kind>               unaryops{Token::Kind::sin, Token::Kind::cos, Token::Kind::tan, Token::Kind::exp, Token::Kind::ln, Token::Kind::sqrt};
        std::map<std::string, CompoundGate> compoundGates;

        Expr* Exponentiation();
        Expr* Factor();
        Expr* Term();
        Expr* Exp();

        static Expr* RewriteExpr(Expr* expr, std::map<std::string, Expr*>& exprMap);

    public:
        Token                     la, t;
        Token::Kind               sym = Token::Kind::none;
        Scanner*                  scanner;
        qc::QuantumRegisterMap&   qregs;
        qc::ClassicalRegisterMap& cregs;
        dd::QubitCount            nqubits   = 0;
        std::size_t               nclassics = 0;
        qc::Permutation           initialLayout{};
        qc::Permutation           outputPermutation{};

        explicit Parser(std::istream& is, qc::QuantumRegisterMap& qregs, qc::ClassicalRegisterMap& cregs):
            in(is), qregs(qregs), cregs(cregs) {
            scanner = new Scanner(in);
        }

        virtual ~Parser() {
            delete scanner;

            for (auto& cGate: compoundGates)
                for (auto& gate: cGate.second.gates)
                    delete gate;
        }

        void scan();

        void check(Token::Kind expected);

        qc::QuantumRegister ArgumentQreg();

        qc::ClassicalRegister ArgumentCreg();

        void ExpList(std::vector<Expr*>& expressions);

        void ArgList(std::vector<qc::QuantumRegister>& arguments);

        void IdList(std::vector<std::string>& identifiers);

        std::unique_ptr<qc::Operation> Gate();

        void OpaqueGateDecl();

        void GateDecl();

        std::unique_ptr<qc::Operation> Qop();

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
