/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_PARSER_H
#define INTERMEDIATEREPRESENTATION_PARSER_H

#include <utility>
#include <vector>
#include <set>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Scanner.hpp"
#include "operations/StandardOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/CompoundOperation.hpp"

namespace qasm {
	static constexpr long double PI = 3.14159265358979323846264338327950288419716939937510L;

	using registerMap = std::map<std::string, std::pair<unsigned short, unsigned short>, std::greater<>>;

	class QASMParserException : public std::invalid_argument {
		std::string msg;
	public:
		explicit QASMParserException(const std::string& msg) : std::invalid_argument("QASM Parser Exception") {
			std::stringstream ss{};
			ss << "[qasm parser] " << msg;
			this->msg = ss.str();
		}

		const char *what() const noexcept override {
			return msg.c_str();
		}
	};

	class Parser {

		struct Expr {
			enum class Kind {
				number, plus, minus, sign, times, sin, cos, tan, exp, ln, sqrt, div, power, id
			};
			fp num;
			Kind kind;
			Expr* op1 = nullptr;
			Expr* op2 = nullptr;
			std::string id;

			explicit Expr(Kind kind, fp num = 0L, Expr *op1 = nullptr, Expr *op2 = nullptr, std::string id = "") : num(num), kind(kind), op1(op1), op2(op2), id(std::move(id)) { }
			Expr(const Expr& expr): num(expr.num), kind(expr.kind), id(expr.id) {
				if (expr.op1 != nullptr)
					op1 = new Expr(*expr.op1);
				if (expr.op2 != nullptr)
					op2 = new Expr(*expr.op2);
			}
			Expr& operator=(const Expr& expr) {
				if (&expr == this)
					return *this;

				num = expr.num;
				kind = expr.kind;
				id = expr.id;
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
		
		struct Ugate : public BasisGate {
			Expr *theta = nullptr;
			Expr *phi = nullptr;
			Expr *lambda = nullptr;
			std::string target;

			Ugate(Expr *theta, Expr *phi, Expr *lambda, std::string  target) : theta(theta), phi(phi), lambda(lambda), target(std::move(target)) { }

			~Ugate() override {
				delete theta;
				delete phi;
				delete lambda;
			}
		};

		struct CUgate : public BasisGate {
			Expr *theta = nullptr;
			Expr *phi = nullptr;
			Expr *lambda = nullptr;
			std::vector<std::string> controls;
			std::string target;

			CUgate(Expr *theta, Expr *phi, Expr *lambda, std::vector<std::string> controls, std::string  target) : theta(theta), phi(phi), lambda(lambda), controls(std::move(controls)), target(std::move(target)) { }

			~CUgate() override {
				delete theta;
				delete phi;
				delete lambda;
			}
		};

		struct CXgate : public BasisGate {
			std::string control;
			std::string target;

			CXgate(std::string  control, std::string  target) : control(std::move(control)), target(std::move(target)) { }
		};

		struct SWAPgate : public BasisGate {
			std::string target0;
			std::string target1;

			SWAPgate(std::string target0, std::string target1) : target0(std::move(target0)), target1(std::move(target1)) { }
		};

		struct MCXgate : public BasisGate {
			std::vector<std::string> controls;
			std::string target;

			MCXgate(std::vector<std::string> controls, std::string target) : controls(std::move(controls)), target(std::move(target)) { }
		};
		
		struct CompoundGate {
			std::vector<std::string> parameterNames;
			std::vector<std::string> argumentNames;
			std::vector<BasisGate*>  gates;
		};

		std::istream&                       in;
		std::set<Token::Kind>               unaryops{ Token::Kind::sin, Token::Kind::cos, Token::Kind::tan, Token::Kind::exp, Token::Kind::ln, Token::Kind::sqrt };
		std::map<std::string, CompoundGate> compoundGates;

		Expr* Exponentiation();
		Expr* Factor();
		Expr* Term();
		Expr* Exp();

		static Expr *RewriteExpr(Expr *expr, std::map<std::string, Expr *>& exprMap);

	public:
		Token          la, t;
		Token::Kind    sym = Token::Kind::none;
		Scanner       *scanner;
		registerMap&   qregs;
		registerMap&   cregs;
		unsigned short nqubits = 0;
		unsigned short nclassics = 0;

		explicit Parser(std::istream& is, registerMap& qregs, registerMap& cregs) :in(is), qregs(qregs), cregs(cregs) {
			scanner = new Scanner(in);
		}

		virtual ~Parser() {
			delete scanner;

			for (auto& cGate:compoundGates)
				for (auto& gate: cGate.second.gates)
					delete gate;
		}

		void scan();

		void check(Token::Kind expected);

		std::pair<unsigned short , unsigned short> ArgumentQreg();

		std::pair<unsigned short, unsigned short> ArgumentCreg();

		void ExpList(std::vector<Expr*>& expressions);

		void ArgList(std::vector<std::pair<unsigned short, unsigned short>>& arguments);

		void IdList(std::vector<std::string>& identifiers);

		std::unique_ptr<qc::Operation> Gate();

		void OpaqueGateDecl();

		void GateDecl();

		std::unique_ptr<qc::Operation> Qop();

		void error [[ noreturn ]](const std::string& msg) const {
			std::ostringstream oss{};
			oss << "l:" << t.line << " c:" << t.col << " msg: " << msg;
			throw QASMParserException(oss.str());
		}
	};

}
#endif //INTERMEDIATEREPRESENTATION_PARSER_H
