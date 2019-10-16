//
// Created by Lukas Burgholzer on 22.10.19.
//

#ifndef INTERMEDIATEREPRESENTATION_PARSER_H
#define INTERMEDIATEREPRESENTATION_PARSER_H

#include <utility>
#include <vector>
#include <set>
#include <cmath>
#include <QuantumComputation.h>

#include "Scanner.h"
#include "Operations/StandardOperation.h"
#include "Operations/NonUnitaryOperation.h"
#include "Operations/CompoundOperation.h"

namespace qasm {

	static constexpr long double PI = 3.14159265358979323846264338327950288419716939937510L;

	using registerMap = std::map<std::string, std::pair<unsigned short, unsigned short>>;

	class Parser {

		struct Expr {
			enum class Kind {
				number, plus, minus, sign, times, sin, cos, tan, exp, ln, sqrt, div, power, id
			};
			long double num;
			Kind kind;
			Expr* op1 = nullptr;
			Expr* op2 = nullptr;
			std::string id = "";

			explicit Expr(Kind kind, long double num = 0L, Expr *op1 = nullptr, Expr *op2 = nullptr, std::string id = "") : num(num), kind(kind), op1(op1), op2(op2), id(std::move(id)) { }
			Expr(const Expr& expr): num(expr.num), kind(expr.kind), id(expr.id) {
				if (expr.op1 != nullptr)
					op1 = new Expr(*expr.op1);
				if (expr.op2 != nullptr)
					op2 = new Expr(*expr.op2);
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
		struct CXgate : public BasisGate {
			std::string control;
			std::string target;

			CXgate(std::string  control, std::string  target) : control(std::move(control)), target(std::move(target)) { }
		};
		struct CompoundGate {
			std::vector<std::string> parameterNames;
			std::vector<std::string> argumentNames;
			std::vector<BasisGate*> gates;
		};
		std::istream& in;

		std::set<Token::Kind> unaryops{ Token::Kind::sin, Token::Kind::cos, Token::Kind::tan, Token::Kind::exp, Token::Kind::ln, Token::Kind::sqrt };
		std::map<std::string, CompoundGate> compoundGates;

		Expr* Exponentiation() {
			Expr* x;
			if(sym == Token::Kind::real) {
				scan();
				return new Expr(Expr::Kind::number, t.valReal);
			} else if (sym == Token::Kind::nninteger) {
				scan();
				return new Expr(Expr::Kind::number, t.val);
			} else if (sym == Token::Kind::pi) {
				scan();
				return new Expr(Expr::Kind::number, PI);
			} else if (sym == Token::Kind::identifier) {
				scan();
				return new Expr(Expr::Kind::id, 0, nullptr, nullptr, t.str);
			} else if (sym == Token::Kind::lpar) {
				scan();
				x = Exp();
				check(Token::Kind::rpar);
				return x;
			} else if (unaryops.find(sym) != unaryops.end()) {
				auto op = sym;
				scan();
				check(Token::Kind::lpar);
				x = Exp();
				check(Token::Kind::rpar);
				if (x->kind == Expr::Kind::number) {
					if (op == Token::Kind::sin) {
						x->num = std::sin(x->num);
					} else if (op == Token::Kind::cos) {
						x->num = std::cos(x->num);
					} else if (op == Token::Kind::tan) {
						x->num = std::tan(x->num);
					} else if (op == Token::Kind::exp) {
						x->num = std::exp(x->num);
					} else if (op == Token::Kind::ln) {
						x->num = std::log(x->num);
					} else if (op == Token::Kind::sqrt) {
						x->num = std::sqrt(x->num);
					}
					return x;
				} else {
					if (op == Token::Kind::sin) {
						return new Expr(Expr::Kind::sin, 0, x);
					} else if (op == Token::Kind::cos) {
						return new Expr(Expr::Kind::cos, 0, x);
					} else if (op == Token::Kind::tan) {
						return new Expr(Expr::Kind::tan, 0, x);
					} else if (op == Token::Kind::exp) {
						return new Expr(Expr::Kind::exp, 0, x);
					} else if (op == Token::Kind::ln) {
						return new Expr(Expr::Kind::ln, 0, x);
					} else if (op == Token::Kind::sqrt) {
						return new Expr(Expr::Kind::sqrt, 0, x);
					}
				}
			} else {
				std::cerr << "Invalid Expression" << std::endl;
			}

			return nullptr;
		}
		Expr* Factor() {
			Expr* x;
			Expr* y;
			x = Exponentiation();
			while (sym == Token::Kind::power) {
				scan();
				y = Exponentiation();
				if ( x->kind == Expr::Kind::number && y->kind == Expr::Kind::number) {
					x->num = std::pow(x->num, y->num);
					delete y;
				} else {
					x = new Expr(Expr::Kind::power, 0, x, y);
				}
			}
			return x;
		}
		Expr* Term() {
			Expr* x = Factor();
			Expr* y;

			while(sym == Token::Kind::times || sym == Token::Kind::div) {
				auto op = sym;
				scan();
				y = Factor();
				if (op == Token::Kind::times) {
					if (x->kind == Expr::Kind::number && y->kind == Expr::Kind::number) {
						x->num = x->num * y->num;
						delete y;
					} else {
						x = new Expr(Expr::Kind::times, 0, x, y);
					}
				} else {
					if (x->kind == Expr::Kind::number && y->kind == Expr::Kind::number) {
						x->num = x->num / y->num;
						delete y;
					} else {
						x = new Expr(Expr::Kind::div, 0, x, y);
					}
				}
			}
			return x;
		}
		Expr* Exp() {
			Expr* x;
			Expr* y;
			if (sym == Token::Kind::minus) {
				scan();
				x = Term();
				if (x->kind == Expr::Kind::number)
					x->num= -x->num;
				else
					x = new Expr(Expr::Kind::sign, 0, x);
			} else {
				x = Term();
			}

			while(sym == Token::Kind::plus || sym == Token::Kind::minus) {
				auto op = sym;
				scan();
				y = Term();
				if (op == Token::Kind::plus) {
					if (x->kind == Expr::Kind::number && y->kind == Expr::Kind::number)
						x->num += y->num;
					else
						x = new Expr(Expr::Kind::plus, 0, x, y);
				} else {
					if (x->kind == Expr::Kind::number && y->kind == Expr::Kind::number)
						x->num -= y->num;
					else
						x = new Expr(Expr::Kind::minus, 0, x, y);
				}
			}
			return x;
		}

		static Expr *RewriteExpr(Expr *expr, std::map<std::string, Expr *>& exprMap) {
			if (expr == nullptr) return nullptr;
			Expr *op1 = RewriteExpr(expr->op1, exprMap);
			Expr *op2 = RewriteExpr(expr->op2, exprMap);

			if (expr->kind == Expr::Kind::number) {
				return new Expr(expr->kind, expr->num, op1, op2, expr->id);
			} else if (expr->kind == Expr::Kind::plus) {
				if (op1->kind == Expr::Kind::number && op2->kind == Expr::Kind::number) {
					op1->num = op1->num + op2->num;
					delete op2;
					return op1;
				}
			} else if (expr->kind == Expr::Kind::minus) {
				if (op1->kind == Expr::Kind::number && op2->kind == Expr::Kind::number) {
					op1->num = op1->num - op2->num;
					delete op2;
					return op1;
				}
			} else if (expr->kind == Expr::Kind::sign) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = -op1->num;
					return op1;
				}
			} else if (expr->kind == Expr::Kind::times) {
				if (op1->kind == Expr::Kind::number && op2->kind == Expr::Kind::number) {
					op1->num = op1->num * op2->num;
					delete op2;
					return op1;
				}
			} else if (expr->kind == Expr::Kind::div) {
				if (op1->kind == Expr::Kind::number && op2->kind == Expr::Kind::number) {
					op1->num = op1->num / op2->num;
					delete op2;
					return op1;
				}
			} else if (expr->kind == Expr::Kind::power) {
				if (op1->kind == Expr::Kind::number && op2->kind == Expr::Kind::number) {
					op1->num = std::pow(op1->num, op2->num);
					delete op2;
					return op1;
				}
			} else if (expr->kind == Expr::Kind::sin) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = std::sin(op1->num);
					return op1;
				}
			} else if (expr->kind == Expr::Kind::cos) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = std::cos(op1->num);
					return op1;
				}
			} else if (expr->kind == Expr::Kind::tan) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = std::tan(op1->num);
					return op1;
				}
			} else if (expr->kind == Expr::Kind::exp) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = std::exp(op1->num);
					return op1;
				}
			} else if (expr->kind == Expr::Kind::ln) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = std::log(op1->num);
					return op1;
				}
			} else if (expr->kind == Expr::Kind::sqrt) {
				if (op1->kind == Expr::Kind::number) {
					op1->num = std::sqrt(op1->num);
					return op1;
				}
			} else if (expr->kind == Expr::Kind::id) {
				return new Expr(*exprMap[expr->id]);
			}

			return new Expr(expr->kind, expr->num, op1, op2, expr->id);
		}
		static void error [[ noreturn ]] (const std::string& msg, int code) {
			std::cerr << msg << std::endl;
			exit(code);
		}
	public:
		Token la, t;
		Token::Kind sym = Token::Kind::none;
		Scanner *scanner;
		registerMap& qregs;
		registerMap& cregs;
		unsigned short nqubits = 0;

		explicit Parser(std::istream& is, registerMap& qregs, registerMap& cregs) :in(is), qregs(qregs), cregs(cregs) {
			scanner = new Scanner(in);
		}

		virtual ~Parser() {
			delete scanner;

			for (auto& cGate:compoundGates)
				for (auto& gate: cGate.second.gates)
					delete gate;
		}

		void scan() {
			t = la;
			la = scanner->next();
			sym = la.kind;
		}
		void check(Token::Kind expected) {
			if (sym == expected)
				scan();
			else
				error("ERROR while parsing QASM file: expected '" + qasm::KindNames[expected] + "' but found '" + qasm::KindNames[sym] + "' in line " + std::to_string(la.line) + ", column " + std::to_string(la.col), 1);
		}

		std::pair<unsigned short , unsigned short> ArgumentQreg() {
			check(Token::Kind::identifier);
			std::string s = t.str;
			if (qregs.find(s) == qregs.end())
				error("Argument is not a qreg: " + s, 1);

			if (sym == Token::Kind::lbrack) {
				scan();
				check(Token::Kind::nninteger);
				unsigned short offset = t.val;
				check(Token::Kind::rbrack);
				return std::make_pair(qregs[s].first + offset, 1);
			}
			return std::make_pair(qregs[s].first, qregs[s].second);
		}
		std::pair<unsigned short, unsigned short> ArgumentCreg() {
			check(Token::Kind::identifier);
			std::string s = t.str;
			if (cregs.find(s) == cregs.end())
				error("Argument is not a creg: " + s, 1);

			if (sym == Token::Kind::lbrack) {
				scan();
				check(Token::Kind::nninteger);
				unsigned short offset = t.val;
				check(Token::Kind::rbrack);
				return std::make_pair(cregs[s].first+offset, 1);
			}

			return std::make_pair(cregs[s].first, cregs[s].second);
		}

		void ExpList(std::vector<Expr*>& expressions) {
			expressions.emplace_back(Exp());
			while(sym == Token::Kind::comma) {
				scan();
				expressions.emplace_back(Exp());
			}
		}
		void ArgList(std::vector<std::pair<unsigned short, unsigned short>>& arguments) {
			arguments.emplace_back(ArgumentQreg());
			while(sym == Token::Kind::comma) {
				scan();
				arguments.emplace_back(ArgumentQreg());
			}
		}
		void IdList(std::vector<std::string>& identifiers) {
			check(Token::Kind::identifier);
			identifiers.emplace_back(t.str);
			while (sym == Token::Kind::comma) {
				scan();
				check(Token::Kind::identifier);
				identifiers.emplace_back(t.str);
			}
		}

		std::unique_ptr<qc::Operation> Gate() {
			if (sym == Token::Kind::ugate) {
				scan();
				check(Token::Kind::lpar);
				std::unique_ptr<Expr> theta(Exp());
				check(Token::Kind::comma);
				std::unique_ptr<Expr> phi(Exp());
				check(Token::Kind::comma);
				std::unique_ptr<Expr> lambda(Exp());
				check(Token::Kind::rpar);
				auto target = ArgumentQreg();
				check(Token::Kind::semicolon);

				if (target.second == 1) {
					return std::make_unique<qc::StandardOperation>(nqubits, target.first, qc::U3, lambda->num, phi->num, theta->num);
				}

				// TODO: multiple targets could be useful here
				auto gate = qc::CompoundOperation(nqubits);
				for (unsigned short i = 0; i < target.second; ++i) {
					gate.emplace_back<qc::StandardOperation>(nqubits, target.first + i, qc::U3, lambda->num, phi->num, theta->num);
				}
				return std::make_unique<qc::CompoundOperation>(gate);
			} else if (sym == Token::Kind::cxgate) {
				scan();
				auto control = ArgumentQreg();
				check(Token::Kind::comma);
				auto target = ArgumentQreg();
				check(Token::Kind::semicolon);

				// return corresponding operation
				if (control.second == 1 && target.second == 1) {
					return std::make_unique<qc::StandardOperation>(nqubits, control.first, target.first, qc::X);
				} else {
					auto gate = qc::CompoundOperation(nqubits);
					if (control.second == target.second) {
						for (int i = 0; i < target.second; ++i)
							gate.emplace_back<qc::StandardOperation>(nqubits, control.first + i, target.first+i, qc::X);
					} else if (control.second == 1) {
						// TODO: multiple targets could be useful here
						for (int i = 0; i < target.second; ++i)
							gate.emplace_back<qc::StandardOperation>(nqubits, control.first, target.first + i, qc::X);
					} else if (target.second == 1) {
						for (int i = 0; i < control.second; ++i)
							gate.emplace_back<qc::StandardOperation>(nqubits, control.first + i, target.first, qc::X);
					} else {
						error("Register size does not match for CX gate!", 1);
					}
					return std::make_unique<qc::CompoundOperation>(gate);
				}

			} else if (sym == Token::Kind::identifier) {
				scan();
				auto gateIt = compoundGates.find(t.str);
				if (gateIt != compoundGates.end()) {
					auto gateName = t.str;
					std::vector<Expr*> parameters;
					std::vector<std::pair<unsigned short , unsigned short>> arguments;
					if (sym == Token::Kind::lpar) {
						scan();
						if (sym != Token::Kind::rpar)
							ExpList(parameters);
						check(Token::Kind::rpar);
					}
					ArgList(arguments);
					check(Token::Kind::semicolon);

					// return corresponding operation
					registerMap argMap;
					std::map<std::string, Expr*> paramMap;
					unsigned short size = 1;
					for (size_t i = 0; i < arguments.size(); ++i) {
						argMap[gateIt->second.argumentNames[i]] = arguments[i];
						if (arguments[i].second > 1 && size != 1 && arguments[i].second != size)
							error("Register sizes do not match!", 1);

						if (arguments[i].second > 1)
							size = arguments[i].second;
					}

					for (size_t i = 0; i < parameters.size(); ++i)
						paramMap[gateIt->second.parameterNames[i]] = parameters[i];

					// check if single controlled gate
					if (gateName.front() == 'c' && size == 1) {
						auto cGateName = gateName;
						std::map<std::string, CompoundGate>::iterator cGateIt;
						int cCount = 0;
						while (cGateName.front() == 'c') {
							cGateName = cGateName.substr(1);
							cGateIt = compoundGates.find(cGateName);
							if (cGateIt != compoundGates.end())
								cCount++;
						}
						// TODO: this could be enhanced for the case that any argument is a register
						if (cGateIt->second.gates.size() == 1) {
							std::vector<short> controls;
							for (int j = 0; j < cCount; ++j)
								controls.push_back(argMap[gateIt->second.argumentNames[j]].first);

							// special treatment for Toffoli
							if (cGateName == "x" && cCount > 1) {
								return std::make_unique<qc::StandardOperation>(nqubits, controls, argMap[gateIt->second.argumentNames.back()].first);
							}

							auto cGate = cGateIt->second.gates.front();
							for (size_t j = 0; j < parameters.size(); ++j)
								paramMap[cGateIt->second.parameterNames[j]] = parameters[j];

							if (auto cu = dynamic_cast<Ugate *>(cGate)) {
								std::unique_ptr<Expr> theta(RewriteExpr(cu->theta, paramMap));
								std::unique_ptr<Expr> phi(RewriteExpr(cu->phi, paramMap));
								std::unique_ptr<Expr> lambda(RewriteExpr(cu->lambda, paramMap));

								return std::make_unique<qc::StandardOperation>(nqubits, controls, argMap[gateIt->second.argumentNames.back()].first, qc::U3, lambda->num, phi->num, theta->num);
							} else {
								error("Cast to u-Gate not possible for controlled operation.", 1);
							}
						}
					}

					// identifier specifies just a single operation (U3 or CX)
					if (gateIt->second.gates.size() == 1) {
						auto gate = gateIt->second.gates.front();
						if (auto u = dynamic_cast<Ugate *>(gate)) {
							std::unique_ptr<Expr> theta(RewriteExpr(u->theta, paramMap));
							std::unique_ptr<Expr> phi(RewriteExpr(u->phi, paramMap));
							std::unique_ptr<Expr> lambda(RewriteExpr(u->lambda, paramMap));

							if (argMap[u->target].second == 1) {
								return std::make_unique<qc::StandardOperation>(nqubits, argMap[u->target].first, qc::U3, lambda->num, phi->num, theta->num);
							}
						} else if (auto cx = dynamic_cast<CXgate *>(gate)) {
							if (argMap[cx->control].second == 1 && argMap[cx->target].second == 1) {
								return std::make_unique<qc::StandardOperation>(nqubits, argMap[cx->control].first, argMap[cx->target].first, qc::X);
							}
						}
					}

					qc::CompoundOperation op(nqubits);
					for (auto& gate: gateIt->second.gates) {
						if (auto u = dynamic_cast<Ugate*>(gate)) {
							std::unique_ptr<Expr> theta(RewriteExpr(u->theta, paramMap));
							std::unique_ptr<Expr> phi(RewriteExpr(u->phi, paramMap));
							std::unique_ptr<Expr> lambda(RewriteExpr(u->lambda, paramMap));

							if (argMap[u->target].second == 1) {
								op.emplace_back<qc::StandardOperation>(nqubits, argMap[u->target].first, qc::U3, lambda->num, phi->num, theta->num);
							} else {
								// TODO: multiple targets could be useful here
								for (unsigned short j = 0; j < argMap[u->target].second; ++j) {
									op.emplace_back<qc::StandardOperation>(nqubits, argMap[u->target].first + j, qc::U3, lambda->num, phi->num, theta->num);
								}
							}
						} else if (auto cx = dynamic_cast<CXgate*>(gate)) {
							if (argMap[cx->control].second == 1 && argMap[cx->target].second == 1) {
								op.emplace_back<qc::StandardOperation>(nqubits, argMap[cx->control].first, argMap[cx->target].first, qc::X);
							} else if (argMap[cx->control].second == argMap[cx->target].second) {
								for (int j = 0; j < argMap[cx->target].second; ++j)
									op.emplace_back<qc::StandardOperation>(nqubits, argMap[cx->control].first + j, argMap[cx->target].first + j, qc::X);
							} else if (argMap[cx->control].second == 1) {
								// TODO: multiple targets could be useful here
								for (int k = 0; k < argMap[cx->target].second; ++k)
									op.emplace_back<qc::StandardOperation>(nqubits,argMap[cx->control].first, argMap[cx->target].first + k, qc::X);
							} else if (argMap[cx->target].second == 1) {
								for (int l = 0; l < argMap[cx->control].second; ++l)
									op.emplace_back<qc::StandardOperation>(nqubits, argMap[cx->control].first + l, argMap[cx->target].first, qc::X);
							} else {
								error("Register size does not match for CX gate!",1);
							}
						}
					}
					return std::make_unique<qc::CompoundOperation>(op);
				} else {
					error("Undefined gate " + t.str, 1);
				}
			} else {
				error("Symbol " + qasm::KindNames[sym] + " not expected in Gate() routine!", 1);
			}
		}

		void OpaqueGateDecl() {
			check(Token::Kind::opaque);
			check(Token::Kind::identifier);

			CompoundGate gate;
			auto gateName = t.str;
			if (sym == Token::Kind::lpar) {
				scan();
				if (sym != Token::Kind::rpar) {
					IdList(gate.argumentNames);
				}
				check(Token::Kind::rpar);
			}
			IdList(gate.argumentNames);
			compoundGates[gateName] = gate;
			check(Token::Kind::semicolon);
		}
		void GateDecl() {
			check(Token::Kind::gate);
			check(Token::Kind::identifier);

			CompoundGate gate;
			std::string gateName = t.str;
			if (sym == Token::Kind::lpar) {
				scan();
				if (sym != Token::Kind::rpar) {
					IdList(gate.parameterNames);
				}
				check(Token::Kind::rpar);
			}
			IdList(gate.argumentNames);
			check(Token::Kind::lbrace);

			while (sym != Token::Kind::rbrace) {
				if (sym == Token::Kind::ugate) {
					scan();
					check(Token::Kind::lpar);
					Expr *theta = Exp();
					check(Token::Kind::comma);
					Expr *phi = Exp();
					check(Token::Kind::comma);
					Expr *lambda = Exp();
					check(Token::Kind::rpar);
					check(Token::Kind::identifier);

					gate.gates.push_back(new Ugate(theta, phi, lambda, t.str));
					check(Token::Kind::semicolon);
				} else if (sym == Token::Kind::cxgate) {
					scan();
					check(Token::Kind::identifier);
					std::string control = t.str;
					check(Token::Kind::comma);
					check(Token::Kind::identifier);
					gate.gates.push_back(new CXgate(control, t.str));
					check(Token::Kind::semicolon);

				} else if (sym == Token::Kind::identifier) {
					scan();
					std::string name = t.str;

					std::vector<Expr *> parameters;
					std::vector<std::string> arguments;
					if (sym == Token::Kind::lpar) {
						scan();
						if (sym != Token::Kind::rpar) {
							ExpList(parameters);
						}
						check(Token::Kind::rpar);
					}
					IdList(arguments);
					check(Token::Kind::semicolon);

					CompoundGate g = compoundGates[name];
					std::map<std::string, std::string> argsMap;
					for (unsigned long i = 0; i < arguments.size(); i++) {
						argsMap[g.argumentNames[i]] = arguments[i];
					}

					std::map<std::string, Expr *> paramsMap;
					for (unsigned long i = 0; i < parameters.size(); i++) {
						paramsMap[g.parameterNames[i]] = parameters[i];
					}

					for (auto & it : g.gates) {
						if (auto u = dynamic_cast<Ugate *>(it)) {
							gate.gates.push_back(new Ugate(RewriteExpr(u->theta, paramsMap), RewriteExpr(u->phi, paramsMap), RewriteExpr(u->lambda, paramsMap), argsMap[u->target]));
						} else if (auto cx = dynamic_cast<CXgate *>(it)) {
							gate.gates.push_back(new CXgate(argsMap[cx->control], argsMap[cx->target]));
						} else {
							error("Unexpected gate in GateDecl!", 1);
						}
					}

					for (auto & parameter : parameters) {
						delete parameter;
					}
				} else if (sym == Token::Kind::barrier) {
					scan();
					std::vector<std::string> arguments;
					IdList(arguments);
					check(Token::Kind::semicolon);
					//Nothing to do here for the simulator
				} else {
					error("Error in gate declaration!", 1);
				}
			}
			compoundGates[gateName] = gate;
			check(Token::Kind::rbrace);
		}

		std::unique_ptr<qc::Operation> Qop() {
			if (sym == Token::Kind::ugate || sym == Token::Kind::cxgate || sym == Token::Kind::identifier)
				return Gate();
			else if (sym == Token::Kind::measure) {
				scan();
				auto qreg = ArgumentQreg();
				check(Token::Kind::minus);
				check(Token::Kind::gt);
				auto creg = ArgumentCreg();
				check(Token::Kind::semicolon);

				if (qreg.second == creg.second) {
					std::vector<unsigned short> qubits, classics;
					for (int i = 0; i < qreg.second; ++i) {
						qubits.emplace_back(qreg.first + i);
						classics.emplace_back(creg.first + i);
					}
					return std::make_unique<qc::NonUnitaryOperation>(nqubits, qubits, classics);
				} else {
					error("Mismatch of qreg and creg size in measurement", 1);
				}
			} else if (sym == Token::Kind::reset) {
				scan();
				auto qreg = ArgumentQreg();
				check(Token::Kind::semicolon);

				std::vector<unsigned short> qubits;
				for (int i = 0; i < qreg.second; ++i) {
					qubits.emplace_back(qreg.first + i);
				}
				return std::make_unique<qc::NonUnitaryOperation>(nqubits, qubits);
			} else {
				error("No valid Qop: " + t.str, 1);
			}
		}

	};

}
#endif //INTERMEDIATEREPRESENTATION_PARSER_H
