#include "parsers/qasm_parser/Parser.hpp"

#include "Definitions.hpp"
#include "operations/Control.hpp"

#include <memory>
#include <vector>

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
    auto x = expr();
    check(Token::Kind::Rpar);
    return x;
  }
  if (unaryops.find(sym) != unaryops.end()) {
    const auto op = sym;
    scan();
    check(Token::Kind::Lpar);
    auto x = expr();
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

std::shared_ptr<Parser::Expr> Parser::expr() {
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

std::shared_ptr<Parser::Expr>
Parser::rewriteExpr(const std::shared_ptr<Expr>& expr,
                    std::map<std::string, std::shared_ptr<Expr>>& exprMap) {
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
  if (const auto initial = checkForInitialLayout(t.str); !initial.empty()) {
    if (!initialLayout.empty()) {
      error("Multiple initial layout specifications found.");
    } else {
      initialLayout = initial;
    }
  }
  if (const auto output = checkForOutputPermutation(t.str); !output.empty()) {
    if (!outputPermutation.empty()) {
      error("Multiple output permutation specifications found.");
    } else {
      outputPermutation = output;
    }
  }
}

qc::Permutation Parser::checkForInitialLayout(std::string comment) {
  static const auto INITIAL_LAYOUT_REGEX = std::regex("i (\\d+ )*(\\d+)");
  static const auto QUBIT_REGEX = std::regex("\\d+");
  qc::Permutation initial{};
  if (std::regex_search(comment, INITIAL_LAYOUT_REGEX)) {
    qc::Qubit logicalQubit = 0;
    for (std::smatch m; std::regex_search(comment, m, QUBIT_REGEX);
         comment = m.suffix()) {
      auto physicalQubit = static_cast<qc::Qubit>(std::stoul(m.str()));
      initial.insert({physicalQubit, logicalQubit});
      ++logicalQubit;
    }
  }
  return initial;
}

qc::Permutation Parser::checkForOutputPermutation(std::string comment) {
  static const auto OUTPUT_PERMUTATION_REGEX = std::regex("o (\\d+ )*(\\d+)");
  static const auto QUBIT_REGEX = std::regex("\\d+");
  qc::Permutation output{};
  if (std::regex_search(comment, OUTPUT_PERMUTATION_REGEX)) {
    qc::Qubit logicalQubit = 0;
    for (std::smatch m; std::regex_search(comment, m, QUBIT_REGEX);
         comment = m.suffix()) {
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
  t = la;
  la = scanner->next();
  sym = la.kind;
}

void Parser::check(const Token::Kind expected) {
  while (sym == Token::Kind::Comment) {
    scan();
    handleComment();
  }

  if (sym == expected) {
    scan();
  } else {
    error("Expected '" + qasm::KIND_NAMES.at(expected) + "' but found '" +
          qasm::KIND_NAMES.at(sym) + "' in line " + std::to_string(la.line) +
          ", column " + std::to_string(la.col));
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
    const auto offset = static_cast<std::size_t>(t.val);
    check(Token::Kind::Rbrack);
    return std::make_pair(static_cast<qc::Qubit>(qregs[s].first + offset), 1U);
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
    const auto offset = static_cast<std::size_t>(t.val);
    check(Token::Kind::Rbrack);
    return std::make_pair(cregs[s].first + offset, 1);
  }

  return std::make_pair(cregs[s].first, cregs[s].second);
}

void Parser::expList(std::vector<std::shared_ptr<Parser::Expr>>& expressions) {
  expressions.emplace_back(expr());
  while (sym == Token::Kind::Comma) {
    scan();
    expressions.emplace_back(expr());
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
  if (sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive ||
      sym == Token::Kind::McxVchain) {
    const auto type = sym;
    scan();
    std::vector<qc::QuantumRegister> registers{};
    registers.emplace_back(argumentQreg());
    while (sym != Token::Kind::Semicolon) {
      check(Token::Kind::Comma);
      registers.emplace_back(argumentQreg());
    }
    scan();

    // drop ancillaries since our library can natively work with MCTs
    if (type == Token::Kind::McxVchain) {
      // n controls, 1 target, n-2 ancillaries = 2n-1 qubits
      const auto ancillaries = (registers.size() + 1) / 2 - 2;
      for (std::size_t i = 0; i < ancillaries; ++i) {
        registers.pop_back();
      }
    } else if (type == Token::Kind::McxRecursive) {
      // 1 ancillary if more than 4 controls
      if (registers.size() > 5) {
        registers.pop_back();
      }
    }
    const auto target = registers.back();
    registers.pop_back();

    auto info = GATE_MAP.at("x");
    info.nControls = registers.size();
    return knownGate(info, {}, registers, {target});
  }
  if (sym == Token::Kind::Mcphase) {
    scan();
    check(Token::Kind::Lpar);
    const auto lambda = expr();
    check(Token::Kind::Rpar);

    std::vector<qc::QuantumRegister> registers{};
    registers.emplace_back(argumentQreg());
    while (sym != Token::Kind::Semicolon) {
      check(Token::Kind::Comma);
      registers.emplace_back(argumentQreg());
    }
    scan();

    const auto target = registers.back();
    registers.pop_back();
    auto info = GATE_MAP.at("p");
    info.nControls = registers.size();
    return knownGate(info, {lambda->num}, registers, {target});
  }
  if (sym == Token::Kind::Identifier) {
    scan();
    const auto gateName = t.str;

    GateInfo info{};
    const auto found = gateInfo(gateName, info);
    if (found) {
      return knownGate(info);
    }

    // at this point, the gate has to be user-defined
    if (compoundGates.find(gateName) == compoundGates.end()) {
      error("Unknown gate " + gateName);
    }
    auto& compoundGate = compoundGates.find(gateName)->second;

    std::vector<std::shared_ptr<Parser::Expr>> parameters;
    if (sym == Token::Kind::Lpar) {
      scan();
      if (sym != Token::Kind::Rpar) {
        expList(parameters);
      }
      check(Token::Kind::Rpar);
    }
    std::vector<qc::QuantumRegister> arguments;
    argList(arguments);
    check(Token::Kind::Semicolon);

    if (compoundGate.argumentNames.size() != arguments.size()) {
      std::ostringstream oss{};
      if (compoundGate.argumentNames.size() < arguments.size()) {
        oss << "Too many arguments for ";
      } else {
        oss << "Too few arguments for ";
      }
      oss << gateName << " gate! Expected " << compoundGate.argumentNames.size()
          << ", but got " << arguments.size();
      error(oss.str());
    }

    qc::QuantumRegisterMap argMap;
    std::size_t size = 1;
    for (size_t i = 0; i < arguments.size(); ++i) {
      argMap[compoundGate.argumentNames[i]] = arguments[i];
      if (arguments[i].second > 1 && size != 1 && arguments[i].second != size) {
        error("Register sizes do not match!");
      }

      if (arguments[i].second > 1) {
        size = arguments[i].second;
      }
    }

    std::map<std::string, std::shared_ptr<Parser::Expr>> paramMap;
    for (size_t i = 0; i < parameters.size(); ++i) {
      paramMap[compoundGate.parameterNames[i]] = parameters[i];
    }

    qc::CompoundOperation op(nqubits);
    for (auto& g : compoundGate.gates) {
      const auto* gate = dynamic_cast<StandardGate*>(g.get());
      if (gate == nullptr) {
        error("Unsupported gate type in compound gate definition");
      }
      std::vector<qc::fp> rewrittenParameters;
      rewrittenParameters.reserve(gate->parameters.size());
      for (const auto& p : gate->parameters) {
        rewrittenParameters.emplace_back(rewriteExpr(p, paramMap)->num);
      }
      std::vector<qc::QuantumRegister> rewrittenControls;
      rewrittenControls.reserve(gate->controls.size());
      for (const auto& control : gate->controls) {
        rewrittenControls.emplace_back(argMap.at(control));
      }
      std::vector<qc::QuantumRegister> rewrittenTargets;
      rewrittenTargets.reserve(gate->targets.size());
      for (const auto& target : gate->targets) {
        rewrittenTargets.emplace_back(argMap.at(target));
      }

      if (compoundGate.gates.size() == 1) {
        return knownGate(gate->info, rewrittenParameters, rewrittenControls,
                         rewrittenTargets);
      }

      op.getOps().emplace_back(knownGate(gate->info, rewrittenParameters,
                                         rewrittenControls, rewrittenTargets));
    }
    return std::make_unique<qc::CompoundOperation>(std::move(op));
  }
  error("Symbol " + qasm::KIND_NAMES.at(sym) +
        " not expected in Gate() routine!");
}

void Parser::opaqueGateDecl() {
  check(Token::Kind::Opaque);
  check(Token::Kind::Identifier);

  CompoundGate gate;
  auto gateName = t.str;
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
  if (sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive ||
      sym == Token::Kind::McxVchain || sym == Token::Kind::Mcphase) {
    while (sym != Token::Kind::Rbrace) {
      scan();
    }
    check(Token::Kind::Rbrace);
    return;
  }
  check(Token::Kind::Identifier);
  const std::string gateName = t.str;
  GateInfo info{};
  const auto found = gateInfo(gateName, info);
  if (found) {
    // gate is already supported natively -> skip declaration
    while (sym != Token::Kind::Rbrace) {
      scan();
    }
    check(Token::Kind::Rbrace);
    return;
  }

  // at this point, this is a new gate definition
  CompoundGate gate;
  if (sym == Token::Kind::Lpar) {
    scan();
    if (sym != Token::Kind::Rpar) {
      idList(gate.parameterNames);
    }
    check(Token::Kind::Rpar);
  }
  idList(gate.argumentNames);
  check(Token::Kind::Lbrace);
  while (sym != Token::Kind::Rbrace) {
    if (sym == Token::Kind::McxGray || sym == Token::Kind::McxRecursive ||
        sym == Token::Kind::McxVchain) {
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

      const auto target = arguments.back();
      arguments.pop_back();
      auto mcx = StandardGate{};
      mcx.info = GATE_MAP.at("x");
      mcx.info.nControls = arguments.size();
      mcx.controls = std::move(arguments);
      mcx.targets.emplace_back(target);
      gate.gates.push_back(std::make_shared<StandardGate>(std::move(mcx)));
    } else if (sym == Token::Kind::Mcphase) {
      scan();
      check(Token::Kind::Lpar);
      const auto lambda = expr();
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
      const auto target = arguments.back();
      arguments.pop_back();
      auto mcp = StandardGate{};
      mcp.info = GATE_MAP.at("p");
      mcp.info.nControls = arguments.size();
      mcp.controls = std::move(arguments);
      mcp.targets.emplace_back(target);
      mcp.parameters.emplace_back(lambda);
      gate.gates.push_back(std::make_shared<StandardGate>(std::move(mcp)));
    } else if (sym == Token::Kind::Identifier) {
      scan();
      const std::string name = t.str;

      std::vector<std::shared_ptr<Parser::Expr>> parameters;
      std::vector<std::string> arguments;
      if (sym == Token::Kind::Lpar) {
        scan();
        if (sym != Token::Kind::Rpar) {
          expList(parameters);
        }
        check(Token::Kind::Rpar);
      }
      idList(arguments);
      check(Token::Kind::Semicolon);

      GateInfo gateInf{};
      const auto gateFound = gateInfo(name, gateInf);
      if (gateFound) {
        // gate is already supported natively
        auto sGate = StandardGate{};
        sGate.info = gateInf;
        for (std::size_t i = 0; i < sGate.info.nControls; ++i) {
          sGate.controls.emplace_back(arguments.front());
          arguments.erase(arguments.begin());
        }
        sGate.targets = std::move(arguments);
        sGate.parameters = std::move(parameters);
        gate.gates.emplace_back(
            std::make_shared<StandardGate>(std::move(sGate)));
        continue;
      }

      // gate is not supported natively -> check if a definition for it exists
      if (compoundGates.find(name) == compoundGates.end()) {
        error("Unsupported gate: " + name + " (no definition found)");
      }

      const auto& compoundGate = compoundGates.at(name);
      if (compoundGate.argumentNames.size() != arguments.size()) {
        std::ostringstream oss{};
        if (compoundGate.argumentNames.size() < arguments.size()) {
          oss << "Too many arguments for ";
        } else {
          oss << "Too few arguments for ";
        }
        oss << name << " gate! Expected " << compoundGate.argumentNames.size()
            << ", but got " << arguments.size() << "!";
        error(oss.str());
      }

      std::map<std::string, std::string> argMap;
      for (size_t i = 0; i < arguments.size(); ++i) {
        argMap[compoundGate.argumentNames[i]] = arguments[i];
      }

      std::map<std::string, std::shared_ptr<Parser::Expr>> paramMap;
      for (size_t i = 0; i < parameters.size(); ++i) {
        paramMap[compoundGate.parameterNames[i]] = parameters[i];
      }

      for (const auto& gatePtr : compoundGate.gates) {
        const auto* gateDef = dynamic_cast<StandardGate*>(gatePtr.get());
        if (gateDef == nullptr) {
          error("Unsupported gate type in compound gate definition");
        }
        std::vector<std::shared_ptr<Expr>> rewrittenParameters;
        rewrittenParameters.reserve(gateDef->parameters.size());
        for (const auto& p : gateDef->parameters) {
          rewrittenParameters.emplace_back(rewriteExpr(p, paramMap));
        }
        std::vector<std::string> rewrittenControls;
        rewrittenControls.reserve(gateDef->controls.size());
        for (const auto& control : gateDef->controls) {
          rewrittenControls.emplace_back(argMap.at(control));
        }
        std::vector<std::string> rewrittenTargets;
        rewrittenTargets.reserve(gateDef->targets.size());
        for (const auto& target : gateDef->targets) {
          rewrittenTargets.emplace_back(argMap.at(target));
        }
        auto newGate = std::make_shared<StandardGate>();
        newGate->info = gateDef->info;
        newGate->controls = std::move(rewrittenControls);
        newGate->targets = std::move(rewrittenTargets);
        newGate->parameters = std::move(rewrittenParameters);
        gate.gates.emplace_back(newGate);
      }
    } else if (sym == Token::Kind::Barrier) {
      scan();
      std::vector<std::string> arguments;
      idList(arguments);
      check(Token::Kind::Semicolon);
      // Nothing to do here for the simulator
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
  if (sym == Token::Kind::Identifier || sym == Token::Kind::McxGray ||
      sym == Token::Kind::McxRecursive || sym == Token::Kind::McxVchain ||
      sym == Token::Kind::Mcphase) {
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
      std::vector<qc::Bit> classics{};
      for (std::size_t i = 0; i < qreg.second; ++i) {
        const auto qubit = qreg.first + i;
        const auto clbit = creg.first + i;
        if (qubit >= nqubits) {
          std::stringstream ss{};
          ss << "Qubit " << qubit
             << " cannot be measured since the circuit only contains "
             << nqubits << " qubits";
          error(ss.str());
        }
        if (clbit >= nclassics) {
          std::stringstream ss{};
          ss << "Bit " << clbit
             << " cannot be target of a measurement since the circuit only "
                "contains "
             << nclassics << " classical bits";
          error(ss.str());
        }
        qubits.emplace_back(static_cast<qc::Qubit>(qubit));
        classics.emplace_back(clbit);
      }
      return std::make_unique<qc::NonUnitaryOperation>(nqubits, qubits,
                                                       classics);
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
        ss << "Qubit " << qubit
           << " cannot be reset since the circuit only contains " << nqubits
           << " qubits";
        error(ss.str());
      }
      qubits.emplace_back(static_cast<qc::Qubit>(qubit));
    }
    return std::make_unique<qc::NonUnitaryOperation>(nqubits, qubits);
  }
  error("No valid Qop: " + t.str);
}

void Parser::parseParameters(const GateInfo& info,
                             std::vector<qc::fp>& parameters) {
  // if the gate has parameters, then parse them first
  if (info.nParameters > 0) {
    check(Token::Kind::Lpar);
    for (std::size_t i = 0; i < info.nParameters; ++i) {
      parameters.emplace_back(expr()->num);
      if (i < (info.nParameters - 1)) {
        check(Token::Kind::Comma);
      }
    }
    check(Token::Kind::Rpar);
  }
}

void Parser::parseArguments(const GateInfo& info,
                            std::vector<qc::fp>& parameters,
                            std::vector<qc::QuantumRegister>& controlRegisters,
                            std::vector<qc::QuantumRegister>& targetRegisters) {
  parseParameters(info, parameters);

  // if the gate has controls, collect them next
  if (info.nControls > 0) {
    for (std::size_t i = 0; i < info.nControls; ++i) {
      controlRegisters.emplace_back(argumentQreg());
      if (i < (info.nControls - 1)) {
        check(Token::Kind::Comma);
      }
    }
  }

  // finally, if the gate has targets, collect them
  if (info.nTargets > 0) {
    if (info.nControls > 0U) {
      check(Token::Kind::Comma);
    }
    for (std::size_t i = 0; i < info.nTargets; ++i) {
      targetRegisters.emplace_back(argumentQreg());
      if (i < (info.nTargets - 1)) {
        check(Token::Kind::Comma);
      }
    }
  }
  check(Token::Kind::Semicolon);
}

bool Parser::gateInfo(const std::string& name, GateInfo& info) {
  if (const auto it = GATE_MAP.find(name); it != GATE_MAP.end()) {
    info = it->second;
    return true;
  }
  auto cName = name;
  std::size_t nControls = 0;
  while (cName.front() == 'c') {
    cName = cName.substr(1);
    ++nControls;
  }
  if (const auto it = GATE_MAP.find(cName); it != GATE_MAP.end()) {
    info = it->second;
    info.nControls += nControls;
    return true;
  }
  return false;
}

std::unique_ptr<qc::Operation>
Parser::knownGate(const GateInfo& info, const std::vector<qc::fp>& parameters,
                  const std::vector<qc::QuantumRegister>& controlRegisters,
                  const std::vector<qc::QuantumRegister>& targetRegisters) {
  bool broadcasting = false;
  qc::Controls controls{};
  for (const auto& [startQubit, length] : controlRegisters) {
    if (length != 1) {
      broadcasting = true;
    }
    for (std::size_t i = 0; i < length; ++i) {
      const auto control = qc::Control{static_cast<qc::Qubit>(startQubit + i)};
      if (std::find(controls.begin(), controls.end(), control) !=
          controls.end()) {
        error("Duplicate control qubit in multi-qubit gate.");
      }
      controls.emplace(control);
    }
  }

  qc::Targets targets{};
  for (const auto& [startQubit, length] : targetRegisters) {
    if (length != 1) {
      broadcasting = true;
    }
    for (std::size_t i = 0; i < length; ++i) {
      const auto target = static_cast<qc::Qubit>(startQubit + i);
      if (std::find(targets.begin(), targets.end(), target) != targets.end()) {
        error("Duplicate target qubit in multi-qubit gate.");
      }
      if (std::find(controls.begin(), controls.end(), qc::Control{target}) !=
          controls.end()) {
        error("Duplicate qubit argument in multi-qubit gate.");
      }
      targets.emplace_back(target);
    }
  }

  if (!broadcasting) {
    // standard case: no broadcasting, just a simple operation
    return std::make_unique<qc::StandardOperation>(nqubits, controls, targets,
                                                   info.type, parameters);
  }

  // handle case where there are no controls
  if (info.nControls == 0) {
    // handle single-qubit gates
    if (info.nTargets == 1) {
      const auto& [startQubit, length] = targetRegisters.front();
      auto gate = qc::CompoundOperation(nqubits);
      for (std::size_t i = 0; i < length; ++i) {
        gate.emplace_back<qc::StandardOperation>(
            nqubits, static_cast<qc::Qubit>(startQubit + i), info.type,
            parameters);
      }
      return std::make_unique<qc::CompoundOperation>(std::move(gate));
    }
    error("Broadcasting not supported for multi-target gates.");
  }

  // the single control, single target case is special as we support
  // broadcasting
  if (info.nControls == 1) {
    const auto& [startControl, lengthControl] = controlRegisters.front();
    if (info.nTargets == 1) {
      const auto& [startTarget, lengthTarget] = targetRegisters.front();
      auto gate = qc::CompoundOperation(nqubits);
      if (lengthControl == 1 && lengthTarget > 1) {
        for (std::size_t i = 0; i < lengthTarget; ++i) {
          gate.emplace_back<qc::StandardOperation>(
              nqubits, qc::Control{startControl},
              static_cast<qc::Qubit>(startTarget + i), info.type, parameters);
        }
        return std::make_unique<qc::CompoundOperation>(std::move(gate));
      }
      if (lengthControl > 1 && lengthTarget == 1) {
        for (std::size_t i = 0; i < lengthControl; ++i) {
          gate.emplace_back<qc::StandardOperation>(
              nqubits, qc::Control{static_cast<qc::Qubit>(startControl + i)},
              startTarget, info.type, parameters);
        }
        return std::make_unique<qc::CompoundOperation>(std::move(gate));
      }
      if (lengthControl == lengthTarget) {
        for (std::size_t i = 0; i < lengthControl; ++i) {
          gate.emplace_back<qc::StandardOperation>(
              nqubits, qc::Control{static_cast<qc::Qubit>(startControl + i)},
              static_cast<qc::Qubit>(startTarget + i), info.type, parameters);
        }
        return std::make_unique<qc::CompoundOperation>(std::move(gate));
      }
      error("Ill-formed broadcasting statement.");
    }
  }
  error("Broadcasting not supported for multi-control, multi-target gates.");
}

} // namespace qasm
