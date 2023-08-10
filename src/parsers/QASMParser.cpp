#include "QuantumComputation.hpp"

void qc::QuantumComputation::importOpenQASM(std::istream& is) {
  using namespace qasm;
  // initialize parser
  Parser p(is, qregs, cregs);

  p.scan();
  while (p.sym == Token::Kind::Comment) {
    p.scan();
    p.handleComment();
  }
  if (p.sym == Token::Kind::Openqasm) {
    p.scan();
    p.check(Token::Kind::Real);
    p.check(Token::Kind::Semicolon);
  }

  while (p.sym != Token::Kind::Eof) {
    if (p.sym == Token::Kind::Qreg) {
      // quantum register definition
      p.scan();
      p.check(Token::Kind::Identifier);
      const std::string s = p.t.str;
      p.check(Token::Kind::Lbrack);
      p.check(Token::Kind::Nninteger);
      const auto n = static_cast<std::size_t>(p.t.val);
      p.check(Token::Kind::Rbrack);
      p.check(Token::Kind::Semicolon);
      addQubitRegister(n, s);
      p.nqubits = nqubits;
    } else if (p.sym == Token::Kind::Creg) {
      // classical register definition
      p.scan();
      p.check(Token::Kind::Identifier);
      const std::string s = p.t.str;
      p.check(Token::Kind::Lbrack);
      p.check(Token::Kind::Nninteger);
      const auto n = static_cast<std::size_t>(p.t.val);
      p.check(Token::Kind::Rbrack);
      p.check(Token::Kind::Semicolon);
      addClassicalRegister(n, s);
      p.nclassics = nclassics;
    } else if (p.sym == Token::Kind::Identifier ||
               p.sym == Token::Kind::Measure || p.sym == Token::Kind::Reset ||
               p.sym == Token::Kind::McxGray ||
               p.sym == Token::Kind::McxRecursive ||
               p.sym == Token::Kind::McxVchain ||
               p.sym == Token::Kind::Mcphase) {
      // gate application
      ops.emplace_back(p.qop());
    } else if (p.sym == Token::Kind::Gate) {
      // gate definition
      p.gateDecl();
    } else if (p.sym == Token::Kind::Include) {
      // include statement
      p.scan();
      p.check(Token::Kind::String);
      p.scanner->addFileInput(p.t.str);
      p.check(Token::Kind::Semicolon);
    } else if (p.sym == Token::Kind::Barrier) {
      // barrier statement
      p.scan();
      std::vector<qc::QuantumRegister> args;
      p.argList(args);
      p.check(Token::Kind::Semicolon);

      std::vector<qc::Qubit> qubits{};
      for (auto& arg : args) {
        for (std::size_t q = 0; q < arg.second; ++q) {
          qubits.emplace_back(static_cast<Qubit>(arg.first + q));
        }
      }
      barrier(qubits);
    } else if (p.sym == Token::Kind::Opaque) {
      // opaque gate definition
      p.opaqueGateDecl();
    } else if (p.sym == Token::Kind::If) {
      // classically-controlled operation
      p.scan();
      p.check(Token::Kind::Lpar);
      p.check(Token::Kind::Identifier);
      const std::string creg = p.t.str;
      p.check(Token::Kind::Eq);
      p.check(Token::Kind::Nninteger);
      const auto n = static_cast<std::size_t>(p.t.val);
      p.check(Token::Kind::Rpar);

      auto it = p.cregs.find(creg);
      if (it == p.cregs.end()) {
        p.error("Error in if statement: " + creg + " is not a creg!");
      } else {
        emplace_back<ClassicControlledOperation>(p.qop(), it->second, n);
      }
    } else if (p.sym == Token::Kind::Snapshot) {
      // snapshot statement
      p.scan();
      p.check(Token::Kind::Lpar);
      p.check(Token::Kind::Nninteger);
      p.check(Token::Kind::Rpar);
      std::vector<qc::QuantumRegister> arguments{};
      p.argList(arguments);
      p.check(Token::Kind::Semicolon);
      // Snapshots have no meaning for MQT Core and are ignored.
    } else if (p.sym == Token::Kind::Probabilities) {
      p.scan();
      p.check(Token::Kind::Semicolon);
      // Show probabilities has no meaning for MQT Core and is ignored.
    } else if (p.sym == Token::Kind::Comment) {
      // comment
      p.scan();
      p.handleComment();
    } else {
      p.error("Unexpected statement: started with " + KIND_NAMES.at(p.sym) +
              "!");
    }
  }

  // if any I/O information was gathered during parsing, transfer it to the
  // QuantumComputation
  if (!p.initialLayout.empty()) {
    initialLayout = std::move(p.initialLayout);
  }
  if (!p.outputPermutation.empty()) {
    outputPermutation = std::move(p.outputPermutation);
  }
}
