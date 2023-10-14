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
class Parser {
  struct GateInfo {
    std::size_t nControls;
    std::size_t nTargets;
    std::size_t nParameters;
    qc::OpType type;
  };

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
    qc::fp num;
    Kind kind;
    std::shared_ptr<Expr> op1 = nullptr;
    std::shared_ptr<Expr> op2 = nullptr;
    std::string id;

    explicit Expr(const Kind k, const qc::fp n = 0.,
                  std::shared_ptr<Expr> operation1 = nullptr,
                  std::shared_ptr<Expr> operation2 = nullptr,
                  std::string identifier = "")
        : num(n), kind(k), op1(std::move(operation1)),
          op2(std::move(operation2)), id(std::move(identifier)) {}
    Expr(const Expr& expr) : num(expr.num), kind(expr.kind), id(expr.id) {
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

      num = expr.num;
      kind = expr.kind;
      id = expr.id;

      op1 = expr.op1;
      op2 = expr.op2;

      return *this;
    }

    virtual ~Expr() = default;
  };

  struct BasisGate {
    virtual ~BasisGate() = default;
  };

  struct StandardGate : public BasisGate {
    GateInfo info;
    std::vector<std::shared_ptr<Expr>> parameters;
    std::vector<std::string> controls;
    std::vector<std::string> targets;
  };

  struct CompoundGate {
    std::vector<std::string> parameterNames;
    std::vector<std::string> argumentNames;
    std::vector<std::shared_ptr<BasisGate>> gates;
  };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::istream& in;
  std::set<Token::Kind> unaryops{Token::Kind::Sin, Token::Kind::Cos,
                                 Token::Kind::Tan, Token::Kind::Exp,
                                 Token::Kind::Ln,  Token::Kind::Sqrt};
  std::map<std::string, CompoundGate> compoundGates;

  std::shared_ptr<Expr> exponentiation();
  std::shared_ptr<Expr> factor();
  std::shared_ptr<Expr> term();
  std::shared_ptr<Expr> expr();

  static std::shared_ptr<Expr>
  rewriteExpr(const std::shared_ptr<Expr>& expr,
              std::map<std::string, std::shared_ptr<Expr>>& exprMap);

public:
  Token la, t;
  Token::Kind sym = Token::Kind::None;
  std::shared_ptr<Scanner> scanner;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  qc::QuantumRegisterMap& qregs;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  qc::ClassicalRegisterMap& cregs;
  std::size_t nqubits = 0;
  std::size_t nclassics = 0;
  qc::Permutation initialLayout{};
  qc::Permutation outputPermutation{};

  static const inline std::unordered_map<std::string, GateInfo> GATE_MAP = {
      {"gphase", {0, 0, 1, qc::GPhase}},
      {"i", {0, 1, 0, qc::I}},
      {"id", {0, 1, 0, qc::I}},
      {"x", {0, 1, 0, qc::X}},
      {"y", {0, 1, 0, qc::Y}},
      {"z", {0, 1, 0, qc::Z}},
      {"h", {0, 1, 0, qc::H}},
      {"s", {0, 1, 0, qc::S}},
      {"sdg", {0, 1, 0, qc::Sdg}},
      {"t", {0, 1, 0, qc::T}},
      {"tdg", {0, 1, 0, qc::Tdg}},
      {"sx", {0, 1, 0, qc::SX}},
      {"sxdg", {0, 1, 0, qc::SXdg}},
      {"rx", {0, 1, 1, qc::RX}},
      {"ry", {0, 1, 1, qc::RY}},
      {"rz", {0, 1, 1, qc::RZ}},
      {"u1", {0, 1, 1, qc::P}},
      {"p", {0, 1, 1, qc::P}},
      {"u2", {0, 1, 2, qc::U2}},
      {"u3", {0, 1, 3, qc::U}},
      {"U", {0, 1, 3, qc::U}},
      {"u", {0, 1, 3, qc::U}},
      {"teleport", {0, 3, 0, qc::Teleportation}},
      {"swap", {0, 2, 0, qc::SWAP}},
      {"iswap", {0, 2, 0, qc::iSWAP}},
      {"cnot", {1, 1, 0, qc::X}},
      {"CX", {1, 1, 0, qc::X}},
      {"cx", {1, 1, 0, qc::X}},
      {"dcx", {0, 2, 0, qc::DCX}},
      {"ecr", {0, 2, 0, qc::ECR}},
      {"rxx", {0, 2, 1, qc::RXX}},
      {"ryy", {0, 2, 1, qc::RYY}},
      {"rzz", {0, 2, 1, qc::RZZ}},
      {"rzx", {0, 2, 1, qc::RZX}},
      {"xx_minus_yy", {0, 2, 2, qc::XXminusYY}},
      {"xx_plus_yy", {0, 2, 2, qc::XXplusYY}}};

  explicit Parser(std::istream& is, qc::QuantumRegisterMap& q,
                  qc::ClassicalRegisterMap& c)
      : in(is), qregs(q), cregs(c) {
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

  static bool gateInfo(const std::string& name, GateInfo& info);

  void parseParameters(const GateInfo& info, std::vector<qc::fp>& parameters);

  void parseArguments(const GateInfo& info, std::vector<qc::fp>& parameters,
                      std::vector<qc::QuantumRegister>& controlRegisters,
                      std::vector<qc::QuantumRegister>& targetRegisters);

  std::unique_ptr<qc::Operation>
  knownGate(const GateInfo& info, const std::vector<qc::fp>& parameters,
            const std::vector<qc::QuantumRegister>& controlRegisters,
            const std::vector<qc::QuantumRegister>& targetRegisters);

  std::unique_ptr<qc::Operation> knownGate(const GateInfo& info) {
    std::vector<qc::fp> parameters{};
    std::vector<qc::QuantumRegister> controlRegisters{};
    std::vector<qc::QuantumRegister> targetRegisters{};
    parseArguments(info, parameters, controlRegisters, targetRegisters);
    return knownGate(info, parameters, controlRegisters, targetRegisters);
  }

  void error [[noreturn]] (const std::string& msg) const {
    std::ostringstream oss{};
    oss << "[qasm parser] l:" << t.line << " c:" << t.col << " msg: " << msg;
    throw std::runtime_error(oss.str());
  }

  void handleComment();
  // check string for I/O layout information of the form
  //      'i Q_i Q_j ... Q_k' meaning, e.g. q_0 is mapped to Q_i, q_1 to Q_j,
  //      etc. 'o Q_i Q_j ... Q_k' meaning, e.g. q_0 is found at Q_i, q_1 at
  //      Q_j, etc.
  // where i describes the initial layout, e.g. 'i 2 1 0' means q0 -> Q2, q1 ->
  // Q1, q2 -> Q0 and o describes the output permutation, e.g. 'o 2 1 0' means
  // q0 is expected at Q2, q1 at Q1, and q2 at Q0
  static qc::Permutation checkForInitialLayout(std::string comment);
  static qc::Permutation checkForOutputPermutation(std::string comment);
};

} // namespace qasm
