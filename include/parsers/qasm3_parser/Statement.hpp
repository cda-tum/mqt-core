#pragma once

#include "InstVisitor.hpp"
#include "QuantumComputation.hpp"
#include "Types.hpp"

#include <any>
#include <string>
#include <utility>
#include <vector>

namespace qasm3 {

struct DebugInfo {
  size_t line;
  size_t column;
  size_t endLine;
  size_t endColumn;
  std::string filename;
  std::shared_ptr<DebugInfo> parent;

  DebugInfo(size_t line, size_t column, size_t endLine, size_t endColumn,
            std::string filename, std::shared_ptr<DebugInfo> parent = nullptr)
      : line(line), column(column), endLine(endLine), endColumn(endColumn),
        filename(std::move(std::move(filename))), parent(std::move(parent)) {}

  [[nodiscard]] std::string toString() const {
    // TOOD: also print endLine and endColumn
    return filename + ":" + std::to_string(line) + ":" + std::to_string(column);
  }
};

// Expressions
class Expression {
public:
  // TODO: in the future, add inferred type here.
  virtual ~Expression() = default;

  virtual std::string getName() = 0;
};

class DeclarationExpression {
public:
  std::shared_ptr<Expression> expression;

  explicit DeclarationExpression(std::shared_ptr<Expression> expression)
      : expression(std::move(expression)) {}

  virtual ~DeclarationExpression() = default;
};

class Constant : public Expression {
private:
  std::variant<int64_t, double> val;
  bool isSigned;
  bool isFp;

public:
  Constant(int64_t val, bool isSigned)
      : val(val), isSigned(isSigned), isFp(false) {}

  explicit Constant(double val) : val(val), isSigned(true), isFp(true) {}

  [[nodiscard]] bool isInt() const { return !isFp; }
  [[nodiscard]] bool isSInt() const { return !isFp && isSigned; }
  [[nodiscard]] bool isUInt() const { return !isFp && !isSigned; }
  [[nodiscard]] bool isFP() const { return isFp; };
  [[nodiscard]] virtual int64_t getSInt() const { return std::get<0>(val); };
  [[nodiscard]] virtual uint64_t getUInt() const {
    return static_cast<uint64_t>(std::get<0>(val));
  };
  [[nodiscard]] virtual double getFP() const { return std::get<1>(val); };
  [[nodiscard]] virtual double asFP() const {
    if (isFp) {
      return getFP();
    }
    if (isSigned) {
      return static_cast<double>(getSInt());
    }
    return static_cast<double>(getUInt());
  };

  std::string getName() override { return "Constant"; }
};

class BinaryExpression : public Expression,
                         public std::enable_shared_from_this<BinaryExpression> {
public:
  enum Op {
    Power,
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    LeftShift,
    RightShift,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
    BitwiseAnd,
    BitwiseXor,
    BitwiseOr,
    LogicalAnd,
    LogicalOr,
  };

  Op op;
  std::shared_ptr<Expression> lhs;
  std::shared_ptr<Expression> rhs;

  BinaryExpression(Op op, std::shared_ptr<Expression> lhs,
                   std::shared_ptr<Expression> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  std::string getName() override { return "BinaryExpr"; }
};

class UnaryExpression : public Expression,
                        public std::enable_shared_from_this<UnaryExpression> {
public:
  enum Op {
    BitwiseNot,
    LogicalNot,
    Negate,
    DurationOf,
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
  };

  std::shared_ptr<Expression> operand;
  Op op;

  UnaryExpression(Op op, std::shared_ptr<Expression> operand)
      : operand(std::move(operand)), op(op) {}

  std::string getName() override { return "UnaryExpr"; }
};

class IdentifierExpression
    : public Expression,
      public std::enable_shared_from_this<IdentifierExpression> {
public:
  std::string identifier;

  explicit IdentifierExpression(std::string identifier)
      : identifier(std::move(identifier)) {}

  std::string getName() override {
    return std::string{"IdentifierExpr ("} + identifier + ")";
  }
};

class IdentifierList : public Expression,
                       public std::enable_shared_from_this<IdentifierList> {
public:
  std::vector<std::shared_ptr<IdentifierExpression>> identifiers;

  explicit IdentifierList(
      std::vector<std::shared_ptr<IdentifierExpression>> identifiers)
      : identifiers(std::move(identifiers)) {}

  explicit IdentifierList() = default;

  std::string getName() override { return "IdentifierList"; }
};

// TODO: physical qubits are currently not supported
class GateOperand {
public:
  std::string identifier;
  std::shared_ptr<Expression> expression;

  GateOperand(std::string identifier, std::shared_ptr<Expression> expression)
      : identifier(std::move(identifier)), expression(std::move(expression)) {}
};

class MeasureExpression
    : public Expression,
      public std::enable_shared_from_this<MeasureExpression> {
public:
  std::shared_ptr<GateOperand> gate;

  explicit MeasureExpression(std::shared_ptr<GateOperand> gate)
      : gate(std::move(gate)) {}

  std::string getName() override { return "MeasureExpression"; }
};

// Statements

class Statement {
public:
  std::shared_ptr<DebugInfo> debugInfo;
  explicit Statement(std::shared_ptr<DebugInfo> debugInfo)
      : debugInfo(std::move(debugInfo)) {}
  virtual ~Statement() = default;

  virtual void accept(InstVisitor* visitor) = 0;
};

class GateDeclaration : public Statement,
                        public std::enable_shared_from_this<GateDeclaration> {
public:
  std::string identifier;
  std::shared_ptr<IdentifierList> parameters;
  std::shared_ptr<IdentifierList> qubits;
  std::vector<std::shared_ptr<GateCallStatement>> statements;

  explicit GateDeclaration(
      std::shared_ptr<DebugInfo> debugInfo, std::string identifier,
      std::shared_ptr<IdentifierList> parameters,
      std::shared_ptr<IdentifierList> qubits,
      std::vector<std::shared_ptr<GateCallStatement>> statements)
      : Statement(std::move(debugInfo)), identifier(std::move(identifier)),
        parameters(std::move(parameters)), qubits(std::move(qubits)),
        statements(std::move(statements)) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitGateStatement(shared_from_this());
  }
};

class VersionDeclaration
    : public Statement,
      public std::enable_shared_from_this<VersionDeclaration> {
public:
  double version;

  explicit VersionDeclaration(std::shared_ptr<DebugInfo> debugInfo,
                              double version)
      : Statement(std::move(debugInfo)), version(version) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitVersionDeclaration(shared_from_this());
  }
};

class InitialLayout : public Statement,
                      public std::enable_shared_from_this<InitialLayout> {
public:
  qc::Permutation permutation;

  explicit InitialLayout(std::shared_ptr<DebugInfo> debugInfo,
                         qc::Permutation permutation)
      : Statement(std::move(debugInfo)), permutation(std::move(permutation)) {}

private:
  void accept(InstVisitor* visitor) override {
    visitor->visitInitialLayout(shared_from_this());
  }
};

class OutputPermutation
    : public Statement,
      public std::enable_shared_from_this<OutputPermutation> {
public:
  qc::Permutation permutation;

  explicit OutputPermutation(std::shared_ptr<DebugInfo> debugInfo,
                             qc::Permutation permutation)
      : Statement(std::move(debugInfo)), permutation(std::move(permutation)) {}

private:
  void accept(InstVisitor* visitor) override {
    visitor->visitOutputPermutation(shared_from_this());
  }
};

class DeclarationStatement
    : public Statement,
      public std::enable_shared_from_this<DeclarationStatement> {
public:
  bool isConst;
  std::variant<std::shared_ptr<TypeExpr>, std::shared_ptr<ResolvedType>> type;
  std::string identifier;
  std::shared_ptr<DeclarationExpression> expression;

  DeclarationStatement(std::shared_ptr<DebugInfo> debugInfo, bool isConst,
                       std::shared_ptr<TypeExpr> type, std::string identifier,
                       std::shared_ptr<DeclarationExpression> expression)
      : Statement(std::move(debugInfo)), isConst(isConst), type(type),
        identifier(std::move(identifier)), expression(std::move(expression)) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitDeclarationStatement(shared_from_this());
  }
};

class GateModifier : public std::enable_shared_from_this<GateModifier> {
protected:
  GateModifier() {}

public:
  virtual ~GateModifier() = default;
};

class InvGateModifier : public GateModifier,
                        public std::enable_shared_from_this<InvGateModifier> {
public:
  explicit InvGateModifier() = default;
};

class PowGateModifier : public GateModifier,
                        public std::enable_shared_from_this<PowGateModifier> {
public:
  std::shared_ptr<Expression> expression;

  explicit PowGateModifier(std::shared_ptr<Expression> expression)
      : expression(std::move(expression)) {}
};

class CtrlGateModifier : public GateModifier,
                         public std::enable_shared_from_this<CtrlGateModifier> {
public:
  bool ctrlType;
  std::shared_ptr<Expression> expression;

  explicit CtrlGateModifier(bool ctrlType,
                            std::shared_ptr<Expression> expression)
      : ctrlType(ctrlType), expression(std::move(expression)) {}
};

class GateCallStatement
    : public Statement,
      public std::enable_shared_from_this<GateCallStatement> {
public:
  std::string identifier;
  std::vector<std::shared_ptr<GateModifier>> modifiers;
  std::vector<std::shared_ptr<Expression>> arguments;
  std::vector<std::shared_ptr<GateOperand>> operands;

  GateCallStatement(std::shared_ptr<DebugInfo> debugInfo,
                    std::string identifier,
                    std::vector<std::shared_ptr<GateModifier>> modifiers,
                    std::vector<std::shared_ptr<Expression>> arguments,
                    std::vector<std::shared_ptr<GateOperand>> operands)
      : Statement(std::move(debugInfo)), identifier(std::move(identifier)),
        modifiers(std::move(modifiers)), arguments(std::move(arguments)),
        operands(std::move(operands)) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitGateCallStatement(shared_from_this());
  }
};

class AssignmentStatement
    : public Statement,
      public std::enable_shared_from_this<AssignmentStatement> {
public:
  enum Type {
    Assignment,
    PlusAssignment,
    MinusAssignment,
    TimesAssignment,
    DivAssignment,
    BitwiseAndAssignment,
    BitwiseOrAssignment,
    BitwiseNotAssignment,
    BitwiseXorAssignment,
    LeftShiftAssignment,
    RightShiftAssignment,
    ModuloAssignment,
    PowerAssignment,
  } type;
  std::shared_ptr<IdentifierExpression> identifier;
  std::shared_ptr<Expression> indexExpression;
  std::shared_ptr<DeclarationExpression> expression;

  AssignmentStatement(std::shared_ptr<DebugInfo> debugInfo, Type type,
                      std::shared_ptr<IdentifierExpression> identifier,
                      std::shared_ptr<Expression> indexExpression,
                      std::shared_ptr<DeclarationExpression> expression)
      : Statement(std::move(debugInfo)), type(type),
        identifier(std::move(identifier)),
        indexExpression(std::move(indexExpression)),
        expression(std::move(expression)) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitAssignmentStatement(shared_from_this());
  }
};

class BarrierStatement : public Statement,
                         public std::enable_shared_from_this<BarrierStatement> {
public:
  std::vector<std::shared_ptr<GateOperand>> gates;

  explicit BarrierStatement(std::shared_ptr<DebugInfo> debugInfo,
                            std::vector<std::shared_ptr<GateOperand>> gates)
      : Statement(std::move(debugInfo)), gates(std::move(gates)) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitBarrierStatement(shared_from_this());
  }
};

class ResetStatement : public Statement,
                       public std::enable_shared_from_this<ResetStatement> {
public:
  std::shared_ptr<GateOperand> gate;

  explicit ResetStatement(std::shared_ptr<DebugInfo> debugInfo,
                          std::shared_ptr<GateOperand> gate)
      : Statement(std::move(debugInfo)), gate(std::move(gate)) {}

  void accept(InstVisitor* visitor) override {
    visitor->visitResetStatement(shared_from_this());
  }
};
} // namespace qasm3
