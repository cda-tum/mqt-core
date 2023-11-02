#pragma once

#include "../NestedEnvironment.hpp"
#include "CompilerPass.hpp"

namespace qasm3 {
namespace const_eval {
struct ConstEvalValue {
  enum Type {
    ConstInt,
    ConstUint,
    ConstFloat,
    ConstBool,
  } type;
  std::variant<int64_t, double, bool> value;
  size_t width;

  explicit ConstEvalValue(double val, size_t w = 64)
      : type(ConstFloat), value(val), width(w) {}
  explicit ConstEvalValue(int64_t val, bool isSigned, size_t w = 64)
      : type(isSigned ? Type::ConstInt : Type::ConstUint), value(val),
        width(w) {}

  std::shared_ptr<Constant> toExpr() {
    switch (type) {
    case ConstEvalValue::ConstInt:
      return std::make_shared<Constant>(Constant(std::get<0>(value), true));
    case ConstEvalValue::ConstUint:
      return std::make_shared<Constant>(Constant(std::get<0>(value), false));
    case ConstEvalValue::ConstFloat:
      return std::make_shared<Constant>(Constant(std::get<1>(value)));
    case ConstEvalValue::ConstBool:
      return std::make_shared<Constant>(
          Constant(static_cast<int64_t>(std::get<2>(value)), false));
    }

    __builtin_unreachable();
  }
};

class ConstEvalPass : public CompilerPass,
                      public DefaultInstVisitor,
                      public ExpressionVisitor<std::optional<ConstEvalValue>>,
                      public TypeVisitor<std::shared_ptr<Expression>> {
private:
  NestedEnvironment<ConstEvalValue> env{};

  template <typename T> int64_t castToWidth(int64_t value) {
    return static_cast<int64_t>(static_cast<T>(value));
  }

  ConstEvalValue evalIntExpression(BinaryExpression::Op op, int64_t lhs,
                                   int64_t rhs, size_t width, bool isSigned);
  ConstEvalValue evalFloatExpression(BinaryExpression::Op op, double lhs,
                                     double rhs);
  ConstEvalValue evalBoolExpression(BinaryExpression::Op op, bool lhs,
                                    bool rhs);

public:
  ConstEvalPass() = default;
  ~ConstEvalPass() override = default;

  void addConst(const std::string& identifier, ConstEvalValue val) {
    env.emplace(identifier, val);
  }

  void addConst(const std::string& identifier, double val) {
    env.emplace(identifier, ConstEvalValue(val));
  }

  void processStatement(Statement& statement) override {
    statement.accept(this);
  }

  void pushEnv() { env.push(); }
  void popEnv() { env.pop(); }

  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override;
  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override;

  std::optional<ConstEvalValue> visitBinaryExpression(
      std::shared_ptr<BinaryExpression> binaryExpression) override;
  std::optional<ConstEvalValue> visitUnaryExpression(
      std::shared_ptr<UnaryExpression> unaryExpression) override;
  std::optional<ConstEvalValue>
  visitConstantExpression(std::shared_ptr<Constant> constant) override;
  std::optional<ConstEvalValue> visitIdentifierExpression(
      std::shared_ptr<IdentifierExpression> identifierExpression) override;
  std::optional<ConstEvalValue>
  visitIdentifierList(std::shared_ptr<IdentifierList> identifierList) override;
  std::optional<ConstEvalValue> visitMeasureExpression(
      std::shared_ptr<MeasureExpression> measureExpression) override;

  std::shared_ptr<ResolvedType>
  visitDesignatedType(DesignatedType* designatedType) override;
  std::shared_ptr<ResolvedType> visitUnsizedType(
      UnsizedType<std::shared_ptr<Expression>>* unsizedType) override;
  std::shared_ptr<ResolvedType>
  visitArrayType(ArrayType<std::shared_ptr<Expression>>* arrayType) override;
};
} // namespace const_eval
} // namespace qasm3
