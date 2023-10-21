#pragma once

#include "../InstVisitor.hpp"
#include "../Types.hpp"
#include "CompilerPass.hpp"
#include "ConstEvalPass.hpp"

namespace qasm3 {

struct InferredType {
public:
  bool isError;
  std::shared_ptr<ResolvedType> type;

  InferredType(bool isError, std::shared_ptr<ResolvedType> type)
      : isError(isError), type(type) {}

  explicit InferredType(std::shared_ptr<ResolvedType> type)
      : isError(false), type(type) {}

  static InferredType error() { return InferredType{true, nullptr}; }

  bool matches(const InferredType& other) {
    if (isError || other.isError) {
      return true;
    }

    return *type == *other.type;
  }

  std::string to_string() {
    if (isError) {
      return "error";
    }
    return type->to_string();
  }
};

class TypeCheckPass : public CompilerPass,
                      public InstVisitor,
                      public ExpressionVisitor<InferredType>,
                      public TypeVisitor<std::shared_ptr<Expression>> {
private:
  bool hasError = false;
  std::map<std::string, InferredType> env;
  // We need a reference to a const eval pass to evaluate types before type
  // checking.
  ConstEvalPass* constEvalPass;

  InferredType error(const std::string& msg,
                     std::shared_ptr<DebugInfo> debugInfo = nullptr) {
    std::cerr << "Type check error: " << msg << '\n';
    if (debugInfo) {
      std::cerr << "  " << debugInfo->toString() << '\n';
    }
    hasError = true;
    return InferredType::error();
  }

public:
  TypeCheckPass(ConstEvalPass* constEvalPass) : constEvalPass(constEvalPass) {}

  ~TypeCheckPass() override = default;

  void addBuiltin(std::string identifier, InferredType ty) {
    env.emplace(identifier, ty);
  }

  void processStatement(Statement& statement) override {
    statement.accept(this);

    if (hasError) {
      error("Type check failed for statement.", statement.debugInfo);
      throw std::runtime_error("Type check failed.");
    }
  }

  void checkGateOperand(GateOperand& operand) {
    if (operand.expression == nullptr) {
      return;
    }

    auto type = visit(operand.expression);
    if (!type.isError && !type.type->isUint()) {
      error("Index must be an unsigned integer");
    }
  }

  void
  visitGateStatement(std::shared_ptr<GateDeclaration> gateStatement) override;
  void visitVersionDeclaration(
      std::shared_ptr<VersionDeclaration> versionDeclaration) override;
  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override;
  void
  visitInitialLayout(std::shared_ptr<InitialLayout> initialLayout) override;
  void visitOutputPermutation(
      std::shared_ptr<OutputPermutation> outputPermutation) override;
  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override;
  void visitAssignmentStatement(
      std::shared_ptr<AssignmentStatement> assignmentStatement) override;
  void visitBarrierStatement(
      std::shared_ptr<BarrierStatement> barrierStatement) override;
  void
  visitResetStatement(std::shared_ptr<ResetStatement> resetStatement) override;
  InferredType visitBinaryExpression(
      std::shared_ptr<BinaryExpression> binaryExpression) override;
  InferredType visitUnaryExpression(
      std::shared_ptr<UnaryExpression> unaryExpression) override;
  InferredType
  visitConstantExpression(std::shared_ptr<Constant> constantInt) override;
  InferredType visitIdentifierExpression(
      std::shared_ptr<IdentifierExpression> identifierExpression) override;
  InferredType
  visitIdentifierList(std::shared_ptr<IdentifierList> identifierList) override;
  InferredType visitMeasureExpression(
      std::shared_ptr<MeasureExpression> measureExpression) override;
  std::shared_ptr<ResolvedType>
  visitDesignatedType(DesignatedType* designatedType) override;
  std::shared_ptr<ResolvedType> visitUnsizedType(
      UnsizedType<std::shared_ptr<Expression>>* unsizedType) override;
  std::shared_ptr<ResolvedType>
  visitArrayType(ArrayType<std::shared_ptr<Expression>>* arrayType) override;
};
} // namespace qasm3
