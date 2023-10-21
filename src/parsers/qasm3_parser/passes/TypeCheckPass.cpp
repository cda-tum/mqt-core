#include "parsers/qasm3_parser/passes/TypeCheckPass.hpp"

#include <vector>

namespace qasm3 {

void TypeCheckPass::visitGateStatement(
    std::shared_ptr<GateDeclaration> gateStatement) {
  // we save the current environment to restore it afterwards
  auto oldEnv = env;

  for (auto& param : gateStatement->parameters->identifiers) {
    env.emplace(param->identifier, InferredType{SizedType::getFloatTy()});
  }
  for (auto& operand : gateStatement->qubits->identifiers) {
    env.emplace(operand->identifier, InferredType{SizedType::getQubitTy()});
  }

  for (auto& stmt : gateStatement->statements) {
    stmt->accept(this);
  }

  // restore the environment
  env = oldEnv;
}

void TypeCheckPass::visitVersionDeclaration(
    std::shared_ptr<VersionDeclaration> /*versionDeclaration*/) {}

void TypeCheckPass::visitDeclarationStatement(
    std::shared_ptr<DeclarationStatement> declarationStatement) {
  // Type checking declarations is a bit involved. If the type contains a
  // designator expression, we need to resolve the statement in three steps.
  auto typeExpr = std::get<0>(declarationStatement->type);
  // First, type-check the type itself.
  if (typeExpr->getDesignator() != nullptr &&
      visit(typeExpr->getDesignator()).isError) {
    error("Designator expression type check failed.",
          declarationStatement->debugInfo);
    return;
  }
  // Now we know the type is valid, we can evaluate the designator expression.
  auto resolvedType =
      std::get<0>(declarationStatement->type)->accept(constEvalPass);
  if (!resolvedType) {
    throw std::runtime_error("Expression in types must be const.");
  }
  declarationStatement->type = resolvedType;

  // Lastly, we type check the actual expression
  if (declarationStatement->expression != nullptr) {
    auto exprType = visit(declarationStatement->expression->expression);
    if (!resolvedType->fits(*exprType.type)) {
      std::stringstream ss;
      ss << "Type mismatch in declaration statement: Expected '";
      ss << resolvedType->to_string();
      ss << "', found '";
      ss << exprType.type->to_string();
      ss << "'.";
      error(ss.str());
    }
  }

  env.emplace(declarationStatement->identifier, InferredType{resolvedType});
}

void TypeCheckPass::visitInitialLayout(
    std::shared_ptr<InitialLayout> /*initialLayout*/) {}

void TypeCheckPass::visitOutputPermutation(
    std::shared_ptr<OutputPermutation> /*outputPermutation*/) {}

void TypeCheckPass::visitGateCallStatement(
    std::shared_ptr<GateCallStatement> gateCallStatement) {
  for (auto& arg : gateCallStatement->arguments) {
    visit(arg);
  }
}

void TypeCheckPass::visitAssignmentStatement(
    std::shared_ptr<AssignmentStatement> assignmentStatement) {
  if (assignmentStatement->indexExpression != nullptr) {
    auto indexTy = visit(assignmentStatement->indexExpression);
    if (!indexTy.isError && !indexTy.type->isUint()) {
      error("Index must be an unsigned integer.",
            assignmentStatement->debugInfo);
      return;
    }
  }

  auto exprTy = visit(assignmentStatement->expression->expression);
  auto idTy = env.find(assignmentStatement->identifier->identifier);

  if (idTy == env.end()) {
    error("Unknown identifier '" + assignmentStatement->identifier->identifier +
              "'.",
          assignmentStatement->debugInfo);
    return;
  }

  if (!idTy->second.type->fits(*exprTy.type)) {
    std::stringstream ss;
    ss << "Type mismatch in assignment. Expected '";
    ss << idTy->second.type->to_string();
    ss << "', found '";
    ss << exprTy.type->to_string();
    ss << "'.";
    error(ss.str(), assignmentStatement->debugInfo);
  }
}

void TypeCheckPass::visitBarrierStatement(
    std::shared_ptr<BarrierStatement> barrierStatement) {
  for (auto& gate : barrierStatement->gates) {
    if (!gate->expression) {
      continue;
    }
    checkGateOperand(*gate);
  }
}

void TypeCheckPass::visitResetStatement(
    std::shared_ptr<ResetStatement> resetStatement) {
  checkGateOperand(*resetStatement->gate);
}

InferredType TypeCheckPass::visitBinaryExpression(
    std::shared_ptr<BinaryExpression> binaryExpression) {
  auto lhs = visit(binaryExpression->lhs);
  auto rhs = visit(binaryExpression->rhs);
  if (rhs.isError) {
    return rhs;
  }
  if (lhs.isError) {
    return lhs;
  }

  auto ty = lhs;
  if (lhs.type->isNumber() && rhs.type->isFP()) {
    // we allow implicit coercion to float
    ty = rhs;
  } else if (rhs.type->isNumber() && lhs.type->isFP()) {
    ty = lhs;
  } else if (lhs.type->isNumber() && rhs.type->isUint()) {
    // we allow implicit coercion to signed int
    ty = lhs;
  } else if (rhs.type->isNumber() && lhs.type->isUint()) {
    ty = rhs;
  } else if (lhs.type != rhs.type) {
    std::stringstream ss;
    ss << "Type mismatch in binary expression: ";
    ss << lhs.type->to_string();
    ss << ", ";
    ss << rhs.type->to_string();
    ss << ".";
    error(ss.str());
    return InferredType::error();
  }

  switch (binaryExpression->op) {
  case BinaryExpression::Power:
  case BinaryExpression::Add:
  case BinaryExpression::Subtract:
  case BinaryExpression::Multiply:
  case BinaryExpression::Divide:
  case BinaryExpression::Modulo:
  case BinaryExpression::LeftShift:
  case BinaryExpression::RightShift:
    if (!ty.type->isNumber()) {
      error("Cannot apply arithmetic operation to non-numeric type.");
      return InferredType::error();
    }
    break;
  case BinaryExpression::LessThan:
  case BinaryExpression::LessThanOrEqual:
  case BinaryExpression::GreaterThan:
  case BinaryExpression::GreaterThanOrEqual:
    // all types except for bool
    if (ty.type->isBool()) {
      error("Cannot compare boolean types.");
      return InferredType::error();
    }
    break;
  case BinaryExpression::Equal:
  case BinaryExpression::NotEqual:
    break;
  case BinaryExpression::BitwiseAnd:
  case BinaryExpression::BitwiseXor:
  case BinaryExpression::BitwiseOr:
    if (!ty.type->isNumber()) {
      error("Cannot apply bitwise operation to non-numeric type.");
      return InferredType::error();
    }
    break;
  case BinaryExpression::LogicalAnd:
  case BinaryExpression::LogicalOr:
    if (!ty.type->isBool()) {
      error("Cannot apply logical operation to non-boolean type.");
      return InferredType::error();
    }
    break;
  }

  return ty;
}

InferredType TypeCheckPass::visitUnaryExpression(
    std::shared_ptr<UnaryExpression> unaryExpression) {
  auto type = visit(unaryExpression->operand);

  switch (unaryExpression->op) {
  case UnaryExpression::BitwiseNot:
    if (!type.type->isNumber()) {
      error("Cannot apply bitwise not to non-numeric type.");
      return InferredType::error();
    }
    break;
  case UnaryExpression::LogicalNot:
    if (!type.type->isBool()) {
      error("Cannot apply logical not to non-boolean type.");
      return InferredType::error();
    }
    break;
  case UnaryExpression::Negate:
    break;
  case UnaryExpression::DurationOf:
    return InferredType{UnsizedType<uint64_t>::getDurationTy()};
  case UnaryExpression::Sin:
  case UnaryExpression::Cos:
  case UnaryExpression::Tan:
  case UnaryExpression::Exp:
  case UnaryExpression::Ln:
  case UnaryExpression::Sqrt:
    return InferredType{SizedType::getFloatTy()};
  }

  return type;
}

InferredType
TypeCheckPass::visitConstantExpression(std::shared_ptr<Constant> constant) {
  if (constant->isFP()) {
    return InferredType{SizedType::getFloatTy()};
  }
  assert(constant->isInt());

  if (constant->isSInt()) {
    return InferredType{SizedType::getIntTy()};
  }

  return InferredType{SizedType::getUintTy()};
}

InferredType TypeCheckPass::visitIdentifierExpression(
    std::shared_ptr<IdentifierExpression> identifierExpression) {
  auto type = env.find(identifierExpression->identifier);
  if (type == env.end()) {
    error("Unknown identifier '" + identifierExpression->identifier + "'.");
    return InferredType::error();
  }
  return type->second;
}

InferredType TypeCheckPass::visitIdentifierList(
    std::shared_ptr<IdentifierList> /*identifierList*/) {
  throw std::runtime_error(
      "TypeCheckPass::visitIdentifierList not implemented");
}

InferredType TypeCheckPass::visitMeasureExpression(
    std::shared_ptr<MeasureExpression> measureExpression) {
  size_t width = 1;
  if (measureExpression->gate->expression != nullptr) {
    visit(measureExpression->gate->expression);
  } else {
    auto gate = env.find(measureExpression->gate->identifier);
    if (gate == env.end()) {
      error("Unknown identifier '" + measureExpression->gate->identifier +
            "'.");
      return InferredType::error();
    }
    width = gate->second.type->getDesignator();
  }

  return InferredType{SizedType::getBitTy(width)};
}
std::shared_ptr<ResolvedType>
TypeCheckPass::visitDesignatedType(DesignatedType* designatedType) {
  auto resolvedTy = visit(designatedType->designator);
  if (!resolvedTy.isError && !resolvedTy.type->isUint()) {
    error("Designator must be an unsigned integer");
  }
  return nullptr;
}
std::shared_ptr<ResolvedType> TypeCheckPass::visitUnsizedType(
    UnsizedType<std::shared_ptr<Expression>>* /*unsizedType*/) {
  // Nothing to type check
  return nullptr;
}
std::shared_ptr<ResolvedType> TypeCheckPass::visitArrayType(
    ArrayType<std::shared_ptr<Expression>>* arrayType) {
  // TODO: check the size once it is converted to an expression.
  arrayType->accept(this);
  return nullptr;
}
} // namespace qasm3
