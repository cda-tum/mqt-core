#include "parsers/qasm3_parser/passes/ConstEvalPass.hpp"

namespace qasm3 {
namespace const_eval {
void ConstEvalPass::visitDeclarationStatement(
    const std::shared_ptr<DeclarationStatement> declarationStatement) {
  // The type designator expression is already resolved by the type check pass.
  if (!declarationStatement->isConst) {
    return;
  }

  const auto value = visit(declarationStatement->expression->expression);
  if (!value) {
    throw std::runtime_error(
        "Constant declaration initialization expression must be const.");
  }

  declarationStatement->expression->expression = value->toExpr();

  this->env.emplace(declarationStatement->identifier, *value);
}

void ConstEvalPass::visitGateCallStatement(
    std::shared_ptr<GateCallStatement> gateCallStatement) {
  for (auto& arg : gateCallStatement->arguments) {
    if (auto evaluatedArg = visit(arg)) {
      arg = evaluatedArg->toExpr();
    }
  }
  for (auto& op : gateCallStatement->operands) {
    if (op->expression == nullptr) {
      continue;
    }
    if (auto evaluatedArg = visit(op->expression)) {
      op->expression = evaluatedArg->toExpr();
    }
  }
  for (auto& modifier : gateCallStatement->modifiers) {
    if (auto powModifier = std::dynamic_pointer_cast<PowGateModifier>(modifier);
        powModifier != nullptr && powModifier->expression != nullptr) {
      if (auto evaluatedArg = visit(powModifier->expression)) {
        powModifier->expression = evaluatedArg->toExpr();
      }
    } else if (auto ctrlModifier =
                   std::dynamic_pointer_cast<CtrlGateModifier>(modifier);
               ctrlModifier != nullptr && ctrlModifier->expression != nullptr) {
      if (auto evaluatedArg = visit(ctrlModifier->expression)) {
        ctrlModifier->expression = evaluatedArg->toExpr();
      }
    }
  }
}

ConstEvalValue ConstEvalPass::evalIntExpression(BinaryExpression::Op op,
                                                int64_t lhs, int64_t rhs,
                                                size_t width, bool isSigned) {
  auto lhsU = static_cast<uint64_t>(lhs);
  auto rhsU = static_cast<uint64_t>(rhs);
  ConstEvalValue result{0, isSigned, width};

  switch (op) {
  case BinaryExpression::Power:
    if (isSigned) {
      result.value = static_cast<int64_t>(pow(lhs, rhs));
    } else {
      result.value = static_cast<int64_t>(pow(lhsU, rhsU));
    }
    break;
  case BinaryExpression::Add:
    result.value = static_cast<int64_t>(lhsU + rhsU);
    break;
  case BinaryExpression::Subtract:
    result.value = static_cast<int64_t>(lhsU - rhsU);
    break;
  case BinaryExpression::Multiply:
    if (isSigned) {
      result.value = lhs * rhs;
    } else {
      result.value = static_cast<int64_t>(lhsU * rhsU);
    }
    break;
  case BinaryExpression::Divide:
    if (isSigned) {
      result.value = lhs / rhs;
    } else {
      result.value = static_cast<int64_t>(lhsU / rhsU);
    }
    break;
  case BinaryExpression::Modulo:
    if (isSigned) {
      result.value = lhs % rhs;
    } else {
      result.value = static_cast<int64_t>(lhsU % rhsU);
    }
    break;
  case BinaryExpression::LeftShift:
    if (isSigned) {
      result.value = lhs << rhs;
    } else {
      result.value = static_cast<int64_t>(lhsU << rhsU);
    }
    break;
  case BinaryExpression::RightShift:
    if (isSigned) {
      result.value = lhs >> rhs;
    } else {
      result.value = static_cast<int64_t>(lhsU >> rhsU);
    }
    break;
  case BinaryExpression::LessThan:
    if (isSigned) {
      result.value = lhs < rhs;
    } else {
      result.value = lhsU < rhsU;
    }
    break;
  case BinaryExpression::LessThanOrEqual:
    if (isSigned) {
      result.value = lhs <= rhs;
    } else {
      result.value = lhsU <= rhsU;
    }
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::GreaterThan:
    if (isSigned) {
      result.value = lhs > rhs;
    } else {
      result.value = lhsU > rhsU;
    }
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::GreaterThanOrEqual:
    if (isSigned) {
      result.value = lhs >= rhs;
    } else {
      result.value = lhsU >= rhsU;
    }
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::Equal:
    if (isSigned) {
      result.value = lhs == rhs;
    } else {
      result.value = lhsU == rhsU;
    }
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::NotEqual:
    if (isSigned) {
      result.value = lhs != rhs;
    } else {
      result.value = lhsU != rhsU;
    }
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::BitwiseAnd:
    result.value = static_cast<int64_t>(lhsU & rhsU);
    break;
  case BinaryExpression::BitwiseXor:
    result.value = static_cast<int64_t>(lhsU ^ rhsU);
    break;
  case BinaryExpression::BitwiseOr:
    result.value = static_cast<int64_t>(lhsU | rhsU);
    break;
  default:
    throw std::runtime_error(
        "Unsupported binary expression operator on integer.");
  }

  // now we need to make sure the result is correct according to the bit width
  // of the types
  if (result.type == ConstEvalValue::ConstInt ||
      result.type == ConstEvalValue::ConstUint) {
    switch (width) {
    case 8:
      result.value = castToWidth<int8_t>(std::get<0>(result.value));
      break;
    case 16:
      result.value = castToWidth<int16_t>(std::get<0>(result.value));
      break;
    case 32:
      result.value = castToWidth<int32_t>(std::get<0>(result.value));
      break;
    case 64:
      result.value = castToWidth<int64_t>(std::get<0>(result.value));
      break;
    default:
      throw std::runtime_error("Unsupported bit width.");
    }
  }

  return result;
}

ConstEvalValue ConstEvalPass::evalFloatExpression(BinaryExpression::Op op,
                                                  double lhs, double rhs) {
  ConstEvalValue result{0.0};

  switch (op) {
  case BinaryExpression::Power:
    result.value = pow(lhs, rhs);
    break;
  case BinaryExpression::Add:
    result.value = lhs + rhs;
    break;
  case BinaryExpression::Subtract:
    result.value = lhs - rhs;
    break;
  case BinaryExpression::Multiply:
    result.value = lhs * rhs;
    break;
  case BinaryExpression::Divide:
    result.value = lhs / rhs;
    break;
  case BinaryExpression::Modulo:
    result.value = fmod(lhs, rhs);
    break;
  case BinaryExpression::LessThan:
    result.value = lhs < rhs;
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::LessThanOrEqual:
    result.value = lhs <= rhs;
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::GreaterThan:
    result.value = lhs > rhs;
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::GreaterThanOrEqual:
    result.value = lhs >= rhs;
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::Equal:
    result.value = lhs == rhs;
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  case BinaryExpression::NotEqual:
    result.value = lhs != rhs;
    result.type = ConstEvalValue::Type::ConstBool;
    break;
  default:
    throw std::runtime_error(
        "Unsupported binary expression operator on floating point.");
  }

  return result;
}
ConstEvalValue ConstEvalPass::evalBoolExpression(const BinaryExpression::Op op,
                                                 const bool lhs,
                                                 const bool rhs) {
  ConstEvalValue result{ConstEvalValue::Type::ConstBool, false, 1};

  switch (op) {
  case BinaryExpression::Op::Equal:
    result.value = lhs == rhs;
    break;
  case BinaryExpression::Op::NotEqual:
    result.value = lhs != rhs;
    break;
  case BinaryExpression::Op::BitwiseAnd:
    result.value = lhs && rhs;
    break;
  case BinaryExpression::Op::BitwiseXor:
    result.value = lhs != rhs;
    break;
  case BinaryExpression::Op::BitwiseOr:
    result.value = lhs || rhs;
    break;
  case BinaryExpression::Op::LogicalAnd:
    result.value = lhs && rhs;
    break;
  case BinaryExpression::Op::LogicalOr:
    result.value = lhs || rhs;
    break;
  default:
    throw std::runtime_error(
        "Unsupported binary expression operator on boolean.");
  }

  return result;
}

std::optional<ConstEvalValue> ConstEvalPass::visitBinaryExpression(
    const std::shared_ptr<BinaryExpression> binaryExpression) {
  auto lhsVal = visit(binaryExpression->lhs);
  if (!lhsVal) {
    return std::nullopt;
  }
  auto rhsVal = visit(binaryExpression->rhs);
  if (!rhsVal) {
    return std::nullopt;
  }

  // We need to coerce the values to the correct type.
  // ConstInt and ConstUint should be able to coerce to ConstFloat.
  // ConstUint should coerce to ConstInt.
  // ConstBool should not coerce.
  // All other combinations are disallowed.
  if ((lhsVal->type == ConstEvalValue::Type::ConstInt &&
       rhsVal->type == ConstEvalValue::Type::ConstUint) ||
      (lhsVal->type == ConstEvalValue::Type::ConstUint &&
       rhsVal->type == ConstEvalValue::Type::ConstInt)) {
    lhsVal->type = ConstEvalValue::Type::ConstUint;
    rhsVal->type = ConstEvalValue::Type::ConstUint;
  } else if (lhsVal->type == ConstEvalValue::Type::ConstUint &&
             rhsVal->type == ConstEvalValue::Type::ConstFloat) {
    lhsVal->value =
        static_cast<double>(static_cast<int64_t>(std::get<0>(lhsVal->value)));
    lhsVal->type = ConstEvalValue::Type::ConstFloat;
  } else if (lhsVal->type == ConstEvalValue::Type::ConstInt &&
             rhsVal->type == ConstEvalValue::Type::ConstFloat) {
    lhsVal->value = static_cast<double>(std::get<0>(lhsVal->value));
    lhsVal->type = ConstEvalValue::Type::ConstFloat;
  } else if (lhsVal->type == ConstEvalValue::Type::ConstFloat &&
             rhsVal->type == ConstEvalValue::Type::ConstUint) {
    rhsVal->value =
        static_cast<double>(static_cast<uint64_t>(std::get<0>(rhsVal->value)));
    rhsVal->type = ConstEvalValue::Type::ConstFloat;
  } else if (lhsVal->type == ConstEvalValue::Type::ConstFloat &&
             rhsVal->type == ConstEvalValue::Type::ConstInt) {
    rhsVal->value = static_cast<double>(std::get<0>(rhsVal->value));
    rhsVal->type = ConstEvalValue::Type::ConstFloat;
  } else if (lhsVal->type != rhsVal->type) {
    throw std::runtime_error(
        "Type mismatch, cannot evaluate binary expression on types " +
        std::to_string(lhsVal->type) + " and " + std::to_string(rhsVal->type) +
        ".");
  }

  size_t const width = std::max(lhsVal->width, rhsVal->width);

  switch (lhsVal->type) {
  case ConstEvalValue::Type::ConstInt:
    return evalIntExpression(binaryExpression->op, std::get<0>(lhsVal->value),
                             std::get<0>(rhsVal->value), width, true);
  case ConstEvalValue::Type::ConstUint:
    return evalIntExpression(binaryExpression->op, std::get<0>(lhsVal->value),
                             std::get<0>(rhsVal->value), width, false);
  case ConstEvalValue::Type::ConstFloat:
    return evalFloatExpression(binaryExpression->op, std::get<1>(lhsVal->value),
                               std::get<1>(rhsVal->value));
  case ConstEvalValue::Type::ConstBool:
    return evalBoolExpression(binaryExpression->op, std::get<2>(lhsVal->value),
                              std::get<2>(rhsVal->value));
  }

  throw std::runtime_error("Unhandled binary expression type.");
}

std::optional<ConstEvalValue> ConstEvalPass::visitUnaryExpression(
    const std::shared_ptr<UnaryExpression> unaryExpression) {

  auto val = visit(unaryExpression->operand);
  if (!val) {
    return std::nullopt;
  }

  switch (unaryExpression->op) {
  case UnaryExpression::BitwiseNot:
    val->value = ~std::get<0>(val->value);
    break;
  case UnaryExpression::LogicalNot:
    if (val->type == ConstEvalValue::Type::ConstBool) {
      val->value = !std::get<2>(val->value);
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::Negate:
    if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = -std::get<0>(val->value);
    } else if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = -std::get<1>(val->value);
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::DurationOf:
    return std::nullopt;
  case UnaryExpression::Sin:
    if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = sin(std::get<1>(val->value));
    } else if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = sin(static_cast<double>(std::get<0>(val->value)));
      val->type = ConstEvalValue::Type::ConstFloat;
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::Cos:
    if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = cos(std::get<1>(val->value));
    } else if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = cos(static_cast<double>(std::get<0>(val->value)));
      val->type = ConstEvalValue::Type::ConstFloat;
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::Tan:
    if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = tan(std::get<1>(val->value));
    } else if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = tan(static_cast<double>(std::get<0>(val->value)));
      val->type = ConstEvalValue::Type::ConstFloat;
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::Exp:
    if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = exp(std::get<1>(val->value));
    } else if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = exp(static_cast<double>(std::get<0>(val->value)));
      val->type = ConstEvalValue::Type::ConstFloat;
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::Ln:
    if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = log(std::get<1>(val->value));
    } else if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = log(static_cast<double>(std::get<0>(val->value)));
      val->type = ConstEvalValue::Type::ConstFloat;
    } else {
      return std::nullopt;
    }
    break;
  case UnaryExpression::Sqrt:
    if (val->type == ConstEvalValue::Type::ConstFloat) {
      val->value = sqrt(std::get<1>(val->value));
    } else if (val->type == ConstEvalValue::Type::ConstInt) {
      val->value = sqrt(static_cast<double>(std::get<0>(val->value)));
      val->type = ConstEvalValue::Type::ConstFloat;
    } else {
      return std::nullopt;
    }
    break;
  }

  return val;
}

std::optional<ConstEvalValue> ConstEvalPass::visitConstantExpression(
    const std::shared_ptr<Constant> constant) {

  if (constant->isFP()) {
    return ConstEvalValue{constant->getFP()};
  }
  if (constant->isSInt()) {
    return ConstEvalValue{constant->getSInt(), true};
  }
  assert(constant->isUInt());
  // we still call getSInt here as we will store the int value as its bit
  // representation and won't interpret it as such
  return ConstEvalValue{constant->getSInt(), false};
}

std::optional<ConstEvalValue> ConstEvalPass::visitIdentifierExpression(
    const std::shared_ptr<IdentifierExpression> identifierExpression) {
  return env.find(identifierExpression->identifier);
}

std::optional<ConstEvalValue> ConstEvalPass::visitIdentifierList(
    std::shared_ptr<IdentifierList> /*identifierList*/) {
  return std::nullopt;
}

std::optional<ConstEvalValue> ConstEvalPass::visitMeasureExpression(
    std::shared_ptr<MeasureExpression> /*measureExpression*/) {
  return std::nullopt;
}
std::shared_ptr<ResolvedType>
ConstEvalPass::visitDesignatedType(DesignatedType* designatedType) {
  if (designatedType->designator == nullptr) {
    return std::make_shared<SizedType>(designatedType->type);
  }
  const auto result = visit(designatedType->designator);
  if (!result) {
    throw std::runtime_error("Designator must be a constant expression.");
  }
  if (result->type != ConstEvalValue::Type::ConstUint) {
    throw std::runtime_error("Designator must be an unsigned integer.");
  }
  return std::make_shared<SizedType>(
      designatedType->type, static_cast<uint64_t>(std::get<0>(result->value)));
}
std::shared_ptr<ResolvedType> ConstEvalPass::visitUnsizedType(
    UnsizedType<std::shared_ptr<Expression>>* unsizedType) {
  return std::make_shared<UnsizedType<uint64_t>>(unsizedType->type);
}
std::shared_ptr<ResolvedType> ConstEvalPass::visitArrayType(
    ArrayType<std::shared_ptr<Expression>>* arrayType) {
  std::shared_ptr<Type<uint64_t>> const inner = arrayType->type->accept(this);
  const auto size = visit(arrayType->size);
  if (!size.has_value()) {
    throw std::runtime_error("Array size must be a constant expression.");
  }
  if (size->type != ConstEvalValue::Type::ConstUint) {
    throw std::runtime_error("Array size must be an unsigned integer.");
  }
  return std::make_shared<ArrayType<uint64_t>>(
      inner, static_cast<uint64_t>(std::get<0>(size->value)));
}
} // namespace const_eval
} // namespace qasm3
