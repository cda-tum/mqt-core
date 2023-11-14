#include "QuantumComputation.hpp"
#include "operations/Operation.hpp"
#include "parsers/qasm3_parser/Gate.hpp"
#include "parsers/qasm3_parser/InstVisitor.hpp"
#include "parsers/qasm3_parser/NestedEnvironment.hpp"
#include "parsers/qasm3_parser/Parser.hpp"
#include "parsers/qasm3_parser/Statement.hpp"
#include "parsers/qasm3_parser/StdGates.hpp"
#include "parsers/qasm3_parser/passes/ConstEvalPass.hpp"
#include "parsers/qasm3_parser/passes/TypeCheckPass.hpp"

#include <iostream>
#include <utility>

using namespace qasm3;
using qasm3::const_eval::ConstEvalPass;
using qasm3::const_eval::ConstEvalValue;
using qasm3::type_checking::InferredType;
using qasm3::type_checking::TypeCheckPass;

struct CompilerError : public std::exception {
  std::string message;
  std::shared_ptr<DebugInfo> debugInfo;

  CompilerError(std::string msg, std::shared_ptr<DebugInfo> debug)
      : message(std::move(msg)), debugInfo(std::move(debug)) {}

  [[nodiscard]] std::string toString() const {
    std::stringstream ss;
    ss << debugInfo->toString();

    auto parentDebugInfo = debugInfo->parent;
    while (parentDebugInfo != nullptr) {
      ss << "\n  (included from " << parentDebugInfo->toString() << ")";
      parentDebugInfo = parentDebugInfo->parent;
    }

    ss << ":\n" << message;

    return ss.str();
  }
};

class OpenQasm3Parser : public InstVisitor {
  ConstEvalPass constEvalPass;
  TypeCheckPass typeCheckPass;

  NestedEnvironment<std::shared_ptr<DeclarationStatement>> declarations{};
  qc::QuantumComputation* qc;

  std::vector<std::unique_ptr<qc::Operation>> ops{};

  std::map<std::string, std::shared_ptr<Gate>> gates = STANDARD_GATES;

  [[noreturn]] static void error(const std::string& message,
                                 const std::shared_ptr<DebugInfo>& debugInfo) {
    throw CompilerError(message, debugInfo);
  }

  static std::map<std::string, std::pair<ConstEvalValue, InferredType>>
  initializeBuiltins() {
    std::map<std::string, std::pair<ConstEvalValue, InferredType>> builtins{};

    InferredType const floatTy{std::make_shared<SizedType>(Float, 64)};

    builtins.emplace("pi", std::pair{ConstEvalValue(qc::PI), floatTy});
    builtins.emplace("π", std::pair{ConstEvalValue(qc::PI), floatTy});
    builtins.emplace("tau", std::pair{ConstEvalValue(qc::TAU), floatTy});
    builtins.emplace("τ", std::pair{ConstEvalValue(qc::TAU), floatTy});
    builtins.emplace("euler", std::pair{ConstEvalValue(qc::E), floatTy});
    builtins.emplace("ℇ", std::pair{ConstEvalValue(qc::E), floatTy});

    return builtins;
  }

  static void
  translateGateOperand(const std::shared_ptr<GateOperand>& gateOperand,
                       std::vector<qc::Qubit>& qubits,
                       const qc::QuantumRegisterMap& qregs,
                       const std::shared_ptr<DebugInfo>& debugInfo) {
    translateGateOperand(gateOperand->identifier, gateOperand->expression,
                         qubits, qregs, debugInfo);
  }

  static void
  translateGateOperand(const std::string& gateIdentifier,
                       const std::shared_ptr<Expression>& indexExpr,
                       std::vector<qc::Qubit>& qubits,
                       const qc::QuantumRegisterMap& qregs,
                       const std::shared_ptr<DebugInfo>& debugInfo) {
    auto qubitIter = qregs.find(gateIdentifier);
    if (qubitIter == qregs.end()) {
      error("Usage of unknown quantum register.", debugInfo);
    }
    auto qubit = qubitIter->second;

    if (indexExpr != nullptr) {
      auto result = evaluatePositiveConstant(indexExpr, debugInfo);

      if (result >= qubit.second) {
        error("Index expression must be smaller than the width of the "
              "quantum register.",
              debugInfo);
      }
      qubit.first += result;
      qubit.second = 1;
    }

    for (uint64_t i = 0; i < qubit.second; ++i) {
      qubits.emplace_back(qubit.first + i);
    }
  }

  void translateBitOperand(const std::string& bitIdentifier,
                           const std::shared_ptr<Expression>& indexExpr,
                           std::vector<qc::Bit>& bits,
                           const std::shared_ptr<DebugInfo>& debugInfo) {
    auto iter = qc->getCregs().find(bitIdentifier);
    if (iter == qc->getCregs().end()) {
      error("Usage of unknown classical register.", debugInfo);
    }
    auto creg = iter->second;

    if (indexExpr != nullptr) {
      auto index = evaluatePositiveConstant(indexExpr, debugInfo);
      if (index >= creg.second) {
        error("Index expression must be smaller than the width of the "
              "classical register.",
              debugInfo);
      }

      creg.first += index;
      creg.second = 1;
    }

    for (uint64_t i = 0; i < creg.second; ++i) {
      bits.emplace_back(creg.first + i);
    }
  }

  static uint64_t
  evaluatePositiveConstant(const std::shared_ptr<Expression>& expr,
                           const std::shared_ptr<DebugInfo>& debugInfo,
                           uint64_t defaultValue = 0) {
    if (expr == nullptr) {
      return defaultValue;
    }

    auto constInt = std::dynamic_pointer_cast<Constant>(expr);
    if (!constInt) {
      error("Expected a constant integer expression.", debugInfo);
    }

    return constInt->getUInt();
  }

public:
  explicit OpenQasm3Parser(qc::QuantumComputation* quantumComputation)
      : typeCheckPass(&constEvalPass), qc(quantumComputation) {
    for (auto [identifier, builtin] : initializeBuiltins()) {
      constEvalPass.addConst(identifier, builtin.first);
      typeCheckPass.addBuiltin(identifier, builtin.second);
    }
  }

  ~OpenQasm3Parser() override = default;

  bool visitProgram(std::vector<std::shared_ptr<Statement>>& program) {
    // TODO: in the future, don't exit early, but collect all errors
    // To do this, we need to insert make sure that erroneous declarations
    // actually insert a dummy entry; also, we need to synchronize to the next
    // semicolon, to make sure we don't do some weird stuff and report false
    // errors.
    for (auto& statement : program) {
      try {
        constEvalPass.processStatement(*statement);
        typeCheckPass.processStatement(*statement);
        statement->accept(this);
      } catch (CompilerError& e) {
        std::cerr << e.toString() << '\n';
        return false;
      }
    }

    return true;
  }

  void visitVersionDeclaration(
      std::shared_ptr<VersionDeclaration> versionDeclaration) override {
    if (versionDeclaration->version < 3) {
      qc->updateMaxControls(2);
    }
  }

  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override {
    auto identifier = declarationStatement->identifier;
    if (declarations.find(identifier).has_value()) {
      // TODO: show the location of the previous declaration
      error("Identifier '" + identifier + "' already declared.",
            declarationStatement->debugInfo);
    }

    std::shared_ptr<ResolvedType> const ty =
        std::get<1>(declarationStatement->type);

    if (auto sizedTy = std::dynamic_pointer_cast<SizedType>(ty)) {
      auto designator = sizedTy->getDesignator();
      switch (sizedTy->type) {
      case Qubit:
        qc->addQubitRegister(designator, identifier);
        break;
      case Bit:
      case Int:
      case Uint:
        qc->addClassicalRegister(designator, identifier);
        break;
      case Float:
        // not adding to qc
        break;
      case Angle:
        error("Angle type is currently not supported.",
              declarationStatement->debugInfo);
      }
    } else {
      error("Only sized types are supported.", declarationStatement->debugInfo);
    }
    declarations.emplace(identifier, declarationStatement);

    if (declarationStatement->isConst) {
      // nothing to do
      return;
    }
    if (declarationStatement->expression == nullptr) {
      // value is uninitialized
      return;
    }
    if (auto measureExpression = std::dynamic_pointer_cast<MeasureExpression>(
            declarationStatement->expression->expression)) {
      if (declarationStatement->isConst) {
        error("Cannot initialize a const register with a measure statement.",
              declarationStatement->debugInfo);
      }
      visitMeasureAssignment(identifier, nullptr, measureExpression,
                             declarationStatement->debugInfo);
      return;
    }

    error("Only measure statements are supported for initialization.",
          declarationStatement->debugInfo);
  }

  void visitAssignmentStatement(
      std::shared_ptr<AssignmentStatement> assignmentStatement) override {
    auto identifier = assignmentStatement->identifier->identifier;
    auto declaration = declarations.find(identifier);
    if (!declaration.has_value()) {
      error("Usage of unknown identifier '" + identifier + "'.",
            assignmentStatement->debugInfo);
    }

    if (declaration->get()->isConst) {
      error("Assignment to constant identifier '" + identifier + "'.",
            assignmentStatement->debugInfo);
    }

    if (auto measureExpression = std::dynamic_pointer_cast<MeasureExpression>(
            assignmentStatement->expression->expression)) {
      visitMeasureAssignment(identifier, assignmentStatement->indexExpression,
                             measureExpression, assignmentStatement->debugInfo);
      return;
    }

    // In the future, handle classical computation.
    error("Classical computation not supported.",
          assignmentStatement->debugInfo);
  }

  void
  visitInitialLayout(std::shared_ptr<InitialLayout> initialLayout) override {
    if (!qc->initialLayout.empty()) {
      error("Multiple initial layout specifications found.",
            initialLayout->debugInfo);
    }
    qc->initialLayout = initialLayout->permutation;
  }

  void visitOutputPermutation(
      std::shared_ptr<OutputPermutation> outputPermutation) override {
    if (!qc->outputPermutation.empty()) {
      error("Multiple output permutation specifications found.",
            outputPermutation->debugInfo);
    }
    qc->outputPermutation = outputPermutation->permutation;
  }

  void
  visitGateStatement(std::shared_ptr<GateDeclaration> gateStatement) override {
    auto identifier = gateStatement->identifier;
    if (gateStatement->isOpaque) {
      if (gates.find(identifier) == gates.end()) {
        // only builtin gates may be declared as opaque.
        error("Unsupported opaque gate '" + identifier + "'.",
              gateStatement->debugInfo);
      }

      return;
    }

    if (gates.find(identifier) != gates.end()) {
      // TODO: print location of previous declaration
      error("Gate '" + identifier + "' already declared.",
            gateStatement->debugInfo);
    }

    auto parameters = gateStatement->parameters;
    auto qubits = gateStatement->qubits;

    // first we check that all parameters and qubits are unique
    std::vector<std::string> parameterIdentifiers{};
    for (const auto& parameter : parameters->identifiers) {
      if (std::find(parameterIdentifiers.begin(), parameterIdentifiers.end(),
                    parameter->identifier) != parameterIdentifiers.end()) {
        error("Parameter '" + parameter->identifier + "' already declared.",
              gateStatement->debugInfo);
      }
      parameterIdentifiers.emplace_back(parameter->identifier);
    }
    std::vector<std::string> qubitIdentifiers{};
    for (const auto& qubit : qubits->identifiers) {
      if (std::find(qubitIdentifiers.begin(), qubitIdentifiers.end(),
                    qubit->identifier) != qubitIdentifiers.end()) {
        error("Qubit '" + qubit->identifier + "' already declared.",
              gateStatement->debugInfo);
      }
      qubitIdentifiers.emplace_back(qubit->identifier);
    }

    auto compoundGate = std::make_shared<CompoundGate>(CompoundGate(
        parameterIdentifiers, qubitIdentifiers, gateStatement->statements));

    gates.emplace(identifier, compoundGate);
  }

  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override {
    if (gates.find(gateCallStatement->identifier) == gates.end()) {
      error("Gate '" + gateCallStatement->identifier + "' not declared.",
            gateCallStatement->debugInfo);
    }

    auto qregs = qc->getQregs();

    auto op = evaluateGateCall(gateCallStatement, gateCallStatement->identifier,
                               gateCallStatement->arguments,
                               gateCallStatement->operands, qregs);
    if (op != nullptr) {
      qc->emplace_back(std::move(op));
    }
  }

  std::unique_ptr<qc::Operation>
  evaluateGateCall(const std::shared_ptr<GateCallStatement>& gateCallStatement,
                   const std::string& identifier,
                   const std::vector<std::shared_ptr<Expression>>& parameters,
                   std::vector<std::shared_ptr<GateOperand>> targets,
                   qc::QuantumRegisterMap& qregs) {
    auto iter = gates.find(identifier);
    if (iter == gates.end()) {
      error("Usage of unknown gate '" + identifier + "'.",
            gateCallStatement->debugInfo);
    }
    auto gate = iter->second;

    if (gate->getNParameters() != parameters.size()) {
      error("Gate '" + identifier + "' takes " +
                std::to_string(gate->getNParameters()) + " parameters, but " +
                std::to_string(parameters.size()) + " were supplied.",
            gateCallStatement->debugInfo);
    }

    // here we count the number of controls
    std::vector<std::pair<std::shared_ptr<GateOperand>, bool>> controls;
    // since standard gates may define a number of control targets, we first
    // need to handle those
    size_t nControls{gate->getNControls()};
    if (targets.size() < nControls) {
      error("Gate '" + identifier + "' takes " + std::to_string(nControls) +
                " controls, but only " + std::to_string(targets.size()) +
                " were supplied.",
            gateCallStatement->debugInfo);
    }

    for (size_t i = 0; i < nControls; ++i) {
      controls.emplace_back(targets[i], true);
    }

    bool invertOperation = false;
    for (const auto& modifier : gateCallStatement->modifiers) {
      if (auto ctrlModifier =
              std::dynamic_pointer_cast<CtrlGateModifier>(modifier);
          ctrlModifier != nullptr) {
        size_t const n = evaluatePositiveConstant(ctrlModifier->expression,
                                                  gateCallStatement->debugInfo,
                                                  /*defaultValue=*/1);
        if (targets.size() < n + nControls) {
          error("Gate '" + identifier + "' takes " +
                    std::to_string(n + nControls) + " controls, but only " +
                    std::to_string(targets.size()) + " were supplied.",
                gateCallStatement->debugInfo);
        }

        for (size_t i = 0; i < n; ++i) {
          controls.emplace_back(targets[nControls + i], ctrlModifier->ctrlType);
        }
        nControls += n;
      } else if (auto invModifier =
                     std::dynamic_pointer_cast<InvGateModifier>(modifier);
                 invModifier != nullptr) {
        // if we have an even number of inv modifiers, they cancel each other
        // out
        invertOperation = !invertOperation;
      } else {
        error("Only ctrl/negctrl/inv modifiers are supported.",
              gateCallStatement->debugInfo);
      }
    }
    targets.erase(targets.begin(),
                  targets.begin() + static_cast<int64_t>(nControls));

    if (gate->getNTargets() != targets.size()) {
      error("Gate '" + identifier + "' takes " +
                std::to_string(gate->getNTargets()) + " targets, but " +
                std::to_string(targets.size()) + " were supplied.",
            gateCallStatement->debugInfo);
    }

    // now evaluate all arguments; we only support const arguments.
    std::vector<qc::fp> evaluatedParameters;
    for (const auto& param : parameters) {
      auto result = constEvalPass.visit(param);
      if (!result.has_value()) {
        error("Only const expressions are supported, but found '" +
                  param->getName() + "'.",
              gateCallStatement->debugInfo);
      }

      evaluatedParameters.emplace_back(result->toExpr()->asFP());
    }

    size_t broadcastingWidth{1};
    qc::Targets targetBits{};
    std::vector<size_t> targetBroadcastingIndices{};
    size_t i{0};
    for (const auto& target : targets) {
      qc::Targets t{};
      translateGateOperand(target, t, qregs, gateCallStatement->debugInfo);

      targetBits.emplace_back(t[0]);

      if (t.size() > 1) {
        if (broadcastingWidth != 1 && t.size() != broadcastingWidth) {
          error("When broadcasting, all registers must be of the same width.",
                gateCallStatement->debugInfo);
        }
        broadcastingWidth = t.size();

        targetBroadcastingIndices.emplace_back(i);
      }

      i++;
    }

    std::vector<qc::Control> controlBits{};
    std::vector<size_t> controlBroadcastingIndices{};
    i = 0;
    for (const auto& [control, type] : controls) {
      qc::Targets c{};
      translateGateOperand(control, c, qregs, gateCallStatement->debugInfo);

      controlBits.emplace_back(c[0], type ? qc::Control::Type::Pos
                                          : qc::Control::Type::Neg);

      if (c.size() > 1) {
        if (broadcastingWidth != 1 && c.size() != broadcastingWidth) {
          error("When broadcasting, all registers must be of the same width.",
                gateCallStatement->debugInfo);
        }
        broadcastingWidth = c.size();

        controlBroadcastingIndices.emplace_back(i);
      }

      i++;
    }

    if (broadcastingWidth == 1) {
      return applyQuantumOperation(gate, targetBits, controlBits,
                                   evaluatedParameters, invertOperation,
                                   gateCallStatement->debugInfo);
    }

    // if we are broadcasting, we need to create a compound operation
    auto op = std::make_unique<qc::CompoundOperation>(qc->getNqubits());
    for (size_t j = 0; j < broadcastingWidth; ++j) {
      // first we apply the operation
      auto nestedOp = applyQuantumOperation(
          gate, targetBits, controlBits, evaluatedParameters, invertOperation,
          gateCallStatement->debugInfo);
      if (nestedOp == nullptr) {
        return nullptr;
      }
      op->getOps().emplace_back(std::move(nestedOp));

      // after applying the operation, we update the broadcast bits
      for (auto index : targetBroadcastingIndices) {
        targetBits[index] = qc::Qubit{targetBits[index] + 1};
      }
      for (auto index : controlBroadcastingIndices) {
        controlBits[index].qubit = qc::Qubit{controlBits[index].qubit + 1};
      }
    }
    return op;
  }

  std::unique_ptr<qc::Operation>
  applyQuantumOperation(std::shared_ptr<Gate> gate, qc::Targets targetBits,
                        std::vector<qc::Control> controlBits,
                        std::vector<qc::fp> evaluatedParameters,
                        bool invertOperation,
                        std::shared_ptr<DebugInfo> debugInfo) {
    if (auto* standardGate = dynamic_cast<StandardGate*>(gate.get())) {
      auto op = std::make_unique<qc::StandardOperation>(
          qc->getNqubits(), qc::Controls{}, targetBits, standardGate->info.type,
          evaluatedParameters);
      if (invertOperation) {
        op->invert();
      }
      op->setControls(qc::Controls{controlBits.begin(), controlBits.end()});
      return op;
    }
    if (auto* compoundGate = dynamic_cast<CompoundGate*>(gate.get())) {
      constEvalPass.pushEnv();

      for (size_t i = 0; i < compoundGate->parameterNames.size(); ++i) {
        constEvalPass.addConst(compoundGate->parameterNames[i],
                               evaluatedParameters[i]);
      }

      auto nestedQubits = qc::QuantumRegisterMap{};
      size_t index = 0;
      for (const auto& qubitIdentifier : compoundGate->targetNames) {
        auto qubit = std::pair{targetBits[index], 1};

        nestedQubits.emplace(qubitIdentifier, qubit);
        index++;
      }

      auto op = std::make_unique<qc::CompoundOperation>(qc->getNqubits());
      for (const auto& nestedGate : compoundGate->body) {
        for (const auto& operand : nestedGate->operands) {
          // OpenQASM 3.0 doesn't support indexing of gate arguments.
          if (operand->expression != nullptr &&
              std::find(compoundGate->targetNames.begin(),
                        compoundGate->targetNames.end(), operand->identifier) !=
                  compoundGate->targetNames.end()) {
            error("Gate arguments cannot be indexed within gate body.",
                  debugInfo);
          }
        }

        auto nestedOp = evaluateGateCall(nestedGate, nestedGate->identifier,
                                         nestedGate->arguments,
                                         nestedGate->operands, nestedQubits);
        if (nestedOp == nullptr) {
          return nullptr;
        }
        op->getOps().emplace_back(std::move(nestedOp));
      }
      op->setControls(qc::Controls{controlBits.begin(), controlBits.end()});
      if (invertOperation) {
        op->invert();
      }

      constEvalPass.popEnv();

      if (op->getOps().size() == 1) {
        return std::move(op->getOps()[0]);
      }

      return op;
    }

    error("Unknown gate type.", debugInfo);
  }

  void visitMeasureAssignment(
      const std::string& identifier,
      const std::shared_ptr<Expression>& indexExpression,
      const std::shared_ptr<MeasureExpression>& measureExpression,
      const std::shared_ptr<DebugInfo>& debugInfo) {
    auto decl = declarations.find(identifier);
    if (!decl.has_value()) {
      error("Usage of unknown identifier '" + identifier + "'.", debugInfo);
    }

    if (!std::get<1>(decl.value()->type)->isBit()) {
      error("Measure expression can only be assigned to a bit register.",
            debugInfo);
    }

    std::vector<qc::Qubit> qubits{};
    std::vector<qc::Bit> bits{};
    translateGateOperand(measureExpression->gate, qubits, qc->getQregs(),
                         debugInfo);
    translateBitOperand(identifier, indexExpression, bits, debugInfo);

    if (qubits.size() != bits.size()) {
      error("Classical and quantum register must have the same width in "
            "measure statement. Classical register '" +
                identifier + "' has " + std::to_string(bits.size()) +
                " bits, but quantum register '" +
                measureExpression->gate->identifier + "' has " +
                std::to_string(qubits.size()) + " qubits.",
            debugInfo);
    }

    auto op = std::make_unique<qc::NonUnitaryOperation>(qc->getNqubits(),
                                                        qubits, bits);
    qc->emplace_back(std::move(op));
  }

  void visitBarrierStatement(
      std::shared_ptr<BarrierStatement> barrierStatement) override {
    std::vector<qc::Qubit> qubits{};
    for (const auto& gate : barrierStatement->gates) {
      translateGateOperand(gate, qubits, qc->getQregs(),
                           barrierStatement->debugInfo);
    }

    auto op = std::make_unique<qc::NonUnitaryOperation>(qc->getNqubits(),
                                                        qubits, qc::Barrier);
    qc->emplace_back(std::move(op));
  }

  void
  visitResetStatement(std::shared_ptr<ResetStatement> resetStatement) override {
    std::vector<qc::Qubit> qubits{};
    translateGateOperand(resetStatement->gate, qubits, qc->getQregs(),
                         resetStatement->debugInfo);
    auto op = std::make_unique<qc::NonUnitaryOperation>(qc->getNqubits(),
                                                        qubits, qc::Reset);
    qc->emplace_back(std::move(op));
  }

  void visitIfStatement(std::shared_ptr<IfStatement> ifStatement) override {
    // TODO: for now we only support statements comparing a classical bit reg
    // to a constant.
    auto condition =
        std::dynamic_pointer_cast<BinaryExpression>(ifStatement->condition);
    if (condition == nullptr) {
      error("Condition not supported for if statement.",
            ifStatement->debugInfo);
    }

    qc::ComparisonKind comparisonKind = qc::ComparisonKind::Eq;
    switch (condition->op) {
    case qasm3::BinaryExpression::Op::LessThan:
      comparisonKind = qc::ComparisonKind::Lt;
      break;
    case qasm3::BinaryExpression::Op::LessThanOrEqual:
      comparisonKind = qc::ComparisonKind::Leq;
      break;
    case qasm3::BinaryExpression::Op::GreaterThan:
      comparisonKind = qc::ComparisonKind::Gt;
      break;
    case qasm3::BinaryExpression::Op::GreaterThanOrEqual:
      comparisonKind = qc::ComparisonKind::Geq;
      break;
    case qasm3::BinaryExpression::Op::Equal:
      comparisonKind = qc::ComparisonKind::Eq;
      break;
    case qasm3::BinaryExpression::Op::NotEqual:
      comparisonKind = qc::ComparisonKind::Neq;
      break;
    default:
      error("Unsupported comparison operator.", ifStatement->debugInfo);
    }

    auto lhs = std::dynamic_pointer_cast<IdentifierExpression>(condition->lhs);
    auto rhs = std::dynamic_pointer_cast<Constant>(condition->rhs);

    if (lhs == nullptr) {
      error("Only classical registers are supported in conditions.",
            ifStatement->debugInfo);
    }
    if (rhs == nullptr) {
      error("Can only compare to constants.", ifStatement->debugInfo);
    }

    auto creg = qc->getCregs().find(lhs->identifier);
    if (creg == qc->getCregs().end()) {
      error("Usage of unknown or invalid identifier '" + lhs->identifier +
                "' in condition.",
            ifStatement->debugInfo);
    }

    // translate statements in then block
    auto thenOps = std::make_unique<qc::CompoundOperation>(qc->getNqubits());
    for (const auto& statement : ifStatement->thenStatements) {
      auto gateCall = std::dynamic_pointer_cast<GateCallStatement>(statement);
      if (gateCall == nullptr) {
        error("Only gate calls are supported in if statements.",
              statement->debugInfo);
      }
      auto qregs = qc->getQregs();

      auto op =
          evaluateGateCall(gateCall, gateCall->identifier, gateCall->arguments,
                           gateCall->operands, qregs);

      thenOps->emplace_back(std::move(op));
    }

    if (!ifStatement->elseStatements.empty()) {
      error("Else statements are not supported yet.", ifStatement->debugInfo);
    }

    std::unique_ptr<qc::Operation> thenOp = std::move(thenOps);

    qc->emplace_back(std::make_unique<qc::ClassicControlledOperation>(
        thenOp, creg->second, rhs->getUInt(), comparisonKind));
  }
};

void qc::QuantumComputation::importOpenQASM3(std::istream& is) {
  using namespace qasm3;

  Parser p(is);

  auto program = p.parseProgram();
  OpenQasm3Parser parser{this};
  if (!parser.visitProgram(program)) {
    throw std::runtime_error("Error importing OpenQASM.");
  }
}
