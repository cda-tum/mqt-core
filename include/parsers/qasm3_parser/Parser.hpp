/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#pragma once

#include "Definitions.hpp"
#include "Scanner.hpp"
#include "Statement.hpp"
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

namespace qasm3 {
class Parser {
private:
  struct ScannerState {
  private:
    std::unique_ptr<std::istream> is;

  public:
    Token t{0, 0};
    Token next{0, 0};
    std::unique_ptr<Scanner> scanner;
    std::optional<std::string> filename;

    bool scan() {
      t = next;
      next = scanner->next();

      return t.kind != Token::Kind::Eof;
    }

    explicit ScannerState(std::istream& in,
                          std::optional<std::string> filename = std::nullopt)
        : scanner(std::make_unique<Scanner>(in)),
          filename(std::move(filename)) {
      scan();
    }

    explicit ScannerState(std::unique_ptr<std::istream> in,
                          std::optional<std::string> filename = std::nullopt)
        : is(std::move(in)), scanner(std::make_unique<Scanner>(*is)),
          filename(std::move(filename)) {
      scan();
    }
  };

  std::stack<ScannerState> scanner{};
  std::shared_ptr<DebugInfo> includeDebugInfo{nullptr};

  [[noreturn]] static void error(const Token& token, const std::string& msg) {
    std::cerr << "Error at line " << token.line << ", column " << token.col
              << ": " << msg << '\n';
    throw std::runtime_error("Parser error");
  }

  [[nodiscard]] inline Token current() const {
    if (scanner.empty()) {
      throw std::runtime_error("No scanner available");
    }
    return scanner.top().t;
  }

  [[nodiscard]] inline Token peek() const {
    if (scanner.empty()) {
      throw std::runtime_error("No scanner available");
    }
    return scanner.top().next;
  }

  Token expect(const Token::Kind& expected) {
    if (current().kind != expected) {
      error(current(), "Expected '" + Token::kindToString(expected) +
                           "', got '" + Token::kindToString(current().kind) +
                           "'");
    }

    Token const token = current();
    scan();
    return token;
  }

public:
  explicit Parser(std::istream& is) {
    scanner.emplace(is);
    scan();
  }

  virtual ~Parser() = default;

  std::shared_ptr<VersionDeclaration> parseVersionDeclaration();

  std::vector<std::shared_ptr<Statement>> parseProgram();

  std::shared_ptr<Statement> parseStatement();

  void parseInclude();

  std::shared_ptr<AssignmentStatement> parseAssignmentStatement();

  std::shared_ptr<AssignmentStatement> parseMeasureStatement();

  std::shared_ptr<ResetStatement> parseResetStatement();

  std::shared_ptr<BarrierStatement> parseBarrierStatement();

  std::shared_ptr<Statement> parseDeclaration(bool isConst);

  std::shared_ptr<GateDeclaration> parseGateDefinition();

  std::shared_ptr<GateCallStatement> parseGateCallStatement();

  std::shared_ptr<GateModifier> parseGateModifier();

  std::shared_ptr<GateOperand> parseGateOperand();

  std::shared_ptr<DeclarationExpression> parseDeclarationExpression();

  std::shared_ptr<MeasureExpression> parseMeasureExpression();

  std::shared_ptr<Expression> exponentiation();

  std::shared_ptr<Expression> factor();

  std::shared_ptr<Expression> term();

  std::shared_ptr<Expression> parseExpression();

  std::shared_ptr<IdentifierList> parseIdentifierList();

  std::pair<std::shared_ptr<TypeExpr>, bool> parseType();

  std::shared_ptr<Expression> parseTypeDesignator();

  void scan();

  std::shared_ptr<DebugInfo> makeDebugInfo(Token const& begin,
                                           Token const& end) {
    return std::make_shared<DebugInfo>(
        begin.line, begin.col, end.endLine, end.endCol,
        scanner.top().filename.value_or("<input>"), includeDebugInfo);
  }

  std::shared_ptr<DebugInfo> makeDebugInfo(Token const& token) {
    return std::make_shared<DebugInfo>(
        token.line, token.col, token.line, token.col,
        scanner.top().filename.value_or("<input>"), includeDebugInfo);
  }

  [[nodiscard]] bool isAtEnd() const {
    return current().kind == Token::Kind::Eof;
  }
};

} // namespace qasm3
