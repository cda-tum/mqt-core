#pragma once

#include "Statement.hpp"

namespace qasm3 {
class CompilerError final : std::exception {
public:
  std::string message{};
  std::shared_ptr<DebugInfo> debugInfo{};

  CompilerError(std::string msg, std::shared_ptr<DebugInfo> debug)
      : message(std::move(msg)), debugInfo(std::move(debug)) {}

  [[nodiscard]] std::string toString() const {
    std::stringstream ss{};
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
} // namespace qasm3

class ConstEvalError final : std::exception {
public:
  std::string message{};

  ConstEvalError(std::string msg) : message(std::move(msg)) {}

  [[nodiscard]] std::string toString() const { return message; }
};

class TypeCheckError final : std::exception {
public:
  std::string message{};

  TypeCheckError(std::string msg) : message(std::move(msg)) {}

  [[nodiscard]] std::string toString() const { return message; }
};
