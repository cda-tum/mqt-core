#pragma once

#include <memory>
#include <string>
namespace na {
class NAOperation {
public:
  [[nodiscard]] virtual auto isShuttlingOperation() const -> bool { return false; }
  [[nodiscard]] virtual auto isLocalOperation() const -> bool { return false; }
  [[nodiscard]] virtual auto isGlobalOperation() const -> bool { return false; }
  [[nodiscard]] virtual auto toString() const -> std::string = 0;
  virtual ~NAOperation() = default;
  [[nodiscard]] virtual auto clone() const -> std::unique_ptr<NAOperation> = 0;
};
} // namespace na
