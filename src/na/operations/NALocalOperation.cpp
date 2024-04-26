#include "na/operations/NALocalOperation.hpp"

namespace na {
auto NALocalOperation::toString() const -> std::string {
  std::stringstream ss;
  ss << type;
  if (!params.empty()) {
    ss << "(";
    for (const auto& p : params) {
      ss << p << ", ";
    }
    ss.seekp(-2, std::ios_base::end);
    ss << ")";
  }
  ss << " at ";
  for (const auto& p : positions) {
    ss << *p << ", ";
  }
  ss.seekp(-2, std::ios_base::end);
  ss << ";\n";
  return ss.str();
}
} // namespace na
