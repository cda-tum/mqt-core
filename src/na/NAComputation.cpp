#include "na/NAComputation.hpp"

namespace na {
auto NAComputation::toString() const -> std::string {
  std::stringstream ss;
  ss << "init at ";
  for (const auto& p : initialPositions) {
    ss << "(" << p->x << ", " << p->y << ")"
       << ", ";
  }
  ss.seekp(-2, std::ios_base::end);
  ss << ";\n";
  for (const auto& op : operations) {
    ss << op->toString();
  }
  return ss.str();
}
} // namespace na
