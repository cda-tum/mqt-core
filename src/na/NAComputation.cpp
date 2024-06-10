#include "na/NAComputation.hpp"

#include <ios>
#include <sstream>
#include <string>

namespace na {
auto NAComputation::toString() const -> std::string {
  std::stringstream ss;
  ss << "init at ";
  for (const auto& p : initialPositions) {
    ss << *p << ", ";
  }
  if (ss.tellp() == 8) {
    ss.seekp(-1, std::ios_base::end);
  } else {
    ss.seekp(-2, std::ios_base::end);
  }
  ss << ";\n";
  for (const auto& op : operations) {
    ss << *op;
  }
  return ss.str();
}
} // namespace na
