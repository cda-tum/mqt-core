#include "na/operations/NAShuttlingOperation.hpp"

namespace na {
auto NAShuttlingOperation::toString() const -> std::string {
  std::stringstream ss;
  switch (type) {
  case LOAD:
    ss << "load";
    break;
  case MOVE:
    ss << "move";
    break;
  case STORE:
    ss << "store";
    break;
  }
  ss << " ";
  for (const auto& p : start) {
    ss << *p << ", ";
  }
  ss.seekp(-2, std::ios_base::end);
  ss << " to ";
  for (const auto& p : end) {
    ss << *p << ", ";
  }
  ss.seekp(-2, std::ios_base::end);
  ss << ";\n";
  return ss.str();
}
} // namespace na
