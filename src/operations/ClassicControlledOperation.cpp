#include "operations/ClassicControlledOperation.hpp"

namespace qc {

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind) {
  switch (kind) {
  case ComparisonKind::Eq:
    os << "==";
    break;
  case ComparisonKind::Neq:
    os << "!=";
    break;
  case ComparisonKind::Lt:
    os << "<";
    break;
  case ComparisonKind::Leq:
    os << "<=";
    break;
  case ComparisonKind::Gt:
    os << ">";
    break;
  case ComparisonKind::Geq:
    os << ">=";
    break;
  }

  return os;
}
} // namespace qc
