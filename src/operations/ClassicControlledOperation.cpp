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

ComparisonKind getInvertedComparsionKind(const ComparisonKind kind) {
  switch (kind) {
  case Lt:
    return Geq;
  case Leq:
    return Gt;
  case Gt:
    return Leq;
  case Geq:
    return Lt;
  case Eq:
    return Neq;
  case Neq:
    return Eq;
  default:
#if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(0);
#else
#endif
  }
}
} // namespace qc
