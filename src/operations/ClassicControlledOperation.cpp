#include "operations/ClassicControlledOperation.hpp"

namespace qc {

std::string toString(const ComparisonKind& kind) {
  switch (kind) {
  case ComparisonKind::Eq:
    return "==";
  case ComparisonKind::Neq:
    return "!=";
  case ComparisonKind::Lt:
    return "<";
  case ComparisonKind::Leq:
    return "<=";
  case ComparisonKind::Gt:
    return ">";
  case ComparisonKind::Geq:
    return ">=";
  default:
#if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(0);
#else
#endif
  }
}

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind) {
  os << toString(kind);
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
