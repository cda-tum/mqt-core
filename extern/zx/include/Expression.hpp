#ifndef ZX_INCLUDE_EXPRESSION_HPP_
#define ZX_INCLUDE_EXPRESSION_HPP_

#include "Definitions.hpp"
#include "Rational.hpp"

#include <string>
#include <vector>

namespace zx {
struct Variable {
  int32_t id;
  std::string name;
};

inline bool operator==(const Variable &lhs, const Variable &rhs) {
  return lhs.id == rhs.id;
}
class Term {
public:
  [[nodiscard]] Variable get_var() const { return var; }
  [[nodiscard]] Rational get_coeff() const { return coeff; }

  void add_coeff(Rational r);
  Term(Rational coeff, Variable var) : coeff(coeff), var(var){};

  Term operator-() const { return Term(-coeff, var); }

private:
  Rational coeff;
  Variable var;
};

class Expression {
public:
  using iterator = std::vector<Term>::iterator;
  using const_iterator = std::vector<Term>::const_iterator;

  template <typename... Args> Expression(Term t, Args... ms) {
    terms.emplace_back(t);
    (terms.emplace_back(std::forward<Args>(ms)), ...);
    sort_terms();
    aggregate_equal_terms();
  }

  template <typename... Args> Expression(Variable v, Args... ms) {
    terms.emplace_back(Term(Rational(1, 1), v));
    (terms.emplace_back(std::forward<Args>(ms)), ...);
    sort_terms();
    aggregate_equal_terms();
  }

  Expression() : constant(Rational(0, 1)){};
  Expression(Rational r) : constant(r){};

  iterator begin() { return terms.begin(); }
  iterator end() { return terms.end(); }
  const_iterator begin() const { return terms.cbegin(); }
  const_iterator end() const { return terms.cend(); }
  const_iterator cbegin() const { return terms.cbegin(); }
  const_iterator cend() const { return terms.cend(); }

  [[nodiscard]] bool is_zero() const;
  [[nodiscard]] bool is_constant() const;

  Expression &operator+=(const Expression &rhs);
  Expression &operator+=(const Term &rhs);
  Expression &operator+=(const Rational &rhs);

  Expression &operator-=(const Expression &rhs);
  Expression &operator-=(const Term &rhs);
  Expression &operator-=(const Rational &rhs);

  [[nodiscard]] Expression operator-() const;

private:
  std::vector<Term> terms;
  Rational constant;

  void sort_terms();
  void aggregate_equal_terms();
};

inline Expression operator+(Expression lhs, const Expression &rhs) {
  lhs += rhs;
  return lhs;
}
inline Expression operator+(Expression lhs, const Term &rhs) {
  lhs += rhs;
  return lhs;
}
inline Expression operator+(Expression lhs, const Rational &rhs) {
  lhs += rhs;
  return lhs;
}
inline Expression operator-(Expression lhs, const Expression &rhs) {
  lhs -= rhs;
  return lhs;
}
inline Expression operator-(Expression lhs, const Term &rhs) {
  lhs -= rhs;
  return lhs;
}
inline Expression operator-(Expression lhs, const Rational &rhs) {
  lhs -= rhs;
  return lhs;
}
} // namespace zx

#endif /* ZX_INCLUDE_EXPRESSION_HPP_ */
