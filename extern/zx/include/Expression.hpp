#ifndef ZX_INCLUDE_EXPRESSION_HPP_
#define ZX_INCLUDE_EXPRESSION_HPP_

#include "Rational.hpp"
#include "dd/Definitions.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace zx {
constexpr double TOLERANCE = 1e-13;
struct Variable {
  Variable(int32_t id, std::string name) : id(id), name(name){};
  int32_t id;
  std::string name;
};

inline bool operator==(const Variable &lhs, const Variable &rhs) {
  return lhs.id == rhs.id;
}
class Term {
public:
  [[nodiscard]] Variable get_var() const { return var; }
  [[nodiscard]] dd::fp get_coeff() const { return coeff; }
  [[nodiscard]] bool has_zero_coeff() const {
    return std::abs(coeff) < TOLERANCE;
  }

  void add_coeff(dd::fp r);
  Term(dd::fp coeff, Variable var) : coeff(coeff), var(var){};
  Term(Variable var) : coeff(1), var(var){};

  Term operator-() const { return Term(-coeff, var); }
  Term &operator*=(dd::fp rhs);
  Term &operator/=(dd::fp rhs);

private:
  dd::fp coeff;
  Variable var;
};

inline Term operator*(Term lhs, dd::fp rhs) {
  lhs *= rhs;
  return lhs;
}
inline Term operator/(Term lhs, dd::fp rhs) {
  lhs /= rhs;
  return lhs;
}
inline Term operator*(dd::fp lhs, const Term &rhs) { return rhs * lhs; }

inline Term operator/(dd::fp lhs, const Term &rhs) { return rhs / lhs; }

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
    terms.emplace_back(Term(1, v));
    (terms.emplace_back(std::forward<Args>(ms)), ...);
    sort_terms();
    aggregate_equal_terms();
  }

  Expression() : constant(PyRational(0, 1)){};
  Expression(PyRational r) : constant(r){};

  iterator begin() { return terms.begin(); }
  iterator end() { return terms.end(); }
  const_iterator begin() const { return terms.cbegin(); }
  const_iterator end() const { return terms.cend(); }
  const_iterator cbegin() const { return terms.cbegin(); }
  const_iterator cend() const { return terms.cend(); }

  [[nodiscard]] bool is_zero() const;
  [[nodiscard]] bool is_constant() const;
  [[nodiscard]] bool is_pauli() const;
  [[nodiscard]] bool is_clifford() const;
  [[nodiscard]] bool is_proper_clifford() const;

  Expression &operator+=(const Expression &rhs);
  Expression &operator+=(const Term &rhs);
  Expression &operator+=(const PyRational &rhs);

  Expression &operator-=(const Expression &rhs);
  Expression &operator-=(const Term &rhs);
  Expression &operator-=(const PyRational &rhs);

  [[nodiscard]] Expression operator-() const;

  [[nodiscard]] const Term &operator[](int i) const { return terms[i]; }
  [[nodiscard]] PyRational get_constant() const { return constant; }
  [[nodiscard]] auto num_terms() const { return terms.size(); }

private:
  std::vector<Term> terms;
  PyRational constant;

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
inline Expression operator+(Expression lhs, const PyRational &rhs) {
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
inline Expression operator-(Expression lhs, const PyRational &rhs) {
  lhs -= rhs;
  return lhs;
}

  bool operator==(const Expression& lhs, const Expression& rhs);
} // namespace zx

inline std::ostream &operator<<(std::ostream &os, const zx::Variable &rhs) {
  os << rhs.name;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const zx::Term &rhs) {
  os << rhs.get_coeff() << "*" << rhs.get_var();
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const zx::Expression &rhs) {
  for (auto &t : rhs) {
    os << t << " + ";
  }
  os << rhs.get_constant();
  return os;
}
#endif /* ZX_INCLUDE_EXPRESSION_HPP_ */
