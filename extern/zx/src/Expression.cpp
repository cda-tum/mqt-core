#include "Expression.hpp"
#include <cmath>

namespace zx {
void Term::add_coeff(fp r) { coeff += r; }
Term &Term::operator*=(fp rhs) {
  coeff *= rhs;
  return *this;
}

Term &Term::operator/=(fp rhs) {
  coeff /= rhs;
  return *this;
}

bool Expression::is_zero() const { return terms.empty() && constant.is_zero(); }
bool Expression::is_constant() const { return terms.empty(); }
bool Expression::is_pauli() const {
  return is_constant() && constant.is_integer();
}
bool Expression::is_clifford() const {
  return is_constant() && (constant.is_integer() || constant.denom == 2);
}
bool Expression::is_proper_clifford() const {
  return is_constant() && constant.denom == 2;
}

Expression &Expression::operator+=(const Expression &rhs) {
  if (this->is_zero()) {
    *this = rhs;
    return *this;
  }

  if (rhs.is_zero())
    return *this;

  auto t = rhs.begin();

  /*Small optimisation. Most of the time the first monomials will cancel*/
  // if (m->t == n->t && m->coeff == -n->coeff) {
  //   monomials.erase(m.base());
  //   n++;
  // }

  while (t != rhs.end()) {
    auto insert_pos = std::lower_bound(
        terms.begin(), terms.end(), *t, [&](const Term &lhs, const Term &rhs) {
          return lhs.get_var().id < rhs.get_var().id;
        });
    if (insert_pos != terms.end() && insert_pos->get_var() == t->get_var()) {
      if (insert_pos->get_coeff() == -t->get_coeff()) {
        terms.erase(insert_pos);
      } else {
        insert_pos->add_coeff(t->get_coeff());
      }
    } else {
      terms.insert(insert_pos, *t);
    }
    ++t;
  }
  constant += rhs.constant;
  return *this;
}

Expression &Expression::operator+=(const Term &rhs) {
  return *this += Expression(rhs);
}

Expression &Expression::operator+=(const PiRational &rhs) {
  constant += rhs;
  return *this;
}

Expression &Expression::operator-=(const Expression &rhs) {
  return *this += -rhs;
}

Expression &Expression::operator-=(const Term &rhs) { return *this += -rhs; }

Expression &Expression::operator-=(const PiRational &rhs) {
  return *this += -rhs;
}

Expression Expression::operator-() const {
  Expression e;
  e.terms.reserve(terms.size());
  for (auto &t : terms)
    e.terms.push_back(-t);
  e.constant = -constant;
  return e;
}

void Expression::sort_terms() {
  std::sort(terms.begin(), terms.end(), [&](const Term &lhs, const Term &rhs) {
    return lhs.get_var().id < rhs.get_var().id;
  });
}

void Expression::aggregate_equal_terms() {
  for (auto t = terms.begin(); t != terms.end();) {
    auto next = std::next(t);
    while (next != terms.end() && t->get_var() == next->get_var()) {
      t->add_coeff(next->get_coeff());
      next = terms.erase(next);
    }
    if (t->has_zero_coeff()) {
      t = terms.erase(t);
    } else {
      t = next;
    }
  }
}

  bool operator==(const Expression& lhs, const Expression& rhs) {
    if(lhs.num_terms() != rhs.num_terms() || lhs.get_constant() != rhs.get_constant())
      return false;

    for(size_t i = 0; i < lhs.num_terms(); ++i) {
      if(std::abs(lhs[i].get_coeff() - rhs[i].get_coeff()) >= TOLERANCE)
        return false;
    }
    return true;
  }
} // namespace zx
