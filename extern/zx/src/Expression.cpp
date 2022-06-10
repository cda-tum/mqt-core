#include "Expression.hpp"

namespace zx {
void Term::add_coeff(Rational r) { coeff += r; }

bool Expression::is_zero() const { return terms.empty() && constant.is_zero(); }
bool Expression::is_constant() const { return terms.empty(); }

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
        insert_pos->get_coeff() += t->get_coeff();
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

Expression &Expression::operator+=(const Rational &rhs) {
  constant += rhs;
  return *this;
}

Expression &Expression::operator-=(const Expression &rhs) {
  return *this += -rhs;
}

Expression &Expression::operator-=(const Term &rhs) { return *this += -rhs; }

Expression &Expression::operator-=(const Rational &rhs) {
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
    if (t->get_coeff().is_zero()) {
      t = terms.erase(t);
    } else {
      t = next;
    }
  }
}
} // namespace zx
