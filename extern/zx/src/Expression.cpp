#include "Expression.hpp"

#include <cmath>

namespace zx {
    void Term::addCoeff(fp r) {
        coeff += r;
    }
    Term& Term::operator*=(fp rhs) {
        coeff *= rhs;
        return *this;
    }

    Term& Term::operator/=(fp rhs) {
        coeff /= rhs;
        return *this;
    }

    bool Expression::isZero() const {
        return terms.empty() && constant.isZero();
    }
    bool Expression::isConstant() const {
        return terms.empty();
    }
    bool Expression::isPauli() const {
        return isConstant() && constant.isInteger();
    }
    bool Expression::isClifford() const {
        return isConstant() && (constant.isInteger() || constant.getDenom() == 2);
    }
    bool Expression::isProperClifford() const {
        return isConstant() && constant.getDenom() == 2;
    }

    Expression& Expression::operator+=(const Expression& rhs) {
        if (this->isZero()) {
            *this = rhs;
            return *this;
        }

        if (rhs.isZero())
            return *this;

        auto t = rhs.begin();

        /*Small optimisation. Most of the time the first monomials will cancel*/
        // if (m->t == n->t && m->coeff == -n->coeff) {
        //   monomials.erase(m.base());
        //   n++;
        // }

        while (t != rhs.end()) {
            auto insert_pos = std::lower_bound(
                    terms.begin(), terms.end(), *t, [&](const Term& lhs, const Term& rhs) {
                        return lhs.getVar().id < rhs.getVar().id;
                    });
            if (insert_pos != terms.end() && insert_pos->getVar() == t->getVar()) {
                if (insert_pos->getCoeff() == -t->getCoeff()) {
                    terms.erase(insert_pos);
                } else {
                    insert_pos->addCoeff(t->getCoeff());
                }
            } else {
                terms.insert(insert_pos, *t);
            }
            ++t;
        }
        constant += rhs.constant;
        return *this;
    }

    Expression& Expression::operator+=(const Term& rhs) {
        return *this += Expression(rhs);
    }

    Expression& Expression::operator+=(const PiRational& rhs) {
        constant += rhs;
        return *this;
    }

    Expression& Expression::operator-=(const Expression& rhs) {
        return *this += -rhs;
    }

    Expression& Expression::operator-=(const Term& rhs) {
        return *this += -rhs;
    }

    Expression& Expression::operator-=(const PiRational& rhs) {
        return *this += -rhs;
    }

    Expression Expression::operator-() const {
        Expression e;
        e.terms.reserve(terms.size());
        for (auto& t: terms)
            e.terms.push_back(-t);
        e.constant = -constant;
        return e;
    }

    void Expression::sortTerms() {
        std::sort(terms.begin(), terms.end(), [&](const Term& lhs, const Term& rhs) {
            return lhs.getVar().id < rhs.getVar().id;
        });
    }

    void Expression::aggregateEqualTerms() {
        for (auto t = terms.begin(); t != terms.end();) {
            auto next = std::next(t);
            while (next != terms.end() && t->getVar() == next->getVar()) {
                t->addCoeff(next->getCoeff());
                next = terms.erase(next);
            }
            if (t->hasZeroCoeff()) {
                t = terms.erase(t);
            } else {
                t = next;
            }
        }
    }

    bool operator==(const Expression& lhs, const Expression& rhs) {
        if (lhs.numTerms() != rhs.numTerms() || lhs.getConst() != rhs.getConst())
            return false;

        for (size_t i = 0; i < lhs.numTerms(); ++i) {
            if (std::abs(lhs[i].getCoeff() - rhs[i].getCoeff()) >= TOLERANCE)
                return false;
        }
        return true;
    }
} // namespace zx
