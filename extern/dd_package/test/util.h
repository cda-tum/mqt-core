//
// Created by stefan on 04.12.19.
//

#ifndef DD_PACKAGE_UTIL_H
#define DD_PACKAGE_UTIL_H

#include "DDpackage.h"
#include <string>
#include <cassert>

// X gate matrix
constexpr dd::Matrix2x2 Xmat = {{{ 0, 0 }, { 1, 0 } }, {{ 1, 0 }, { 0, 0 } }};
// Hadamard gate matrix
constexpr dd::Matrix2x2 Hmat = {{{ dd::SQRT_2, 0 }, { dd::SQRT_2,  0 }},
                                {{ dd::SQRT_2, 0 }, { -dd::SQRT_2, 0 }}};
/**
 * Get a single element of the vector or matrix represented by the dd with root edge e
 * @param dd package where the dd lives
 * @param e edge pointing to the root node
 * @param elements string {0, 1, 2, 3}^n describing which outgoing edge should be followed
 *                 (for vectors 0 is the 0-successor and 2 is the 1-successor due to the shared representation)
 *                 If string is longer than required, the additional characters are ignored.
 * @return the complex value of the specified element
 */
inline dd::ComplexValue getValueByPath(dd::Package *dd, dd::Edge e, std::string elements) {
    if(dd::Package::isTerminal(e)) {
        return {dd::ComplexNumbers::val(e.w.r), dd::ComplexNumbers::val(e.w.i)};
    }

    dd::Complex c = dd->cn.getTempCachedComplex(1, 0);
    do {
        dd::ComplexNumbers::mul(c, c, e.w);
        int tmp = elements.at(dd->invVarOrder.at(e.p->v))-'0';
        assert(tmp >= 0 && tmp <= dd::NEDGE);
        e = e.p->e[tmp];
    } while(!dd::Package::isTerminal(e));
    dd::ComplexNumbers::mul(c, c, e.w);

    return {dd::ComplexNumbers::val(c.r), dd::ComplexNumbers::val(c.i)};
}

#endif //DD_PACKAGE_UTIL_H
