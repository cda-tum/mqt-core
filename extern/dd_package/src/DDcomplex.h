/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */


#ifndef DDcomplex_H
#define DDcomplex_H

#include <cstdint>
#include <ostream>

namespace dd_package {
    constexpr unsigned int COMPLEX_NBUCKET = 32768;
    constexpr long double COMPLEX_TOLERANCE = 1e-10l;

    struct complex_table_entry {
        long double val;
        int ref;
        complex_table_entry *next;
    };

    struct complex {
        complex_table_entry *r;
        complex_table_entry *i;
    };

    struct complex_value {
        long double r, i;
    };

    extern complex COMPLEX_ZERO;
    extern complex COMPLEX_ONE;
    extern complex COMPLEX_M_ONE;

    extern complex_table_entry *Complex_table[COMPLEX_NBUCKET];
    extern complex_table_entry *Complex_Avail;
    extern complex_table_entry *CacheStart;
    extern complex_table_entry *ComplexCache_Avail;
    extern int ComplexCount;

    inline long double CVAL(const complex_table_entry *x) {
        if (((uintptr_t) x) & (uintptr_t) 1) {
            return -((complex_table_entry *) (((uintptr_t) x) ^ (uintptr_t) 1))->val;
        }
        return x->val;
    }

    inline bool operator==(const complex &lhs, const complex &rhs) {
        return lhs.r == rhs.r && lhs.i == rhs.i;
    }

    inline bool operator!=(const complex &lhs, const complex &rhs) {
        return lhs.r != rhs.r || lhs.i != rhs.i;
    }

    inline std::ostream& operator<<(std::ostream& os, const complex c)
    {
        os << CVAL(c.r) << std::showpos << CVAL(c.i) << "i" << std::noshowpos;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const complex_value c)
    {
        os << c.r << std::showpos << c.i << "i" << std::noshowpos;
        return os;
    }

    void initComplexTable(); // initialize the complex value table and complex operation tables to empty
    void complexInit();

    void complexIncRef(complex);

    void complexDecRef(complex);

    complex Clookup(const complex &); // lookup a complex value in the complex value table; if not found add it
    complex Cconjugate(complex);

    // basic operations on complex values
    // meanings are self-evident from the names
    // NOTE arguments are the indices to the values
    // in the complex value table not the values themselves

    complex Cnegative(complex);

    void Cadd(complex &, complex, complex);

    void Cmul(complex &, complex, complex);

    void Cdiv(complex &, complex, complex);

    bool Ceq(complex, complex);

    bool Ceq(complex_value, complex_value);

    long double CmagSquared(const complex &);

    void garbageCollectComplexTable();

    long double Ccos(long double fac, long double div);

    long double Csin(long double fac, long double div);

    complex_value Cmake(long double, long double); // make a complex value
}
#endif