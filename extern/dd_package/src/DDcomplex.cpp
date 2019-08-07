/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */


#include "DDcomplex.h"
#include <cmath>
#include <iostream>

namespace dd_package {
    constexpr unsigned int COMPLEX_CHUNK_SIZE = 2000;
    constexpr unsigned int COMPLEX_INIT_SIZE = 300;
    constexpr long double PI = 3.14159265358979323846264338327950288419716939937510L;

    typedef struct complexChunk {
        complex_table_entry *entry;
        complexChunk *next;
    } complexChunk;

    complex COMPLEX_ZERO;
    complex COMPLEX_ONE;
    complex COMPLEX_M_ONE;

    complex_table_entry *Complex_table[COMPLEX_NBUCKET];
    complex_table_entry *Complex_Avail;
    complex_table_entry *CacheStart;
    complex_table_entry *ComplexCache_Avail;
    int ComplexCount;

    complexChunk *complex_chunks;

    complex_table_entry *getComplexTableEntry() {
    // get memory space for a node
        complex_table_entry *r, *r2;

        if (Complex_Avail != nullptr)    // get node from avail chain if possible
        {
            r = Complex_Avail;
            Complex_Avail = Complex_Avail->next;
        } else {            // otherwise allocate COMPLEX_CHUNK_SIZE new nodes
            r = new complex_table_entry[COMPLEX_CHUNK_SIZE];
            complexChunk *c = new complexChunk;
            c->next = complex_chunks;
            c->entry = r;
            complex_chunks = c;

            r2 = r + 1;
            Complex_Avail = r2;
            for (unsigned int i = 0; i < COMPLEX_CHUNK_SIZE - 2; i++, r2++) {
                r2->next = r2 + 1;
            }
            r2->next = nullptr;
        }
        r->next = nullptr;
        r->ref = 0;            // set reference count to 0
        return (r);
    }

    void complexIncRef(complex c) {
        if (c != COMPLEX_ONE && c != COMPLEX_ZERO) {
            ((complex_table_entry *) ((uintptr_t) c.r & (~1ull)))->ref++;
            ((complex_table_entry *) ((uintptr_t) c.i & (~1ull)))->ref++;
        }
    }

    void complexDecRef(complex c) {
        if (c != COMPLEX_ONE && c != COMPLEX_ZERO) {
            ((complex_table_entry *) (((uintptr_t) c.r) & (~1ull)))->ref--;
            ((complex_table_entry *) ((uintptr_t) c.i & (~1ull)))->ref--;
        }
    }

    bool Ceq(const complex a, const complex b) {
        long double ar = CVAL(a.r);
        long double ai = CVAL(a.i);

        long double br = CVAL(b.r);
        long double bi = CVAL(b.i);
        return std::fabs(ar - br) < COMPLEX_TOLERANCE && std::fabs(ai - bi) < COMPLEX_TOLERANCE;
    }

    bool Ceq(complex_value a, complex_value b) {
        return std::fabs(a.r - b.r) < COMPLEX_TOLERANCE && std::fabs(a.i - b.i) < COMPLEX_TOLERANCE;
    }

    void complexInit()
    // initialization
    {
        complex_chunks = nullptr;
        ComplexCache_Avail = new complex_table_entry[COMPLEX_INIT_SIZE * 6];
        CacheStart = ComplexCache_Avail;

        complex_table_entry *r = ComplexCache_Avail;
        for (unsigned int i = 0; i < COMPLEX_INIT_SIZE * 6 - 1; i++, r++) {
            r->next = r + 1;
            r->ref = 0;
        }
        r->next = nullptr;

        initComplexTable();

        complex one;
        one.r = ComplexCache_Avail;
        one.i = ComplexCache_Avail->next;

        one.r->val = 1.0;
        one.i->val = 0.0;

        COMPLEX_ONE = Clookup(one);
        COMPLEX_ZERO.r = COMPLEX_ZERO.i = COMPLEX_ONE.i;
        COMPLEX_M_ONE.r = (complex_table_entry *) (((uintptr_t) COMPLEX_ONE.r) | 1u);
        COMPLEX_M_ONE.i = COMPLEX_ONE.i;

        COMPLEX_ONE.i->ref++;
        COMPLEX_ONE.r->ref++;
    }

    void complexFree() {
        complexChunk *c;
        while (complex_chunks != nullptr) {
            c = complex_chunks;
            complex_chunks = complex_chunks->next;
            delete c->entry;
            delete c;
        }

        delete[] ComplexCache_Avail;
    }


    long double Ccos(const long double fac, const long double div) {
        return std::cos((PI * fac) / div);
    }

    long double Csin(const long double fac, const long double div) {
        return std::sin((PI * fac) / div);
    }

    complex_value Cmake(const long double r, const long double i) {
        return {r, i};
    }

    // initialize the complex value table and complex operation tables to empty
    void initComplexTable()
    {
        for (auto & i : Complex_table) {
            i = nullptr;
        }
        ComplexCount = 0;
    }


    void printComplexTable()
    // print the complex value table entries
    {
        int nentries = 0;

        std::cout << "Complex value table\n";

        complex_table_entry *p;
        int max = -1;
        const auto prec_before = std::cout.precision(20);
        for (unsigned int i = 0; i < COMPLEX_NBUCKET; i++) {
            p = Complex_table[i];
            if (p != nullptr) {
                std::cout << "BUCKET " << i << std::endl;
            }

            int num = 0;
            while (p != nullptr) {
                std::cout << "  " << (intptr_t) p << ": " << p->val << std::endl;
                num++;
                nentries++;
                p = p->next;
            }
            max = std::max(max, num);
        }
        std::cout.precision(prec_before);
        std::cout << "Complex table has " << nentries << " entries\n";
        std::cout << "Largest number of entries in bucket: " << max << "\n";
    }

    unsigned long getKey(const long double val) {
        unsigned long key = (unsigned long) (val * (COMPLEX_NBUCKET - 1));
        if (key > COMPLEX_NBUCKET - 1) {
            key = COMPLEX_NBUCKET - 1;
        }
        return key;
    }

    complex_table_entry *lookup(const long double val) {
        complex_table_entry *r = getComplexTableEntry();
        r->ref = 0;
        r->val = val;

        const unsigned long key = getKey(r->val);

        complex_table_entry *p = Complex_table[key];
        while (p != nullptr) {
            if (std::fabs(p->val - val) < COMPLEX_TOLERANCE) {
                r->next = Complex_Avail;
                Complex_Avail = r;
                return p;
            }
            p = p->next;
        }

        const unsigned long key2 = getKey(val - COMPLEX_TOLERANCE);

        if (key2 != key) {
            p = Complex_table[key2];
            while (p != nullptr) {
                if (std::fabs(p->val - val) < COMPLEX_TOLERANCE) {
                    r->next = Complex_Avail;
                    Complex_Avail = r;
                    return p;
                }
                p = p->next;
            }
        }

        const unsigned long key3 = getKey(val + COMPLEX_TOLERANCE);

        if (key3 != key) {
            p = Complex_table[key3];
            while (p != nullptr) {
                if (std::fabs(p->val - val) < COMPLEX_TOLERANCE) {
                    r->next = Complex_Avail;
                    Complex_Avail = r;
                    return p;
                }
                p = p->next;
            }
        }

        r->next = Complex_table[key];
        Complex_table[key] = r;

        ComplexCount++;
        return r;
    }

    complex Clookup(const complex &c) {
        bool sign_r = std::signbit(c.r->val);
        bool sign_i = std::signbit(c.i->val);

        if (c.r->val == 0) {
            sign_r = false;
        }
        if (c.i->val == 0) {
            sign_i = false;
        }

        complex ret;

        //Lookup real part of complex number
        ret.r = lookup(std::fabs(c.r->val));
        ret.i = lookup(std::fabs(c.i->val));

        //Store sign bit in pointers
        if (sign_r) {
            ret.r = (complex_table_entry *) (((uintptr_t) (ret.r)) | 1u);
        }
        if (sign_i) {
            ret.i = (complex_table_entry *) (((uintptr_t) (ret.i)) | 1u);
        }

        return ret;
    }

    complex Cconjugate(const complex a)
    // return complex conjugate
    {
        complex ret = a;
        if (a.i != COMPLEX_ZERO.i) {
            ret.i = (complex_table_entry *) (((uintptr_t) a.i) ^ 1u);
        }
        return ret;
    }


    // basic operations on complex values
    // meanings are self-evident from the names
    // NOTE arguments are the indices to the values
    // in the complex value table not the values themselves

    complex Cnegative(const complex a) {
        complex ret = a;
        if (a.i != COMPLEX_ZERO.r) {
            ret.i = (complex_table_entry *) (((uintptr_t) a.i) ^ 1u);
        }
        if (a.r != COMPLEX_ZERO.r) {
            ret.r = (complex_table_entry *) (((uintptr_t) a.r) ^ 1u);
        }
        return ret;
    }

    void Cadd(complex &r, const complex a, const complex b) {
        r.r->val = CVAL(a.r) + CVAL(b.r);
        r.i->val = CVAL(a.i) + CVAL(b.i);
    }

    void Csub(complex &r, const complex a, const complex b) {
        r.r->val = CVAL(a.r) - CVAL(b.r);
        r.i->val = CVAL(a.i) - CVAL(b.i);

    }

    void Cmul(complex &r, const complex a, const complex b) {
        if (a == COMPLEX_ONE) {
            r.r->val = CVAL(b.r);
            r.i->val = CVAL(b.i);
            return;
        }
        if (b == COMPLEX_ONE) {
            r.r->val = CVAL(a.r);
            r.i->val = CVAL(a.i);
            return;
        }

        long double ar = CVAL(a.r);
        long double ai = CVAL(a.i);
        long double br = CVAL(b.r);
        long double bi = CVAL(b.i);

        r.r->val = ar * br - ai * bi;
        r.i->val = ar * bi + ai * br;
    }

    void Cdiv(complex &r, const complex a, const complex b) {
        if (a == b) {
            r.r->val = 1;
            r.i->val = 0;
            return;
        }

        long double ar = CVAL(a.r);
        long double ai = CVAL(a.i);
        long double br = CVAL(b.r);
        long double bi = CVAL(b.i);

        long double cmag = br * br + bi * bi;

        r.r->val = (ar * br + ai * bi) / cmag;
        r.i->val = (-ar * bi + ai * br) / cmag;

    }

    long double CmagSquared(const complex &a) {
        long double ar = CVAL(a.r);
        long double ai = CVAL(a.i);

        return ar * ar + ai * ai;
    }

    void garbageCollectComplexTable() {
        complex_table_entry *cur;
        complex_table_entry *prev;
        complex_table_entry *suc;

        for (auto & i : Complex_table) {
            prev = nullptr;
            cur = i;
            while (cur != nullptr) {
                if (cur->ref == 0) {
                    suc = cur->next;
                    if (prev == nullptr) {
                        i = suc;
                    } else {
                        prev->next = suc;
                    }
                    cur->next = Complex_Avail;
                    Complex_Avail = cur;

                    cur = suc;
                    ComplexCount--;
                } else {
                    prev = cur;
                    cur = cur->next;
                }
            }
        }
    }
}