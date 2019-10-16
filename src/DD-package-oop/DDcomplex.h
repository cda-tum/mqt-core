/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDcomplex_H
#define DDcomplex_H

#include <ostream>
#include <cmath>
#include <limits>

namespace dd {
	static constexpr long double PI = 3.14159265358979323846264338327950288419716939937510L;
	static constexpr long double SQRT_2 = 0.707106781186547524400844362104849039284835937688474036588L;

	struct ComplexTableEntry {
        long double val;
		ComplexTableEntry *next;
	    int ref;
    };

    struct Complex {
	    ComplexTableEntry *r;
	    ComplexTableEntry *i;
    };

	static ComplexTableEntry zeroEntry{ 0L, nullptr, 1 };
	static ComplexTableEntry oneEntry{ 1L, nullptr, 1 };
	static constexpr Complex C_ONE{ &oneEntry, &zeroEntry };
	static constexpr Complex C_ZERO{ &zeroEntry, &zeroEntry };

    struct ComplexValue {
        long double r, i;
    };

	inline bool operator==(const Complex& lhs, const Complex& rhs) {
		return lhs.r == rhs.r && lhs.i == rhs.i;
	}

	inline bool operator==(const ComplexValue& lhs, const ComplexValue& rhs) {
		return lhs.r == rhs.r && lhs.i == rhs.i;
	}

	inline bool operator!=(const Complex& lhs, const Complex& rhs) {
		return lhs.r != rhs.r || lhs.i != rhs.i;
	}

	inline bool operator!=(const ComplexValue& lhs, const ComplexValue& rhs) {
		return lhs.r != rhs.r || lhs.i != rhs.i;
	}

    typedef ComplexValue Matrix2x2[2][2];

    class ComplexNumbers {
        struct ComplexChunk {
	        ComplexTableEntry *entry;
            ComplexChunk *next;
        };

        static constexpr unsigned short NBUCKET = 32768;
        static constexpr unsigned short CHUNK_SIZE = 2000;
        static constexpr unsigned short INIT_SIZE = 300;

	    static inline unsigned long getKey(const long double& val) {
		    auto key = (unsigned long) (val * (NBUCKET - 1));
		    if (key > NBUCKET - 1) {
			    key = NBUCKET - 1;
		    }
		    return key;
	    };

	    ComplexTableEntry *lookupVal(const long double& val);

	    ComplexTableEntry *getComplexTableEntry();

    public:

	    static constexpr long double TOLERANCE = 1e-10l;
	    static constexpr unsigned int GCLIMIT1 = 100000;
	    static constexpr unsigned int GCLIMIT_INC = 0;

	    ComplexTableEntry *ComplexTable[NBUCKET]{ };
	    ComplexTableEntry *Avail = nullptr;
	    ComplexTableEntry *Cache_Avail = nullptr;
	    ComplexTableEntry *Cache_Avail_Intial_Pointer = nullptr;
	    ComplexChunk *chunks = nullptr;

	    unsigned int count;

	    ComplexNumbers();
	    ~ComplexNumbers();

	    // operations on complex numbers
	    // meanings are self-evident from the names
	    static inline long double val(const ComplexTableEntry *x) {
            if (((uintptr_t) x) & (uintptr_t) 1) {
	            return -((ComplexTableEntry *) (((uintptr_t) x) ^ (uintptr_t) 1))->val;
            }
            return x->val;
        }

	    static inline bool equals(const Complex& a, const Complex& b) {
		    return std::fabs(val(a.r) - val(b.r)) < TOLERANCE && std::fabs(val(a.i) - val(b.i)) < TOLERANCE;
        };

	    static inline bool equals(const ComplexValue& a, const ComplexValue& b) {
	    	return std::fabs(a.r - b.r) < TOLERANCE && std::fabs(a.i - b.i) < TOLERANCE;
	    }

	    static inline bool equalsZero(const Complex& c) {
	    	return c == C_ZERO || (std::fabs(val(c.r)) < TOLERANCE && std::fabs(val(c.i)) < TOLERANCE);
	    }

	    static inline bool equalsOne(const Complex& c) {
		    return c == C_ONE || (std::fabs(val(c.r) - 1) < TOLERANCE && std::fabs(val(c.i)) < TOLERANCE);
	    }
	    static void add(Complex& r, const Complex& a, const Complex& b);
	    static void sub(Complex& r, const Complex& a, const Complex& b);
	    static void mul(Complex& r, const Complex& a, const Complex& b);
	    static void div(Complex& r, const Complex& a, const Complex& b);

	    static long double mag2(const Complex& a);
	    static Complex conj(const Complex& a);
	    static Complex neg(const Complex& a);

	    inline Complex addCached(const Complex& a, const Complex& b) {
		    auto c = getCachedComplex();
		    add(c, a, b);
		    return c;
	    }

	    inline Complex subCached(const Complex& a, const Complex& b) {
		    auto c = getCachedComplex();
		    sub(c, a, b);
		    return c;
	    }

	    inline Complex mulCached(const Complex& a, const Complex& b) {
		    auto c = getCachedComplex();
		    mul(c, a, b);
		    return c;
	    }

	    inline Complex divCached(const Complex& a, const Complex& b) {
		    auto c = getCachedComplex();
		    div(c, a, b);
		    return c;
	    }

	    inline void releaseCached(const Complex& c) {
		    c.i->next = Cache_Avail;
		    Cache_Avail = c.r;
	    }

	    // lookup a complex value in the complex value table; if not found add it
        Complex lookup(const Complex &c);

	    Complex lookup(const long double& r, const long double& i);

	    inline Complex lookup(const ComplexValue& c) { return lookup(c.r, c.i); }

	    // reference counting and garbage collection
	    static void incRef(const Complex& c);
	    static void decRef(const Complex& c);
        void garbageCollect();

        // provide (temporary) cached complex number
        inline Complex getTempCachedComplex() const {
        	return { Cache_Avail, Cache_Avail->next};
        }

	    inline Complex getTempCachedComplex(const long double& r, const long double& i) const {
		    Cache_Avail->val = r;
		    Cache_Avail->next->val = i;
        	return { Cache_Avail, Cache_Avail->next };
        }
        inline Complex getCachedComplex() {
	        Complex c{ Cache_Avail, Cache_Avail->next };
	        Cache_Avail = Cache_Avail->next->next;
	        return c;
        }

	    inline Complex getCachedComplex(const long double& r, const long double& i) {
		    Complex c{ Cache_Avail, Cache_Avail->next };
		    c.r->val = r;
		    c.i->val = i;
		    Cache_Avail = Cache_Avail->next->next;
		    return c;
	    }

	    // printing
        void printComplexTable();

	    int cacheSize();
    };

	inline std::ostream& operator<<(std::ostream& os, const Complex c) {
		long double r = ComplexNumbers::val(c.r);
		long double i = ComplexNumbers::val(c.i);

		if (r != 0) {
			os << r;
		}
		if (i != 0) {
			os << std::showpos << i << "i" << std::noshowpos;
		}
		if (r == 0 && i == 0) os << 0;

		return os;
	}

	inline std::ostream& operator<<(std::ostream& os, const ComplexValue c) {
		if (c.r != 0) {
			os << c.r;
		}
		if (c.i != 0) {
			os << std::showpos << c.i << "i" << std::noshowpos;
		}
		if (c.r == 0 && c.i == 0) os << 0;

		return os;
	}
}
#endif
