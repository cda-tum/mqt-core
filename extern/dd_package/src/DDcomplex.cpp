/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include <iostream>

#include "DDcomplex.h"

namespace dd {
	ComplexTableEntry ComplexNumbers::zeroEntry{0L, nullptr, 1};
	ComplexTableEntry ComplexNumbers::oneEntry{1L, nullptr, 1 };
	constexpr Complex ComplexNumbers::ONE;
	constexpr Complex ComplexNumbers::ZERO;

    ComplexNumbers::ComplexNumbers() {
        Cache_Avail_Initial_Pointer = new ComplexTableEntry[INIT_SIZE * 6];
	    Cache_Avail = Cache_Avail_Initial_Pointer;

	    ComplexTableEntry *r = Cache_Avail;
        for (unsigned int i = 0; i < INIT_SIZE * 6 - 1; i++, r++) {
            r->next = r + 1;
            r->ref = 0;
        }
        r->next = nullptr;

	    for (auto& entry : ComplexTable) {
		    entry = nullptr;
        }
	    //count = 0;

	    ComplexTable[0] = ZERO.r;
	    ComplexTable[NBUCKET - 1] = ONE.r;
	    count = 2;

	    lookupVal(0.5L)->ref++;
	    lookupVal(SQRT_2)->ref++;
    }

    ComplexNumbers::~ComplexNumbers() {
        ComplexChunk *c;
        while (chunks != nullptr) {
            c = chunks;
	        chunks = chunks->next;
            delete[] c->entry;
            delete c;
        }

        delete[] Cache_Avail_Initial_Pointer;
    }

	ComplexTableEntry *ComplexNumbers::getComplexTableEntry() {
    // get memory space for a node
		ComplexTableEntry *r, *r2;

        if (Avail != nullptr)    // get node from avail chain if possible
        {
            r = Avail;
	        Avail = Avail->next;
        } else {            // otherwise allocate CHUNK_SIZE new nodes
	        r = new ComplexTableEntry[CHUNK_SIZE];
            auto *c = new ComplexChunk;
            c->next = chunks;
            c->entry = r;
	        chunks = c;

            r2 = r + 1;
	        Avail = r2;
            for (unsigned int i = 0; i < CHUNK_SIZE - 2; i++, r2++) {
                r2->next = r2 + 1;
            }
            r2->next = nullptr;
        }
        r->next = nullptr;
        r->ref = 0;            // set reference count to 0
        return r;
    }

	void ComplexNumbers::printComplexTable()
    // print the complex value table entries
    {
        int nentries = 0;

        std::cout << "Complex value table\n";

	    ComplexTableEntry *p;
        int max = -1;
        const auto prec_before = std::cout.precision(20);
        for (unsigned int i = 0; i < NBUCKET; i++) {
            p = ComplexTable[i];
            if (p != nullptr) {
                std::cout << "BUCKET " << i << std::endl;
            }

            int num = 0;
            while (p != nullptr) {
                std::cout << "  " << (intptr_t) p << ": ";
                printFormattedReal(std::cout, p->val);
                std::cout << " " << p->ref;
                std::cout << std::endl;
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

    void ComplexNumbers::statistics() {
	    int nentries = 0;
	    std::cout << "CN statistics:\n";

	    int max = -1;
	    for (auto & p : ComplexTable) {
		    int num = 0;
		    while (p != nullptr) {
			    num++;
			    nentries++;
			    p = p->next;
		    }
		    max = std::max(max, num);
	    }
	    std::cout << "\tComplex table has " << nentries << " entries\n";
	    std::cout << "\tLargest number of entries in bucket: " << max << "\n";
    }

	ComplexTableEntry *ComplexNumbers::lookupVal(const fp& val) {
        assert(!std::isnan(val));

        const auto key = getKey(val);

		auto p = ComplexTable[key];
        while (p != nullptr) {
            if (std::fabs(p->val - val) < TOLERANCE)
                return p;
            p = p->next;
        }

        if (val - TOLERANCE >= 0) {
            const auto key2 = getKey(val - TOLERANCE);
            if (key2 != key) {
                p = ComplexTable[key2];
                while (p != nullptr) {
                    if (std::fabs(p->val - val) < TOLERANCE)
                        return p;
                    p = p->next;
                }
            }
        }

        const auto key3 = getKey(val + TOLERANCE);

        if (key3 != key) {
            p = ComplexTable[key3];
            while (p != nullptr) {
                if (std::fabs(p->val - val) < TOLERANCE)
                    return p;
                p = p->next;
            }
        }

		auto r = getComplexTableEntry();
		r->val = val;
        r->next = ComplexTable[key];
	    ComplexTable[key] = r;

        count++;
        return r;
    }

    Complex ComplexNumbers::lookup(const Complex &c) {
        if (equalsZero(c))
    		return ZERO;
        if (equalsOne(c))
		    return ONE;

        bool sign_r = std::signbit(c.r->val);
        bool sign_i = std::signbit(c.i->val);

        if (c.r->val == 0) {
            sign_r = false;
        }
        if (c.i->val == 0) {
            sign_i = false;
        }

	    Complex ret{ lookupVal(std::abs(val(c.r))), lookupVal(std::abs(val(c.i))) };

        //Store sign bit in pointers
        if (sign_r) {
	        ret.r = (ComplexTableEntry *) (((uintptr_t) (ret.r)) | 1u);
        }
        if (sign_i) {
	        ret.i = (ComplexTableEntry *) (((uintptr_t) (ret.i)) | 1u);
        }

        return ret;
    }

	Complex ComplexNumbers::lookup(const fp& r, const fp& i) {
        if (std::fabs(i) < TOLERANCE) {
    		if (std::abs(r-1L)< TOLERANCE)
			    return ONE;
    		else if (std::abs(r) < TOLERANCE)
			    return ZERO;
        }

		Complex ret{ lookupVal(std::abs(r)), lookupVal(std::abs(i)) };

		//Store sign bit in pointers
		if (r < 0) {
			ret.r = (ComplexTableEntry *) (((uintptr_t) (ret.r)) | 1u);
		}
		if (i < 0) {
			ret.i = (ComplexTableEntry *) (((uintptr_t) (ret.i)) | 1u);
		}

		return ret;
	}


	// return complex conjugate
	Complex ComplexNumbers::conj(const Complex& a)
    {
        Complex ret = a;
        if (a.i != ZERO.i) {
	        ret.i = (ComplexTableEntry *) (((uintptr_t) a.i) ^ 1u);
        }
        return ret;
    }

	Complex ComplexNumbers::neg(const Complex& a) {
		auto ret = a;
        if (a.i != ZERO.i) {
	        ret.i = (ComplexTableEntry *) (((uintptr_t) a.i) ^ 1u);
        }
        if (a.r != ZERO.r) {
	        ret.r = (ComplexTableEntry *) (((uintptr_t) a.r) ^ 1u);
        }
		return ret;
    }

	void ComplexNumbers::add(Complex& r, const Complex& a, const Complex& b) {
        r.r->val = val(a.r) + val(b.r);
        r.i->val = val(a.i) + val(b.i);
    }

	void ComplexNumbers::sub(Complex& r, const Complex& a, const Complex& b) {
        r.r->val = val(a.r) - val(b.r);
        r.i->val = val(a.i) - val(b.i);
    }

	void ComplexNumbers::mul(Complex& r, const Complex& a, const Complex& b) {
        if (equalsOne(a)) {
            r.r->val = val(b.r);
            r.i->val = val(b.i);
            return;
        } else if (equalsOne(b)) {
            r.r->val = val(a.r);
            r.i->val = val(a.i);
            return;
        } else if (equalsZero(a) || equalsZero(b)) {
	        r.r->val = 0.L;
	        r.i->val = 0.L;
	        return;
        }

        auto ar = val(a.r);
        auto ai = val(a.i);
        auto br = val(b.r);
        auto bi = val(b.i);

        r.r->val = ar * br - ai * bi;
        r.i->val = ar * bi + ai * br;
    }

	void ComplexNumbers::div(Complex& r, const Complex& a, const Complex& b) {
        if (equals(a, b)) {
            r.r->val = 1;
            r.i->val = 0;
            return;
        } else if (equalsZero(a)) {
	        r.r->val = 0;
	        r.i->val = 0;
	        return;
        } else if (equalsOne(b)) {
        	r.r->val = val(a.r);
        	r.i->val = val(a.i);
	        return;
        }

        auto ar = val(a.r);
        auto ai = val(a.i);
        auto br = val(b.r);
        auto bi = val(b.i);

        auto cmag = br * br + bi * bi;

        r.r->val = (ar * br + ai * bi) / cmag;
        r.i->val = (-ar * bi + ai * br) / cmag;
    }

    void ComplexNumbers::garbageCollect() {
	    ComplexTableEntry *cur;
	    ComplexTableEntry *prev;
	    ComplexTableEntry *suc;

        for (auto & i : ComplexTable) {
            prev = nullptr;
            cur = i;
            while (cur != nullptr) {
                assert(cur->ref >= 0);
                if (cur->ref == 0) {
                    suc = cur->next;
                    if (prev == nullptr) {
                        i = suc;
                    } else {
                        prev->next = suc;
                    }
                    cur->next = Avail;
	                Avail = cur;

                    cur = suc;
                    count--;
                } else {
                    prev = cur;
                    cur = cur->next;
                }
            }
        }
    }

	int ComplexNumbers::cacheSize() {
		ComplexTableEntry *p = Cache_Avail;
		int size = 0;

		intptr_t min = std::numeric_limits<intptr_t>::max();
		intptr_t max = std::numeric_limits<intptr_t>::min();

		while (p != nullptr && size <= 0.9 * CHUNK_SIZE) {
			if (p->ref != 0) {
				std::cerr << "Entry with refcount != 0 in Cache!\n";
				std::cerr << (intptr_t) p << " " << p->ref << " " << p->val << " " << (intptr_t) p->next << "\n";
			}
			if (((intptr_t) p) < min) { min = (intptr_t) p; }
			if (((intptr_t) p) > max) { max = (intptr_t) p; }

			p = p->next;
			size++;
		}
		if (size > 0.9 * CHUNK_SIZE) {
			p = Cache_Avail;
			for (int i = 0; i < 10; i++) {
				std::cout << i << ": " << (uintptr_t) p << "\n";
				p = p->next;
			}
			std::cerr << "Error in Cache!\n" << std::flush;
			exit(1);
		}
		std::cout << "Min ptr in cache: " << min << ", max ptr in cache: " << max << "\n";
		return size;
	}

	void ComplexNumbers::incRef(const Complex& c) {
		if (c != ZERO && c != ONE) {
		    auto* ptr_r = ((ComplexTableEntry *) ((uintptr_t) c.r & (~1ull)));
		    auto* ptr_i = ((ComplexTableEntry *) ((uintptr_t) c.i & (~1ull)));
			ptr_r->ref++;
			ptr_i->ref++;
		}
	}

	void ComplexNumbers::decRef(const Complex& c) {
        if (c != ZERO && c != ONE) {
            auto* ptr_r = ((ComplexTableEntry *) ((uintptr_t) c.r & (~1ull)));
            auto* ptr_i = ((ComplexTableEntry *) ((uintptr_t) c.i & (~1ull)));
            assert(ptr_r->ref > 0);
            assert(ptr_i->ref > 0);
            ptr_r->ref--;
            ptr_i->ref--;
        }
	}

	void ComplexNumbers::printFormattedReal(std::ostream& os, fp r, bool imaginary) {
		assert(r != 0.L);
    	auto n = std::log2(std::abs(r));
		auto m = std::log2(std::abs(r) / SQRT_2);
		auto o = std::log2(std::abs(r) / PI);

		if (n == 0) { // +-1
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "i";
			} else
				os << (std::signbit(r) ? "-" : "") << 1;
			return;
		}

		if (m == 0) { // +- sqrt(2)
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u221a\u00bdi";
			} else {
				os << (std::signbit(r) ? "-" : "") << "\u221a\u00bd";
			}
			return;
		}

		if (o == 0) { // +- pi
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u03c0i";
			} else {
				os << (std::signbit(r) ? "-" : "") << "\u03c0";
		}
			return;
		}

		if (std::abs(n + 1) < TOLERANCE) { // 1/2
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u00bdi";
			} else
				os << (std::signbit(r) ? "-" : "") << "\u00bd";
			return;
		}

		if (std::abs(m + 1) < TOLERANCE) { // 1/sqrt(2) 1/2
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u221a\u00bd \u00bdi";
			} else
				os << (std::signbit(r) ? "-" : "") << "\u221a\u00bd \u00bd";
			return;
		}

		if (std::abs(o + 1) < TOLERANCE) { // +-pi/2
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u00bd \u03c0i";
			} else
				os << (std::signbit(r) ? "-" : "") << "\u00bd \u03c0";
			return;
	}

		if (std::abs(std::round(n)-n) < TOLERANCE && n < 0) { // 1/2^n
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u00bd\u002a\u002a" << (int) std::round(-n) << "i";
			} else
				os << (std::signbit(r) ? "-" : "") << "\u00bd\u002a\u002a" << (int) std::round(-n);
			return;
		}

		if (std::abs(std::round(m) - m) < TOLERANCE && m < 0) { // 1/sqrt(2) 1/2^m
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u221a\u00bd \u00bd\u002a\u002a" << (int) std::round(-m) << "i";
			} else
				os << (std::signbit(r) ? "-" : "") << "\u221a\u00bd \u00bd\u002a\u002a" << (int) std::round(-m);
			return;
		}

		if (std::abs(std::round(o) - o) < TOLERANCE && o < 0) { // 1/2^o pi
			if (imaginary) {
				os << (std::signbit(r) ? "-" : "+") << "\u00bd\u002a\u002a" << (int) std::round(-o) << " \u03c0i";
			} else
				os << (std::signbit(r) ? "-" : "") << "\u00bd\u002a\u002a" << (int) std::round(-o) << " \u03c0";
			return;
		}

		if (imaginary) { // default
			os << (std::signbit(r) ? "" : "+") << r << "i";
		} else
			os << r;
	}

}
