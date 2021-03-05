/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include <sstream>
#include <iomanip>
#include "DDcomplex.h"

namespace dd {
    ComplexTableEntry ComplexNumbers::zeroEntry{0., nullptr, 1};
    ComplexTableEntry ComplexNumbers::oneEntry{1., nullptr, 1};
    ComplexTableEntry *ComplexNumbers::moneEntryPointer{getNegativePointer(&oneEntry)};
    fp ComplexNumbers::TOLERANCE = 1e-13;

    ComplexNumbers::ComplexNumbers() {
        Cache_Avail_Initial_Pointer = new ComplexTableEntry[INIT_SIZE * 6];
        Cache_Avail = Cache_Avail_Initial_Pointer;

        ComplexTableEntry *r = Cache_Avail;
        for (unsigned int i = 0; i < INIT_SIZE * 6 - 1; i++) {
            r->next = r + 1;
            r->ref = 0;
            ++r;
        }
        r->next = nullptr;

        for (auto &entry : ComplexTable) {
            entry = nullptr;
        }

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
        } else {            // otherwise allocate NODE_CHUNK_SIZE new nodes
            r = new ComplexTableEntry[CHUNK_SIZE];
            auto *c = new ComplexChunk;
            c->next = chunks;
            c->entry = r;
            chunks = c;

            r2 = r + 1;
            Avail = r2;
            for (unsigned int i = 0; i < CHUNK_SIZE - 2; i++) {
                r2->next = r2 + 1;
                ++r2;
            }
            r2->next = nullptr;
        }
        r->next = nullptr;
        r->ref = 0;            // set reference count to 0
        return r;
    }

    // print the complex value table entries
    void ComplexNumbers::printComplexTable() const {
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
                std::cout << "  " << reinterpret_cast<intptr_t>(p) << ": ";
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

    void ComplexNumbers::statistics() const {
        int nentries = 0;
        std::cout << "CN statistics:\n";

        int max = -1;
        for (auto p : ComplexTable) {
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
        std::cout << "\tCT Lookups (total): " << ct_calls << "\n";
        std::cout << "\tCT Lookups (misses): " << ct_miss << "\n";

    }

    ComplexTableEntry *ComplexNumbers::lookupVal(const fp &val) {
        ct_calls++;
        assert(!std::isnan(val));

        const auto key = getKey(val);

        auto p = ComplexTable[key];
        while (p != nullptr) {
            if (std::fabs(p->val - val) < TOLERANCE) {
                return p;
            }
            p = p->next;
        }

        if (val - TOLERANCE >= 0) {
            const auto key2 = getKey(val - TOLERANCE);
            if (key2 != key) {
                p = ComplexTable[key2];
                while (p != nullptr) {
                    if (std::fabs(p->val - val) < TOLERANCE) {
                        return p;
                    }
                    p = p->next;
                }
            }
        }

        const auto key3 = getKey(val + TOLERANCE);

        if (key3 != key) {
            p = ComplexTable[key3];
            while (p != nullptr) {
                if (std::fabs(p->val - val) < TOLERANCE) {
                    return p;
                }
                p = p->next;
            }
        }

        ct_miss++;
        auto *r = getComplexTableEntry();
        r->val = val;
        r->next = ComplexTable[key];
        ComplexTable[key] = r;

        count++;
        return r;
    }

    Complex ComplexNumbers::lookup(const Complex &c) {
        if (c == ZERO) {
            return ZERO;
        }
        if (c == ONE) {
            return ONE;
        }

        auto valr = val(c.r);
        auto vali = val(c.i);
        auto absr = std::abs(valr);
        auto absi = std::abs(vali);
        Complex ret{};

        if (absi < TOLERANCE) {
            if (absr < TOLERANCE)
                return ZERO;
            if (std::abs(valr - 1) < TOLERANCE) {
                return ONE;
            }
            if (std::abs(valr + 1) < TOLERANCE) {
                return {moneEntryPointer, &zeroEntry};
            }
            ret.i = &zeroEntry;
            auto sign_r = std::signbit(valr);
            ret.r = lookupVal(absr);
            if (sign_r) {
	            setNegativePointer(ret.r);
            }
            return ret;
        }

        if (absr < TOLERANCE) {
            if (std::abs(vali - 1) < TOLERANCE) {
                return {&zeroEntry, &oneEntry};
            }
            if (std::abs(vali + 1) < TOLERANCE) {
                return {&zeroEntry, moneEntryPointer};
            }
            ret.r = &zeroEntry;
            auto sign_i = std::signbit(vali);
            ret.i = lookupVal(absi);
            if (sign_i) {
	            setNegativePointer(ret.i);
            }
            return ret;
        }

        auto sign_r = std::signbit(valr);
        auto sign_i = std::signbit(vali);

        ret.r = lookupVal(absr);
        ret.i = lookupVal(absi);

        //Store sign bit in pointers
        if (sign_r) {
	        setNegativePointer(ret.r);
        }
        if (sign_i) {
	        setNegativePointer(ret.i);
        }

        return ret;
    }

    Complex ComplexNumbers::lookup(const fp &r, const fp &i) {
        auto absr = std::abs(r);
        auto absi = std::abs(i);
        Complex ret{};

        if (absi < TOLERANCE) {
            if (absr < TOLERANCE)
                return ZERO;
            if (std::abs(r - 1) < TOLERANCE) {
                return ONE;
            }
            if (std::abs(r + 1) < TOLERANCE) {
                return {moneEntryPointer, &zeroEntry};
            }
            ret.i = &zeroEntry;
            auto sign_r = std::signbit(r);
            ret.r = lookupVal(absr);
            if (sign_r) {
	            setNegativePointer(ret.r);
            }
            return ret;
        }

        if (absr < TOLERANCE) {
            if (std::abs(i - 1) < TOLERANCE) {
                return {&zeroEntry, &oneEntry};
            }
            if (std::abs(i + 1) < TOLERANCE) {
                return {&zeroEntry, moneEntryPointer};
            }
            ret.r = &zeroEntry;
            auto sign_i = std::signbit(i);
            ret.i = lookupVal(absi);
            if (sign_i) {
	            setNegativePointer(ret.i);
            }
            return ret;
        }

        auto sign_r = std::signbit(r);
        auto sign_i = std::signbit(i);

        ret.r = lookupVal(absr);
        ret.i = lookupVal(absi);

        //Store sign bit in pointers
        if (sign_r) {
	        setNegativePointer(ret.r);
        }
        if (sign_i) {
	        setNegativePointer(ret.i);
        }

        return ret;
    }


    // return complex conjugate
    Complex ComplexNumbers::conj(const Complex &a) {
        auto ret = a;
        if (a.i != ZERO.i) {
            ret.i = flipPointerSign(a.i);
        }
        return ret;
    }

    Complex ComplexNumbers::neg(const Complex &a) {
        auto ret = a;
        if (a.i != ZERO.i) {
            ret.i = flipPointerSign(a.i);
        }
        if (a.r != ZERO.r) {
            ret.r = flipPointerSign(a.r);
        }
        return ret;
    }

    void ComplexNumbers::add(Complex &r, const Complex &a, const Complex &b) {
        assert(r != ZERO);
        assert(r != ONE);
        r.r->val = val(a.r) + val(b.r);
        r.i->val = val(a.i) + val(b.i);
    }

    void ComplexNumbers::sub(Complex &r, const Complex &a, const Complex &b) {
        r.r->val = val(a.r) - val(b.r);
        r.i->val = val(a.i) - val(b.i);
    }

    void ComplexNumbers::mul(Complex &r, const Complex &a, const Complex &b) {
        assert(r != ZERO);
        assert(r != ONE);
        if (equalsOne(a)) {
            r.r->val = val(b.r);
            r.i->val = val(b.i);
        } else if (equalsOne(b)) {
            r.r->val = val(a.r);
            r.i->val = val(a.i);
        } else if (equalsZero(a) || equalsZero(b)) {
            r.r->val = 0.;
            r.i->val = 0.;
        } else {
	        auto ar = val(a.r);
	        auto ai = val(a.i);
	        auto br = val(b.r);
	        auto bi = val(b.i);

	        r.r->val = ar * br - ai * bi;
	        r.i->val = ar * bi + ai * br;
        }
    }

    void ComplexNumbers::div(Complex &r, const Complex &a, const Complex &b) {
	    assert(r != ZERO && r != ONE);
	   	if (equals(a, b)) {
            r.r->val = 1.;
            r.i->val = 0.;
        } else if (equalsZero(a)) {
            r.r->val = 0.;
            r.i->val = 0.;
        } else if (equalsOne(b)) {
            r.r->val = val(a.r);
            r.i->val = val(a.i);
        } else {
	        auto ar = val(a.r);
	        auto ai = val(a.i);
	        auto br = val(b.r);
	        auto bi = val(b.i);

	        auto cmag = br * br + bi * bi;

	        r.r->val = (ar * br + ai * bi) / cmag;
	        r.i->val = (ai * br - ar * bi) / cmag;
	   	}
    }

    void ComplexNumbers::garbageCollect() {
        ComplexTableEntry *cur;
        ComplexTableEntry *prev;
        ComplexTableEntry *suc;

        for (auto &i : ComplexTable) {
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

    int ComplexNumbers::cacheSize() const {
        ComplexTableEntry *p = Cache_Avail;
        int size = 0;

        intptr_t min = std::numeric_limits<intptr_t>::max();
        intptr_t max = std::numeric_limits<intptr_t>::min();

        while (p != nullptr && size <= 0.9 * CHUNK_SIZE) {
            if (p->ref != 0) {
                std::cerr << "Entry with refcount != 0 in Cache!\n";
                std::cerr << reinterpret_cast<intptr_t>(p) << " " << p->ref << " " << p->val << " " << reinterpret_cast<intptr_t>(p->next) << "\n";
            }
            if ((reinterpret_cast<intptr_t>(p)) < min) { min = reinterpret_cast<intptr_t>(p); }
            if ((reinterpret_cast<intptr_t>(p)) > max) { max = reinterpret_cast<intptr_t>(p); }

            p = p->next;
            size++;
        }
        if (size > 0.9 * CHUNK_SIZE) {
            p = Cache_Avail;
            for (int i = 0; i < 10; i++) {
                std::cout << i << ": " << reinterpret_cast<intptr_t>(p) << "\n";
                p = p->next;
            }
            std::cerr << "Error in Cache!\n" << std::flush;
            exit(1);
        }
        std::cout << "Min ptr in cache: " << min << ", max ptr in cache: " << max << "\n";
        return size;
    }

    void ComplexNumbers::incRef(const Complex &c) {
        // The reference to 0 and 1 are static (it does not matter
        // if the 0/1 stand for the real or imaginary part). Changing
        // the reference counter of 0/1 leads to problems for concurrent
        // execution
        if (c != ZERO && c != ONE) {
            auto *ptr_r = getAlignedPointer(c.r);
            auto *ptr_i = getAlignedPointer(c.i);
            if (ptr_r != &oneEntry && ptr_r != &zeroEntry) {
                ptr_r->ref++;
            }
            if (ptr_i != &oneEntry && ptr_i != &zeroEntry) {
                ptr_i->ref++;
            }
        }
    }


    void ComplexNumbers::decRef(const Complex &c) {
        // The reference to 0 and 1 are static (it does not matter
        // if the 0/1 stand for the real or imaginary part). Changing
        // the reference counter of 0/1 leads to problems for concurrent
        // execution
        if (c != ZERO && c != ONE) {
            auto *ptr_r = getAlignedPointer(c.r);
            auto *ptr_i = getAlignedPointer(c.i);
            if (ptr_r != &oneEntry && ptr_r != &zeroEntry) {
                assert(ptr_r->ref > 0);
                ptr_r->ref--;
            }
            if (ptr_i != &oneEntry && ptr_i != &zeroEntry) {
                assert(ptr_i->ref > 0);
                ptr_i->ref--;
            }
        }
    }

    void ComplexNumbers::printFormattedReal(std::ostream &os, fp r, bool imaginary) {
        if (r == 0.L) {
            os << (std::signbit(r) ? "-" : "+") << "0" << (imaginary ? "i" : "");
            return;
        }
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

        if (m == 0) { // +- 1/sqrt(2)
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u221a\u00bdi";
            } else {
                os << (std::signbit(r) ? "-" : "") << u8"\u221a\u00bd";
            }
            return;
        }

        if (o == 0) { // +- pi
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u03c0i";
            } else {
                os << (std::signbit(r) ? "-" : "") << u8"\u03c0";
            }
            return;
        }

        if (std::abs(n + 1) < TOLERANCE) { // 1/2
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u00bdi";
            } else
                os << (std::signbit(r) ? "-" : "") << u8"\u00bd";
            return;
        }

        if (std::abs(m + 1) < TOLERANCE) { // 1/sqrt(2) 1/2
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u221a\u00bd \u00bdi";
            } else
                os << (std::signbit(r) ? "-" : "") << u8"\u221a\u00bd \u00bd";
            return;
        }

        if (std::abs(o + 1) < TOLERANCE) { // +-pi/2
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u00bd \u03c0i";
            } else
                os << (std::signbit(r) ? "-" : "") << u8"\u00bd \u03c0";
            return;
        }

        if (std::abs(std::round(n) - n) < TOLERANCE && n < 0) { // 1/2^n
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u00bd\u002a\u002a" << (int) std::round(-n) << "i";
            } else
                os << (std::signbit(r) ? "-" : "") << u8"\u00bd\u002a\u002a" << (int) std::round(-n);
            return;
        }

        if (std::abs(std::round(m) - m) < TOLERANCE && m < 0) { // 1/sqrt(2) 1/2^m
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u221a\u00bd \u00bd\u002a\u002a" << (int) std::round(-m) << "i";
            } else
                os << (std::signbit(r) ? "-" : "") << u8"\u221a\u00bd \u00bd\u002a\u002a" << (int) std::round(-m);
            return;
        }

        if (std::abs(std::round(o) - o) < TOLERANCE && o < 0) { // 1/2^o pi
            if (imaginary) {
                os << (std::signbit(r) ? "-" : "+") << u8"\u00bd\u002a\u002a" << (int) std::round(-o) << u8" \u03c0i";
            } else
                os << (std::signbit(r) ? "-" : "") << u8"\u00bd\u002a\u002a" << (int) std::round(-o) << u8" \u03c0";
            return;
        }

        if (imaginary) { // default
            os << (std::signbit(r) ? "" : "+") << r << "i";
        } else
            os << r;
    }

	std::string ComplexNumbers::toString(const Complex& c, bool formatted, int precision) {
		std::ostringstream ss{};

		if(precision >= 0) ss << std::setprecision(precision);

		auto r = ComplexNumbers::val(c.r);
		auto i = ComplexNumbers::val(c.i);

		if (r != 0.) {
			if(formatted) {
				ComplexNumbers::printFormattedReal(ss, r);
			} else {
				ss << r;
			}
		}
		if (i != 0.) {
			if(formatted) {
				if (r == i) {
					ss << "(1+i)";
					return ss.str();
				} else if (i == -r) {
					ss << "(1-i)";
					return ss.str();
				}
				ComplexNumbers::printFormattedReal(ss, i, true);
			} else {
				if(r == 0.) {
					ss << i;
				} else {
					if(i > 0.) {
						ss << "+";
					}
					ss << i;
				}
				ss << "i";
			}
		}
		if (r == 0. && i == 0.) return "0";

		return ss.str();
	}

    std::ostream &operator<<(std::ostream &os, const Complex &c) {
        auto r = ComplexNumbers::val(c.r);
        auto i = ComplexNumbers::val(c.i);

        if (r != 0) {
            ComplexNumbers::printFormattedReal(os, r);
        }
        if (i != 0) {
            if (r == i) {
                os << "(1+i)";
                return os;
            } else if (i == -r) {
                os << "(1-i)";
                return os;
            }
            ComplexNumbers::printFormattedReal(os, i, true);
        }
        if (r == 0 && i == 0) os << 0;

        return os;
    }

    std::ostream &operator<<(std::ostream &os, const ComplexValue &c) {
        if (c.r != 0) {
            ComplexNumbers::printFormattedReal(os, c.r);
        }
        if (c.i != 0) {
            if (c.r == c.i) {
                os << "(1+i)";
                return os;
            } else if (c.i == -c.r) {
                os << "(1-i)";
                return os;
            }
            ComplexNumbers::printFormattedReal(os, c.i, true);
        }
        if (c.r == 0 && c.i == 0) os << 0;

        return os;
    }
}
