/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_COMPLEXTABLE_HPP
#define DD_PACKAGE_COMPLEXTABLE_HPP

#include "Definitions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dd {
    template<std::size_t NBUCKET = 32768, std::size_t INITIAL_ALLOCATION_SIZE = 2048, std::size_t GROWTH_FACTOR = 2, std::size_t INITIAL_GC_LIMIT = 65536>
    class ComplexTable {
    public:
        struct Entry {
            fp       value{};
            Entry*   next{};
            RefCount refCount{};

            ///
            /// The sign of number is encoded in the least significant bit of its entry pointer
            /// If not handled properly, this causes misaligned access
            /// These routines allow to obtain safe pointers
            ///
            [[nodiscard]] static inline Entry* getAlignedPointer(const Entry* e) {
                return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) & (~1ULL));
            }
            [[nodiscard]] static inline Entry* getNegativePointer(const Entry* e) {
                return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) | 1ULL);
            }
            [[nodiscard]] static inline Entry* flipPointerSign(const Entry* e) {
                return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) ^ 1ULL);
            }
            [[nodiscard]] static inline bool isNegativePointer(const Entry* e) {
                return reinterpret_cast<std::uintptr_t>(e) & 1ULL;
            }

            [[nodiscard]] static inline fp val(const Entry* e) {
                if (isNegativePointer(e)) {
                    return -getAlignedPointer(e)->value;
                }
                return e->value;
            }

            [[nodiscard]] static inline RefCount ref(const Entry* e) {
                if (isNegativePointer(e)) {
                    return -getAlignedPointer(e)->refCount;
                }
                return e->refCount;
            }

            [[nodiscard]] static inline bool approximatelyEquals(const Entry* left, const Entry* right) {
                return std::abs(val(left) - val(right)) < TOLERANCE;
            }

            [[nodiscard]] static inline bool approximatelyZero(const Entry* e) {
                return e == &zero || std::abs(val(e)) < TOLERANCE;
            }

            [[nodiscard]] static inline bool approximatelyOne(const Entry* e) {
                return e == &one || std::abs(val(e) - 1) < TOLERANCE;
            }

            static void writeBinary(const Entry* e, std::ostream& os) {
                auto temp = val(e);
                os.write(reinterpret_cast<const char*>(&temp), sizeof(decltype(temp)));
            }
        };

        static inline Entry zero{0., nullptr, 1};
        static inline Entry sqrt2_2{SQRT2_2, nullptr, 1};
        static inline Entry one{1., nullptr, 1};

        ComplexTable():
            chunkID(0), allocationSize(INITIAL_ALLOCATION_SIZE), gcLimit(INITIAL_GC_LIMIT) {
            // allocate first chunk of numbers
            chunks.emplace_back(allocationSize);
            allocations += allocationSize;
            allocationSize *= GROWTH_FACTOR;
            chunkIt    = chunks[0].begin();
            chunkEndIt = chunks[0].end();

            // add 1/2 to the complex table and increase its ref count (so that it is not collected)
            lookup(0.5L)->refCount++;
        }

        ~ComplexTable() = default;

        static fp tolerance() {
            return TOLERANCE;
        }
        static void setTolerance(fp tol) {
            TOLERANCE = tol;
        }

        static constexpr std::size_t MASK = NBUCKET - 1;

        // linear (clipped) hash function
        static constexpr std::size_t hash(const fp val) {
            assert(val >= 0);
            auto key = static_cast<std::size_t>(val * MASK + 0.5L);
            return std::min(key, MASK);
        }

        static constexpr std::size_t zeroKey    = hash(0.);
        static constexpr std::size_t sqrt2_2Key = hash(SQRT2_2);
        static constexpr std::size_t oneKey     = hash(1.);

        // access functions
        [[nodiscard]] std::size_t getCount() const { return count; }
        [[nodiscard]] std::size_t getPeakCount() const { return peakCount; }
        [[nodiscard]] std::size_t getAllocations() const { return allocations; }
        [[nodiscard]] std::size_t getGrowthFactor() const { return GROWTH_FACTOR; }
        [[nodiscard]] const auto& getTable() const { return table; }
        [[nodiscard]] bool        availableEmpty() const { return available == nullptr; };

        Entry* lookup(const fp& val) {
            assert(!std::isnan(val));

            // special treatment of zero and one (these are not counted as lookups)
            if (std::abs(val) < TOLERANCE)
                return &zero;

            if (std::abs(val - 1.) < TOLERANCE)
                return &one;

            lookups++;

            // search in intended bucket
            const auto key = hash(val);

            // ensure that the exact value of important constants is checked first on all occasions
            if (key == zeroKey) {
                if (std::abs(val) < TOLERANCE) {
                    ++hits;
                    return &zero;
                } else {
                    ++collisions;
                }
            } else if (key == oneKey) {
                if (std::abs(val - 1.) < TOLERANCE) {
                    ++hits;
                    return &one;
                } else {
                    ++collisions;
                }
            } else if (key == sqrt2_2Key) {
                if (std::abs(val - SQRT2_2) < TOLERANCE) {
                    ++hits;
                    return &sqrt2_2;
                } else {
                    ++collisions;
                }
            }

            const auto& bucket = table[key];
            auto        p      = find(bucket, val);
            if (p != nullptr) {
                return p;
            }

            // search in (potentially) lower bucket
            if (val - TOLERANCE >= 0) {
                const auto lowerKey = hash(val - TOLERANCE);
                if (lowerKey != key) {
                    const auto& lowerBucket = table[lowerKey];
                    p                       = find(lowerBucket, val);
                    if (p != nullptr) {
                        return p;
                    }
                }
            }

            // search in (potentially) higher bucket
            const auto upperKey = hash(val - TOLERANCE);
            if (upperKey != key) {
                const auto& upperBucket = table[upperKey];
                p                       = find(upperBucket, val);
                if (p != nullptr) {
                    return p;
                }
            }

            // value was not found in the table -> get a new entry and add it to the central bucket
            Entry* entry = getEntry();
            entry->value = val;
            entry->next  = table[key];
            table[key]   = entry;
            count++;
            peakCount = std::max(peakCount, count);
            return entry;
        }

        [[nodiscard]] Entry* getEntry() {
            // an entry is available on the stack
            if (!availableEmpty()) {
                Entry* entry = available;
                available    = entry->next;
                // returned entries could have a ref count != 0
                entry->refCount = 0;
                return entry;
            }

            // new chunk has to be allocated
            if (chunkIt == chunkEndIt) {
                chunks.emplace_back(allocationSize);
                allocations += allocationSize;
                allocationSize *= GROWTH_FACTOR;
                chunkID++;
                chunkIt    = chunks[chunkID].begin();
                chunkEndIt = chunks[chunkID].end();
            }

            auto entry = &(*chunkIt);
            ++chunkIt;
            return entry;
        }

        void returnEntry(Entry* entry) {
            entry->next = available;
            available   = entry;
        }

        // increment reference count for corresponding entry
        static void incRef(Entry* entry) {
            // get valid pointer
            auto entryPtr = Entry::getAlignedPointer(entry);

            if (entryPtr == nullptr)
                return;

            // important (static) numbers are never altered
            if (entryPtr != &one && entryPtr != &zero && entryPtr != &sqrt2_2) {
                if (entryPtr->refCount == std::numeric_limits<RefCount>::max()) {
                    std::clog << "[WARN] MAXREFCNT reached for " << entryPtr->value << ". Number will never be collected." << std::endl;
                    return;
                }

                // increase reference count
                entryPtr->refCount++;
            }
        }

        // decrement reference count for corresponding entry
        static void decRef(Entry* entry) {
            // get valid pointer
            auto entryPtr = Entry::getAlignedPointer(entry);

            if (entryPtr == nullptr)
                return;

            // important (static) numbers are never altered
            if (entryPtr != &one && entryPtr != &zero && entryPtr != &sqrt2_2) {
                if (entryPtr->refCount == std::numeric_limits<RefCount>::max()) {
                    return;
                }
                if (entryPtr->refCount == 0) {
                    throw std::runtime_error("In ComplexTable: RefCount of entry " + std::to_string(entryPtr->value) + " is zero before decrement");
                }

                // decrease reference count
                entryPtr->refCount--;
            }
        }

        [[nodiscard]] bool possiblyNeedsCollection() const { return count >= gcLimit; }

        std::size_t garbageCollect(bool force = false) {
            gcCalls++;
            // nothing to be done if garbage collection is not forced, and the limit has not been reached,
            // or the current count is minimal (the complex table always contains at least 0.5)
            if ((!force && count < gcLimit) || count <= 1)
                return 0;

            gcRuns++;
            std::size_t collected = 0;
            std::size_t remaining = 0;
            for (auto& bucket: table) {
                Entry* p     = bucket;
                Entry* lastp = nullptr;
                while (p != nullptr) {
                    if (p->refCount == 0) {
                        Entry* next = p->next;
                        if (lastp == nullptr) {
                            bucket = next;
                        } else {
                            lastp->next = next;
                        }
                        returnEntry(p);
                        p = next;
                        collected++;
                    } else {
                        lastp = p;
                        p     = p->next;
                        remaining++;
                    }
                }
            }
            // The garbage collection limit changes dynamically depending on the number of remaining (active) nodes.
            // If it were not changed, garbage collection would run through the complete table on each successive call
            // once the number of remaining entries reaches the garbage collection limit. It is increased whenever the
            // number of remaining entries is rather close to the garbage collection threshold and decreased if the
            // number of remaining entries is much lower than the current limit.
            if (remaining > gcLimit / 10 * 9) {
                gcLimit = remaining + INITIAL_GC_LIMIT;
            } else if (remaining < gcLimit / 128) {
                gcLimit /= 2;
            }
            count = remaining;
            return collected;
        }

        void clear() {
            // clear table buckets
            for (auto& bucket: table) {
                bucket = nullptr;
            }

            // clear available stack
            available = nullptr;

            // release memory of all but the first chunk TODO: it could be desirable to keep the memory
            while (chunkID > 0) {
                chunks.pop_back();
                chunkID--;
            }
            // restore initial chunk setting
            chunkIt        = chunks[0].begin();
            chunkEndIt     = chunks[0].end();
            allocationSize = INITIAL_ALLOCATION_SIZE * GROWTH_FACTOR;
            allocations    = INITIAL_ALLOCATION_SIZE;

            count     = 0;
            peakCount = 0;

            collisions = 0;
            hits       = 0;
            lookups    = 0;

            gcCalls = 0;
            gcRuns  = 0;
            gcLimit = INITIAL_GC_LIMIT;
        };

        void print() {
            for (std::size_t key = 0; key < table.size(); ++key) {
                auto p = table[key];
                if (p != nullptr)
                    std::cout << key << ": "
                              << "\n";

                while (p != nullptr) {
                    std::cout << "\t\t" << p->value << " " << reinterpret_cast<std::uintptr_t>(p) << " " << p->refCount << "\n";
                    p = p->next;
                }

                if (table[key] != nullptr)
                    std::cout << "\n";
            }
        }

        [[nodiscard]] fp hitRatio() const { return static_cast<fp>(hits) / lookups; }
        [[nodiscard]] fp colRatio() const { return static_cast<fp>(collisions) / lookups; }

        std::ostream& printStatistics(std::ostream& os = std::cout) {
            os << "hits: " << hits << ", collisions: " << collisions << ", looks: " << lookups << ", hitRatio: " << hitRatio() << ", colRatio: " << colRatio() << ", gc calls: " << gcCalls << ", gc runs: " << gcRuns << "\n";
            return os;
        }

    private:
        using Bucket = Entry*;
        using Table  = std::array<Bucket, NBUCKET>;

        Table table{};

        // numerical tolerance to be used for floating point values
        static inline fp TOLERANCE = 1e-13;

        Entry*                                available{};
        std::vector<std::vector<Entry>>       chunks{};
        std::size_t                           chunkID;
        typename std::vector<Entry>::iterator chunkIt;
        typename std::vector<Entry>::iterator chunkEndIt;
        std::size_t                           allocationSize;

        std::size_t allocations = 0;
        std::size_t count       = 0;
        std::size_t peakCount   = 0;

        // table lookup statistics
        std::size_t collisions = 0;
        std::size_t hits       = 0;
        std::size_t lookups    = 0;

        // garbage collection
        std::size_t gcCalls = 0;
        std::size_t gcRuns  = 0;
        std::size_t gcLimit = 100000;

        inline Entry* find(const Bucket& bucket, const fp& val) {
            auto p = bucket;
            while (p != nullptr) {
                if (std::abs(p->value - val) < TOLERANCE) {
                    ++hits;
                    return p;
                }
                ++collisions;
                p = p->next;
            }
            return p;
        }
    };
} // namespace dd
#endif //DD_PACKAGE_COMPLEXTABLE_HPP
