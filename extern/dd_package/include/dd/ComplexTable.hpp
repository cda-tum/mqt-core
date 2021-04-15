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
#include <forward_list>
#include <iostream>
#include <limits>
#include <stack>
#include <vector>

namespace dd {
    template<std::size_t NBUCKET = 32768, std::size_t INITIAL_ALLOCATION_SIZE = 2048, std::size_t GROWTH_FACTOR = 2, std::size_t INITIAL_GC_LIMIT = 100000, std::size_t GC_INCREMENT = 0>
    class ComplexTable {
    public:
        struct Entry {
            fp       value{};
            RefCount refCount{};

            ///
            /// The sign of number is encoded in the least significant bit of its entry pointer
            /// If not handled properly, this causes misaligned access
            /// These routines allow to obtain safe pointers
            ///
            [[nodiscard]] inline Entry* getAlignedPointer() const {
                return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(this) & (~1ULL));
            }
            [[nodiscard]] inline Entry* getNegativePointer() const {
                return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(this) | 1ULL);
            }
            [[nodiscard]] inline Entry* flipPointerSign() const {
                return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(this) ^ 1ULL);
            }
            [[nodiscard]] inline bool isNegativePointer() const {
                return reinterpret_cast<std::uintptr_t>(this) & 1ULL;
            }

            [[nodiscard]] inline fp val() const {
                if (isNegativePointer()) {
                    return -getAlignedPointer()->value;
                }
                return value;
            }

            [[nodiscard]] inline RefCount ref() const {
                if (isNegativePointer()) {
                    return -getAlignedPointer()->refCount;
                }
                return refCount;
            }

            [[nodiscard]] inline bool approximatelyEquals(const Entry* entry) const {
                return std::abs(val() - entry->val()) < TOLERANCE;
            }

            [[nodiscard]] inline bool approximatelyZero() const {
                return this == &zero || std::abs(val()) < TOLERANCE;
            }

            [[nodiscard]] inline bool approximatelyOne() const {
                return this == &one || std::abs(val() - 1) < TOLERANCE;
            }

            void writeBinary(std::ostream& os) const {
                auto temp = val();
                os.write(reinterpret_cast<const char*>(&temp), sizeof(decltype(value)));
            }
        };

        static inline Entry zero{0., 1};
        static inline Entry one{1., 1};

        ComplexTable():
            chunkID(0), allocationSize(INITIAL_ALLOCATION_SIZE), gcLimit(INITIAL_GC_LIMIT) {
            // allocate first chunk of numbers
            chunks.emplace_back(allocationSize);
            allocations += allocationSize;
            allocationSize *= GROWTH_FACTOR;
            chunkIt    = chunks[0].begin();
            chunkEndIt = chunks[0].end();

            // emplace static zero and one in the table
            table[0].push_front(&zero);
            table[NBUCKET - 1].push_front(&one);
            count     = 2;
            peakCount = 2;

            // add 1/2 and 1/sqrt(2) to the complex table and increase their ref count (so that they are not collected)
            lookup(0.5L)->refCount++;
            lookup(SQRT2_2)->refCount++;
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
        static std::size_t hash(const fp val) {
            assert(val >= 0);
            auto key = static_cast<std::size_t>(val * MASK);
            return std::min(key, MASK);
        }

        // access functions
        [[nodiscard]] std::size_t getCount() const { return count; }
        [[nodiscard]] std::size_t getPeakCount() const { return peakCount; }
        [[nodiscard]] std::size_t getAllocations() const { return allocations; }
        [[nodiscard]] std::size_t getGrowthFactor() const { return GROWTH_FACTOR; }
        [[nodiscard]] const auto& getTable() const { return table; }

        Entry* lookup(const fp& val) {
            assert(!std::isnan(val));

            // special treatment of zero and one (these are not counted as lookups)
            if (std::abs(val) < TOLERANCE)
                return &zero;

            if (std::abs(val - 1.) < TOLERANCE)
                return &one;

            lookups++;

            // search in intended bucket
            const auto  key    = hash(val);
            const auto& bucket = table[key];
            auto        it     = find(bucket, val);
            if (it != bucket.end()) {
                return (*it);
            }

            // search in (potentially) lower bucket
            if (val - TOLERANCE >= 0) {
                const auto lowerKey = hash(val - TOLERANCE);
                if (lowerKey != key) {
                    const auto& lowerBucket = table[lowerKey];
                    it                      = find(lowerBucket, val);
                    if (it != lowerBucket.end()) {
                        return (*it);
                    }
                }
            }

            // search in (potentially) higher bucket
            const auto upperKey = hash(val - TOLERANCE);
            if (upperKey != key) {
                const auto& upperBucket = table[upperKey];
                it                      = find(upperBucket, val);
                if (it != upperBucket.end()) {
                    return (*it);
                }
            }

            // value was not found in the table -> get a new entry and add it to the central bucket
            auto entry   = getEntry();
            entry->value = val;
            table[key].push_front(entry);
            count++;
            peakCount = std::max(peakCount, count);
            return entry;
        }

        [[nodiscard]] Entry* getEntry() {
            // an entry is available on the stack
            if (!available.empty()) {
                auto entry = available.top();
                available.pop();
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
            available.push(entry);
        }

        // increment reference count for corresponding entry
        static void incRef(Entry* entry) {
            // get valid pointer
            auto entryPtr = entry->getAlignedPointer();

            // `zero` and `one` are static and never altered
            if (entryPtr != &one && entryPtr != &zero) {
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
            auto entryPtr = entry->getAlignedPointer();

            // `zero` and `one` are static and never altered
            if (entryPtr != &one && entryPtr != &zero) {
                assert(entryPtr->refCount > 0);

                // decrease reference count
                entryPtr->refCount--;
            }
        }

        std::size_t garbageCollect(bool force = false) {
            gcCalls++;
            if (!force && count < gcLimit)
                return 0;

            gcRuns++;
            std::size_t collected = 0;
            std::size_t remaining = 0;
            for (auto& bucket: table) {
                auto it     = bucket.begin();
                auto lastit = bucket.before_begin();
                while (it != bucket.end()) {
                    if ((*it)->refCount == 0) {
                        auto entry = (*it);
                        bucket.erase_after(lastit); // erases the element at `it`
                        returnEntry(entry);
                        if (lastit == bucket.before_begin()) {
                            // first entry was removed
                            it = bucket.begin();
                        } else {
                            // entry in middle of list was removed
                            it = ++lastit;
                        }
                        collected++;
                    } else {
                        ++it;
                        ++lastit;
                        remaining++;
                    }
                }
            }
            gcLimit += GC_INCREMENT;
            count = remaining;
            return collected;
        }

        void clear() {
            // clear table buckets
            for (auto& bucket: table) {
                bucket.clear();
            }

            // clear available stack
            while (!available.empty())
                available.pop();

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
                auto& bucket = table[key];
                if (!bucket.empty())
                    std::cout << key << ": ";

                for (const auto& node: bucket)
                    std::cout << "\t\t" << reinterpret_cast<std::uintptr_t>(node) << " " << node->refCount << "\t";

                if (!bucket.empty())
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
        using Bucket = std::forward_list<Entry*>;
        using Table  = std::array<Bucket, NBUCKET>;

        Table table{};

        // numerical tolerance to be used for floating point values
        static inline fp TOLERANCE = 1e-13;

        std::stack<Entry*>                    available{};
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
        std::size_t gcLimit = 250000;

        typename Bucket::const_iterator find(const Bucket& bucket, const fp& val) {
            auto it = bucket.cbegin();
            while (it != bucket.cend()) {
                if (std::abs((*it)->value - val) < TOLERANCE) {
                    ++hits;
                    return it;
                }
                ++collisions;
                ++it;
            }
            return it;
        }
    };
} // namespace dd
#endif //DD_PACKAGE_COMPLEXTABLE_HPP
