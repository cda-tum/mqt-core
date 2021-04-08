/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_H
#define DDpackage_H

#include "DDcomplex.h"

#include <array>
#include <bitset>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

using CN = dd::ComplexNumbers;

namespace dd {
    const std::string        DDversion             = "IIC-DD package v1.2.2";
    constexpr double         SERIALIZATION_VERSION = 0.1;
    constexpr unsigned short RADIX                 = 2;             // logic radix
    constexpr unsigned short NEDGE                 = RADIX * RADIX; // max no. of edges = RADIX^2

    // General package configuration parameters
    constexpr unsigned int   GCLIMIT1          = 250000;                                   // first garbage collection limit
    constexpr unsigned int   GCLIMIT_INC       = 0;                                        // garbage collection limit increment
    constexpr unsigned int   MAXREFCNT         = std::numeric_limits<unsigned int>::max(); // max reference count (saturates at this value)
    constexpr unsigned int   NODECOUNT_BUCKETS = 200000;
    constexpr unsigned short NBUCKET           = 32768;       // no. of hash table buckets; must be a power of 2
    constexpr unsigned short HASHMASK          = NBUCKET - 1; // must be nbuckets-1
    constexpr unsigned short CTSLOTS           = 16384;       // no. of computed table slots
    constexpr unsigned short CTMASK            = CTSLOTS - 1; // must be CTSLOTS-1
    constexpr unsigned short OperationSLOTS    = 16384;
    constexpr unsigned short OperationMASK     = OperationSLOTS - 1;
    constexpr unsigned short TTSLOTS           = 2048;        // Toffoli table slots
    constexpr unsigned short TTMASK            = TTSLOTS - 1; // must be TTSLOTS-1
    constexpr unsigned int   NODE_CHUNK_SIZE   = 2000;        // this parameter may be increased for larger benchmarks to minimize the number of allocations
    constexpr unsigned int   LIST_CHUNK_SIZE   = 2000;

    typedef struct Node* NodePtr;

    struct Edge {
        NodePtr p;
        Complex w;
    };

    struct Node {
        NodePtr      next;        // link for unique table and available space chain
        Edge         e[NEDGE];    // edges out of this node
        unsigned int ref;         // reference count
        short        v;           // variable index (nonterminal) value (-1 for terminal)
        bool         ident, symm; // special matrices flags
    };

    // list definitions for breadth first traversals (e.g. printing)
    typedef struct ListElement* ListElementPtr;

    struct ListElement {
        NodePtr        p;
        ListElementPtr next;
        int            w;
    };

    struct Control {
        enum class Type : bool { pos = true,
                                 neg = false };

        unsigned short qubit;
        Type           type = Type::pos;
    };

    inline bool operator<(const Control& lhs, const Control& rhs) {
        return lhs.qubit < rhs.qubit || (lhs.qubit == rhs.qubit && lhs.type == Control::Type::neg);
    }

    inline bool operator==(const Control& lhs, const Control& rhs) {
        return lhs.qubit == rhs.qubit && lhs.type == rhs.type;
    }

    inline bool operator!=(const Control& lhs, const Control& rhs) {
        return !(lhs == rhs);
    }

    inline namespace literals {
        Control operator""_pc(unsigned long long q);
        Control operator""_nc(unsigned long long q);
    } // namespace literals

    // computed table definitions
    // compute table entry kinds
    enum CTkind {
        none,
        I,
        X,
        Y,
        Z,
        ATrue,
        AFalse,
        ad,
        mult,
        fid,
        transp,
        conjTransp,
        kron,
        ct_count //ct_count must be the final element
    };

    //computed table entry
    struct CTentry1 // computed table entry defn
    {
        Edge   a, b, r; // a and b are arguments, r is the result
        CTkind which;   // type of operation
    };

    struct CTentry2 // computed table entry defn
    {
        Edge         a, b; // a and b are arguments, r is the result
        NodePtr      r;
        CTkind       which; // type of operation
        ComplexValue rw;
    };

    struct CTentry3 // computed table entry defn
    {
        NodePtr      a, b, r; // a and b are arguments, r is the result
        CTkind       which;   // type of operation
        ComplexValue aw, bw, rw;
    };

    struct TTentry // Toffoli table entry defn
    {
        unsigned short    n, target;
        std::set<Control> controls;
        Edge              e;
    };

    struct OperationEntry {
        NodePtr           r;
        ComplexValue      rw;
        unsigned int      operationType;
        unsigned short    target;
        std::set<Control> controls;
    };

    enum Mode {
        Vector,
        Matrix,
        ModeCount
    };

    enum class BasisStates {
        zero,
        one,
        plus,
        minus,
        right,
        left
    };

    class Package {
        static Node            terminal;
        constexpr static Node* terminalNode{&terminal}; // pointer to terminal node

        NodePtr        nodeAvail{}; // pointer to available space chain
        ListElementPtr listAvail{}; // pointer to available list elements for breadth first searches

        unsigned short nqubits;

        // Unique Tables (one per input variable)
        std::vector<std::array<NodePtr, NBUCKET>> Unique{};

        static constexpr int modeCount = static_cast<int>(Mode::ModeCount);
        // Three types since different possibilities for complex numbers  (caused by caching)
        // weights of operands and result are from complex table (e.g., transpose, conjugateTranspose)
        std::array<std::array<CTentry1, CTSLOTS>, modeCount> CTable1{};
        std::array<std::size_t, modeCount>                   CT1count{};

        // weights of operands are from complex table, weight of result from cache/ZERO (e.g., mult)
        std::array<std::array<CTentry2, CTSLOTS>, modeCount> CTable2{};
        std::array<std::size_t, modeCount>                   CT2count{};

        // weights of operands and result are from cache/ZERO (e.g., add)
        std::array<std::array<CTentry3, CTSLOTS>, modeCount> CTable3{};
        std::array<std::size_t, modeCount>                   CT3count{};

        // Toffoli gate table
        //std::array<TTentry, TTSLOTS> TTable{};
        /// gcc is having serious troubles compiling this using std::array (compilation times >15min).
        /// std::vector shouldn't be any less efficient in our application scenario
        /// TODO: revisit this in the future
        std::vector<TTentry> TTable{TTSLOTS};
        // Identity matrix table
        std::vector<Edge> IdTable{};

        // Operation operations table
        //std::array<OperationEntry, OperationSLOTS> OperationTable{};
        /// gcc is having serious troubles compiling this using std::array (compilation times >15min).
        /// std::vector shouldn't be any less efficient in our application scenario
        /// TODO: revisit this in the future
        std::vector<OperationEntry> OperationTable{OperationSLOTS};

        unsigned int              currentNodeGCLimit    = GCLIMIT1;     // current garbage collection limit
        unsigned int              currentComplexGCLimit = CN::GCLIMIT1; // current complex garbage collection limit
        std::vector<unsigned int> active{};                             // number of active nodes for each variable
        unsigned long             nodecount     = 0;                    // node count in unique table
        unsigned long             peaknodecount = 0;                    // records peak node count in unique table

        std::array<unsigned long, ct_count> nOps{};            // operation counters
        std::array<unsigned long, ct_count> CTlook{}, CThit{}; // counters for gathering compute table hit stats
        unsigned long                       operationLook  = 0;
        unsigned long                       operationCThit = 0;

        std::vector<ListElementPtr> allocated_list_chunks;
        std::vector<NodePtr>        allocated_node_chunks;

        Mode mode{Mode::Vector};

        /// private helper routines
        NodePtr getNode();

        Edge         add2(const Edge& x, const Edge& y);
        Edge         multiply2(const Edge& x, const Edge& y, unsigned short var);
        ComplexValue innerProduct(const Edge& x, const Edge& y, int var);
        Edge         trace(const Edge& a, const std::vector<bool>& eliminate, unsigned short alreadyEliminated = 0);
        Edge         kronecker2(const Edge& x, const Edge& y);

        void checkSpecialMatrices(NodePtr p);

        unsigned int   nodeCount(const Edge& e, std::unordered_set<NodePtr>& v) const;
        ListElementPtr newListElement();

    public:
        // edges pointing to zero and one DD constants
        constexpr static Edge DDone{terminalNode, ComplexNumbers::ONE};
        constexpr static Edge DDzero{terminalNode, ComplexNumbers::ZERO};

        // The following variables are supposed to be read only. Treat carefully if you change them!
        unsigned long long node_allocations = 0;            // Number of nodes allocated by getNode()
        unsigned long      activeNodeCount  = 0;            // number of active nodes
        unsigned long      maxActive        = 0;            // maximum number of active nodes
        unsigned long      gc_calls{};                      // number of calls to the garbage collector
        unsigned long      gc_runs{};                       // number of times the GC actually ran
        unsigned long      UTcol{}, UTmatch{}, UTlookups{}; // counter for collisions / matches in hash tables
        ComplexNumbers     cn;                              // instance of the complex number handler

        explicit Package(unsigned short nqubits = 128):
            nqubits(nqubits), cn(ComplexNumbers()) {
            IdTable.resize(nqubits);
            Unique.resize(nqubits);
            active.resize(nqubits);
        };
        ~Package();

        [[nodiscard]] unsigned short qubits() const { return nqubits; }
        void                         resize(unsigned short nq) {
            nqubits = nq;
            IdTable.resize(nqubits);
            Unique.resize(nqubits);
            active.resize(nqubits);
        }

        /// Set normalization mode
        void setMode(const Mode m) {
            mode = m;
        }
        /// Change the tolerance used to decide the equivalence of numbers
        static void setComplexNumberTolerance(const fp tol) {
            CN::setTolerance(tol);
        }

        /// Reset package state
        void reset();

        /// DD build up
        static Edge makeTerminal(const Complex& w) { return {terminalNode, w}; }
        Edge        makeNonterminal(short v, const Edge* edge, bool cached = false);
        Edge        makeNonterminal(const short v, const std::array<Edge, NEDGE>& edge, bool cached = false) {
            return makeNonterminal(v, edge.data(), cached);
        }

        /// State DD generation
        Edge makeZeroState(unsigned short n);
        Edge makeBasisState(unsigned short n, const std::vector<bool>& state);
        Edge makeBasisState(unsigned short n, const std::vector<BasisStates>& state);

        /// Matrix DD generation
        Edge makeIdent(unsigned short n);
        Edge makeIdent(short x, short y);
        Edge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, unsigned short n, unsigned short target) {
            return makeGateDD(mat, n, std::set<Control>{}, target);
        }
        Edge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, unsigned short n, const Control& control, unsigned short target) {
            return makeGateDD(mat, n, std::set{control}, target);
        }
        Edge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, unsigned short n, unsigned short control, unsigned short target) {
            return makeGateDD(mat, n, std::set<Control>{{control}}, target);
        }
        Edge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, unsigned short n, const std::set<Control>& controls, unsigned short target);

        /// Unique table functions
        Edge                      UTlookup(const Edge& e, bool keep_node = false);
        [[nodiscard]] std::string UTcheck(const Edge& e) const;
        static std::size_t        UThash(NodePtr p);
        void                      clearUniqueTable();
        [[nodiscard]] const auto& getUniqueTable() const { return Unique; }

        /// Compute table functions
        Edge                      CTlookup(const Edge& a, const Edge& b, CTkind which);
        void                      CTinsert(const Edge& a, const Edge& b, const Edge& r, CTkind which);
        static unsigned long      CThash(const Edge& a, const Edge& b, CTkind which);
        static unsigned long      CThash2(NodePtr a, const ComplexValue& aw, NodePtr b, const ComplexValue& bw, CTkind which);
        void                      clearComputeTables();
        [[nodiscard]] const auto& getComputeTable1() const { return CTable1; }
        [[nodiscard]] const auto& getComputeTable2() const { return CTable2; }
        [[nodiscard]] const auto& getComputeTable3() const { return CTable3; }

        /// Identity table functions
        inline void clearIdentityTable() {
            for (auto& entry: IdTable) entry.p = nullptr;
        }
        [[nodiscard]] const auto& getIdentityTable() const { return IdTable; }

        /// Toffoli table functions

        // Toffoli table insertion and lookup
        void TTinsert(unsigned short n, const std::set<Control>& controls, unsigned short target, const Edge& e);
        Edge TTlookup(unsigned short n, const std::set<Control>& controls, unsigned short target);

        static inline unsigned short TThash(const std::set<Control>& controls, unsigned short target);
        inline void                  clearToffoliTable() {
            for (auto& entry: TTable) entry.e.p = nullptr;
        }
        [[nodiscard]] const auto& getToffoliTable() const { return TTable; }

        /// Operation table functions
        Edge OperationLookup(unsigned int operationType, unsigned short nQubits, unsigned short target) {
            return OperationLookup(operationType, nQubits, {}, target);
        }
        Edge OperationLookup(unsigned int operationType, unsigned short nQubits, const std::set<Control>& controls, unsigned short target);
        void OperationInsert(unsigned int operationType, const Edge& result, unsigned short nQubits, unsigned short target) {
            OperationInsert(operationType, result, nQubits, {}, target);
        }
        void          OperationInsert(unsigned int operationType, const Edge& result, unsigned short nQubits, const std::set<Control>& controls, unsigned short target);
        unsigned long OperationHash(unsigned int operationType, unsigned short nQubits, const std::set<Control>& controls, unsigned short target);

        inline void clearOperationTable() {
            for (auto& entry: OperationTable) entry.r = nullptr;
        }
        [[nodiscard]] const auto& getOperationTable() const { return OperationTable; }

        /// Operations on DDs
        Edge         multiply(const Edge& x, const Edge& y);
        Edge         add(const Edge& x, const Edge& y);
        Edge         transpose(const Edge& a);
        Edge         conjugateTranspose(const Edge& a);
        Edge         normalize(const Edge& e, bool cached);
        Edge         partialTrace(const Edge& a, const std::vector<bool>& eliminate);
        ComplexValue trace(const Edge& a);
        ComplexValue innerProduct(const Edge& x, const Edge& y);
        fp           fidelity(const Edge& x, const Edge& y);
        Edge         kronecker(const Edge& x, const Edge& y);
        Edge         extend(const Edge& e, unsigned short h = 0, unsigned short l = 0);

        /// Handling of ancillary and garbage qubits
        dd::Edge reduceAncillae(dd::Edge& e, const std::vector<bool>& ancillary, bool regular = true);
        dd::Edge reduceAncillaeRecursion(dd::Edge& e, const std::vector<bool>& ancillary, unsigned short lowerbound, bool regular = true);
        // Garbage reduction works for reversible circuits --- to be thoroughly tested for quantum circuits
        dd::Edge reduceGarbage(dd::Edge& e, const std::vector<bool>& garbage, bool regular = true);
        dd::Edge reduceGarbageRecursion(dd::Edge& e, const std::vector<bool>& garb, unsigned short lowerbound, bool regular = true);

        /// Reference counting and garbage collection
        void incRef(const Edge& e);
        void decRef(const Edge& e);
        void garbageCollect(bool force = false);

        /// Checks
        static inline bool isTerminal(const Edge& e) {
            return e.p == terminalNode;
        }
        static inline bool equals(const Edge& a, const Edge& b) {
            return a.p == b.p && ComplexNumbers::equals(a.w, b.w);
        }

        /// Utility
        // Traverse DD and return product of edge weights along the way
        ComplexValue getValueByPath(const Edge& e, std::string elements);
        ComplexValue getValueByPath(const Edge& e, size_t i, size_t j = 0);
        ComplexValue getValueByPath(const Edge& e, const Complex& amp, size_t i, size_t j);
        using CVec = std::vector<std::pair<float, float>>;
        using CMat = std::vector<CVec>;
        CVec getVector(const Edge& e);
        void getVector(const Edge& e, const Complex& amp, size_t i, CVec& vec);
        CMat getMatrix(const Edge& e);
        void getMatrix(const Edge& e, const Complex& amp, size_t i, size_t j, CMat& mat);
        // Calculate the size of the DD pointed to by e
        unsigned int size(const Edge& e);

        /// Statistics
        void        statistics();
        static void printInformation();

        /// Printing and GraphViz(dot) export
        void printVector(const Edge& e);
        void printActive(int n);
        void printDD(const Edge& e, unsigned int limit);
        void printUniqueTable(unsigned short n);

        /// Debugging
        void        debugnode(NodePtr p) const;
        std::string debugnode_line(NodePtr p) const;
        bool        is_locally_consistent_dd(const Edge& e);
        bool        is_locally_consistent_dd2(const Edge& e);
        bool        is_globally_consistent_dd(const Edge& e);

        void fill_consistency_counter(const Edge& edge, std::map<ComplexTableEntry*, long>& weight_map, std::map<NodePtr, unsigned long>& node_map);
        void check_consistency_counter(const Edge& edge, const std::map<ComplexTableEntry*, long>& weight_map, const std::map<NodePtr, unsigned long>& node_map);
    };

    inline bool operator==(const Edge& lhs, const Edge& rhs) { return Package::equals(lhs, rhs); }
    inline bool operator!=(const Edge& lhs, const Edge& rhs) { return !(lhs == rhs); }
} // namespace dd
#endif
