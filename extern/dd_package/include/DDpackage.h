/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_H
#define DDpackage_H

#include "DDcomplex.h"

#include <unordered_set>
#include <vector>
#include <array>
#include <bitset>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <random>
#include <tuple>



using CN = dd::ComplexNumbers;

namespace dd {
	const std::string DDversion = "IIC-DD package v1.2.2";
	constexpr double SERIALIZATION_VERSION = 0.1;
	constexpr unsigned short RADIX = 2;                     // logic radix
	constexpr unsigned short NEDGE = RADIX * RADIX;   // max no. of edges = RADIX^2

	// General package configuration parameters
	constexpr unsigned int GCLIMIT1 = 250000;         // first garbage collection limit
	constexpr unsigned int GCLIMIT_INC = 0;           // garbage collection limit increment
	constexpr unsigned int MAXREFCNT = 4000000;       // max reference count (saturates at this value)
	constexpr unsigned int NODECOUNT_BUCKETS = 200000;
	constexpr unsigned short NBUCKET = 32768;         // no. of hash table buckets; must be a power of 2
	constexpr unsigned short HASHMASK = NBUCKET - 1;  // must be nbuckets-1
	constexpr unsigned short CTSLOTS = 16384;         // no. of computed table slots
	constexpr unsigned short CTMASK = CTSLOTS - 1;    // must be CTSLOTS-1
    constexpr unsigned short OperationSLOTS = 16384;
    constexpr unsigned short OperationMASK = OperationSLOTS - 1;
	constexpr unsigned short TTSLOTS = 2048;          // Toffoli table slots
	constexpr unsigned short TTMASK = TTSLOTS - 1;    // must be TTSLOTS-1
	constexpr unsigned int NODE_CHUNK_SIZE = 2000;    // this parameter may be increased for larger benchmarks to minimize the number of allocations
	constexpr unsigned int LIST_CHUNK_SIZE = 2000;
	constexpr unsigned short MAXN = 128;              // max no. of inputs

    typedef struct Node *NodePtr;

    struct Edge {
	    NodePtr p;
	    Complex w;
    };

    struct Node {
	    NodePtr next;         // link for unique table and available space chain
	    Edge e[NEDGE];     // edges out of this node
	    unsigned int ref;       // reference count
	    short v;        // variable index (nonterminal) value (-1 for terminal)
	    bool ident, symm; // special matrices flags
    };

    // list definitions for breadth first traversals (e.g. printing)
    typedef struct ListElement *ListElementPtr;

    struct ListElement {
	    NodePtr p;
	    ListElementPtr next;
	    int w;
    };

    // computed table definitions
    // compute table entry kinds
    enum CTkind {
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
        none,
        ct_count //ct_count must be the final element

    };

    //computed table entry
    struct CTentry1// computed table entry defn
    {
	    Edge a, b, r;     // a and b are arguments, r is the result
	    CTkind which;       // type of operation
    };

    struct CTentry2// computed table entry defn
    {
	    Edge a, b;     // a and b are arguments, r is the result
	    NodePtr r;
	    CTkind which;       // type of operation
	    ComplexValue rw;
    };

    struct CTentry3// computed table entry defn
    {
	    NodePtr a, b, r;     // a and b are arguments, r is the result
	    CTkind which;       // type of operation
	    ComplexValue aw, bw, rw;
    };

    struct TTentry // Toffoli table entry defn
    {
	    unsigned short n, m, t;
	    short line[MAXN];
	    Edge e;
    };

    struct OperationEntry
    {
        NodePtr r;
        ComplexValue rw;
        unsigned int operationType;
        short line[MAXN];
    };

	enum Mode {
		Vector, Matrix, ModeCount
	};

	enum class BasisStates {
		zero, one, plus, minus, right, left
	};

    class Package {

    	static Node terminal;
	    constexpr static Node* terminalNode{&terminal};        // pointer to terminal node


        NodePtr nodeAvail{};                 // pointer to available space chain
	    ListElementPtr listAvail{ };           // pointer to available list elements for breadth first searches

	    // Unique Tables (one per input variable)
	    std::array<std::array<NodePtr, NBUCKET>, MAXN> Unique{ };

	    static constexpr int modeCount = static_cast<int>(Mode::ModeCount);
	    // Three types since different possibilities for complex numbers  (caused by caching)
	    // weights of operands and result are from complex table (e.g., transpose, conjugateTranspose)
		std::array<std::array<CTentry1, CTSLOTS>, modeCount> CTable1{};

	    // weights of operands are from complex table, weight of result from cache/ZERO (e.g., mult)
		std::array<std::array<CTentry2, CTSLOTS>, modeCount> CTable2{};

	    // weights of operands and result are from cache/ZERO (e.g., add)
		std::array<std::array<CTentry3, CTSLOTS>, modeCount> CTable3{};

	    // Toffoli gate table
	    std::array<TTentry, TTSLOTS> TTable{ };
	    // Identity matrix table
	    std::array<Edge, MAXN> IdTable{ };

        // Operation operations table
        std::array<OperationEntry, OperationSLOTS> OperationTable{ };

	    unsigned int currentNodeGCLimit = GCLIMIT1;        // current garbage collection limit
	    unsigned int currentComplexGCLimit = CN::GCLIMIT1; // current complex garbage collection limit
		std::array<unsigned int, MAXN> active{ };          // number of active nodes for each variable
	    unsigned long nodecount = 0;                       // node count in unique table
	    unsigned long peaknodecount = 0;                   // records peak node count in unique table

        std::array<unsigned long, ct_count> nOps{};              // operation counters
	    std::array<unsigned long, ct_count> CTlook{}, CThit{};   // counters for gathering compute table hit stats

	    std::vector<ListElementPtr> allocated_list_chunks;
	    std::vector<NodePtr> allocated_node_chunks;

		Mode mode{Mode::Vector};

	    /// private helper routines
	    void initComputeTable();
	    NodePtr getNode();


	    Edge multiply2(const Edge& x, const Edge& y, unsigned short var);
	    ComplexValue innerProduct(const Edge& x, const Edge& y, int var);
	    Edge trace(const Edge& a, const std::bitset<MAXN>& eliminate, unsigned short alreadyEliminated = 0);
	    Edge kronecker2(const Edge& x, const Edge& y);

	    void checkSpecialMatrices(NodePtr p);

        static uintptr_t UThash(NodePtr p);
	    Edge UTlookup(const Edge& e, bool keep_node = false);
        [[nodiscard]] std::string UTcheck(const Edge& e) const;

	    static inline unsigned long CThash(const Edge& a, const Edge& b, CTkind which);
	    static inline unsigned long CThash2(NodePtr a, const ComplexValue& aw, NodePtr b, const ComplexValue& bw, CTkind which);
	    static inline unsigned short TThash(unsigned short n, unsigned short t, const short line[MAXN]);

	    unsigned int nodeCount(const Edge& e, std::unordered_set<NodePtr>& v) const;
	    ListElementPtr newListElement();

    public:
        // edges pointing to zero and one DD constants
        constexpr static Edge DDone{ terminalNode, ComplexNumbers::ONE };
        constexpr static Edge DDzero{ terminalNode, ComplexNumbers::ZERO };

        // The following variables are supposed to be read only. Treat carefully if you change them!
        unsigned long long node_allocations = 0;       // Number of nodes allocated by getNode()
        unsigned long activeNodeCount = 0;             // number of active nodes
        unsigned long maxActive = 0;                   // maximum number of active nodes
        unsigned long gc_calls{};                      // number of calls to the garbage collector
        unsigned long gc_runs{};                       // number of times the GC actually ran
        unsigned long UTcol{}, UTmatch{}, UTlookups{}; // counter for collisions / matches in hash tables
        ComplexNumbers cn;                             // instance of the complex number handler

        Package(): cn(ComplexNumbers()){};
        ~Package();


        // Package setup and reset
        /// Set normalization mode
        void setMode(const Mode m) {
            mode = m;
        }
        /// Change the tolerance till which numbers are considered equal
        static void setComplexNumberTolerance(const fp tol) {
            CN::setTolerance(tol);
        }
        /// Reset package state (probably leaks memory)
        void reset();

        // DD creation
        static inline Edge makeTerminal(const Complex& w) {
            return { terminalNode, w };
        }
	    Edge makeNonterminal(short v, const Edge *edge, bool cached = false);
	    inline Edge makeNonterminal(const short v, const std::array<Edge, NEDGE>& edge, bool cached = false) {
	    	return makeNonterminal(v, edge.data(), cached);
	    }
	    Edge makeZeroState(unsigned short n);
	    Edge makeBasisState(unsigned short n, const std::bitset<MAXN>& state);
	    Edge makeBasisState(unsigned short n, const std::vector<BasisStates>& state);
	    Edge makeIdent(unsigned short n);
	    Edge makeIdent(short x, short y);
	    Edge makeGateDD(const Matrix2x2& mat, unsigned short n, const short *line);
	    Edge makeGateDD(const std::array<ComplexValue,NEDGE>& mat, unsigned short n, const std::array<short,MAXN>& line);
	    Edge makeGateDD(const Matrix2x2& mat, unsigned short n, const std::array<short,MAXN>& line) {
		    return makeGateDD(mat, n, line.data());
	    }

        Edge CTlookup(const Edge& a, const Edge& b, CTkind which);
        void CTinsert(const Edge& a, const Edge& b, const Edge& r, CTkind which);

        long operationCThit = 0;
        long operationLook = 0;

        Edge OperationLookup(unsigned int operationType, const short *line, unsigned short nQubits);
        void OperationInsert(unsigned int operationType, const short *line, const Edge &result, unsigned short nQubits);
        static unsigned long OperationHash(unsigned int operationType, const short *line, unsigned short nQubits);


	    // operations on DDs
	    Edge multiply(const Edge& x, const Edge& y);
	    Edge add(const Edge& x, const Edge& y);
        Edge add2(const Edge& x, const Edge& y);
	    Edge transpose(const Edge& a);
	    Edge conjugateTranspose(const Edge& a);
	    Edge normalize(const Edge& e, bool cached);
	    Edge partialTrace(const Edge& a, const std::bitset<MAXN>& eliminate);
	    ComplexValue trace(const Edge& a);
		ComplexValue innerProduct(const Edge& x, const Edge& y);
	    fp fidelity(const Edge& x, const Edge& y);
	    Edge kronecker(const Edge& x, const Edge& y);
	    Edge extend(const Edge& e, unsigned short h = 0, unsigned short l = 0);

	    // handling of ancillary and garbage qubits
	    dd::Edge reduceAncillae(dd::Edge& e, const std::bitset<dd::MAXN>& ancillary, bool regular = true);
	    dd::Edge reduceAncillaeRecursion(dd::Edge& e, const std::bitset<dd::MAXN>& ancillary, unsigned short lowerbound, bool regular = true);
	    // garbage reduction works for reversible circuits --- to be thoroughly tested for quantum circuits
	    dd::Edge reduceGarbage(dd::Edge& e, const std::bitset<dd::MAXN>& garbage, bool regular = true);
	    dd::Edge reduceGarbageRecursion(dd::Edge& e, const std::bitset<dd::MAXN>& garb, unsigned short lowerbound, bool regular = true);

	    // calculates E_s F(W|s>, |s>), where the expectation is taken over the set of local quantum stimuli
		fp localStimuliExpectation(const Edge& W);

		// utility
        /// Traverse DD and return product of edge weights along the way
		ComplexValue getValueByPath(const Edge& e, std::string elements);
	    ComplexValue getValueByPath(const Edge& e, size_t i, size_t j=0);
	    ComplexValue getValueByPath(const Edge& e, const Complex& amp, size_t i, size_t j);
		using CVec = std::vector<std::pair<float, float>>;
		using CMat = std::vector<CVec>;
		CVec getVector(const Edge& e);
		void getVector(const Edge& e, const Complex& amp, size_t i, CVec& vec);
		CMat getMatrix(const Edge& e);
	    void getMatrix(const Edge& e, const Complex& amp, size_t i, size_t j, CMat& mat);
	    /// Calculate the size of the DD pointed to by e
        unsigned int size(const Edge& e);

	    // reference counting and garbage collection
        void incRef(const Edge &e);
	    void decRef(const Edge &e);
	    void garbageCollect(bool force = false);

	    // checks
	    static inline bool isTerminal(const Edge& e) {
		    return e.p == terminalNode;
	    }
	    static inline bool equals(const Edge& a, const Edge& b) {
		    return a.p == b.p && ComplexNumbers::equals(a.w, b.w);
	    }

	    // Toffoli table insertion and lookup
	    void TTinsert(unsigned short n, unsigned short m, unsigned short t, const short line[MAXN], const Edge& e);
	    void TTinsert(unsigned short n, unsigned short m, unsigned short t, const std::array<short, MAXN>& line, const Edge& e) {
		    TTinsert(n, m, t, line.data(), e);
	    }
	    Edge TTlookup(unsigned short n, unsigned short m, unsigned short t, const short line[MAXN]);
	    Edge TTlookup(unsigned short n, unsigned short m, unsigned short t, const std::array<short, MAXN>& line) {
		    return TTlookup(n, m, t, line.data());
	    }

	    // statistics
        void statistics();
        static void printInformation();

        // printing and GraphViz(dot) export
        void printVector(const Edge& e);
        void printActive(int n);
        void printDD(const Edge& e, unsigned int limit);
        void printUniqueTable(unsigned short n);

        // debugging
        void debugnode(NodePtr p) const;
        std::string debugnode_line(NodePtr p) const;
        bool is_locally_consistent_dd(const Edge& e);
        bool is_locally_consistent_dd2(const Edge& e);
        bool is_globally_consistent_dd(const Edge& e);

        void fill_consistency_counter(const Edge& edge, std::map<ComplexTableEntry*, long>& weight_map, std::map<NodePtr , unsigned long>& node_map);
        void check_consistency_counter(const Edge& edge, const std::map<ComplexTableEntry*, long>& weight_map, const std::map<NodePtr , unsigned long>& node_map);

    };

	inline bool operator==(const Edge& lhs, const Edge& rhs){ return Package::equals(lhs, rhs); }
	inline bool operator!=(const Edge& lhs, const Edge& rhs){ return !(lhs == rhs); }
}
#endif
