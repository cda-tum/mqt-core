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
	const std::string DDversion = "IIC-DD package v1.1";
	constexpr unsigned short RADIX = 2;                     // logic radix
	constexpr unsigned short NEDGE = RADIX * RADIX;   // max no. of edges = RADIX^2

	// General package configuration parameters
	constexpr unsigned int GCLIMIT1 = 250000;                // first garbage collection limit
	constexpr unsigned int GCLIMIT_INC = 0;                  // garbage collection limit increment
	constexpr unsigned int MAXREFCNT = 4000000;     // max reference count (saturates at this value)
	constexpr unsigned int NODECOUNT_BUCKETS = 200000;
	constexpr unsigned short NBUCKET = 32768;                  // no. of hash table buckets; must be a power of 2
	constexpr unsigned short HASHMASK = NBUCKET - 1;  // must be nbuckets-1
	constexpr unsigned short CTSLOTS = 16384;         // no. of computed table slots
	constexpr unsigned short CTMASK = CTSLOTS - 1;    // must be CTSLOTS-1
	constexpr unsigned short TTSLOTS = 2048;          // Toffoli table slots
	constexpr unsigned short TTMASK = TTSLOTS - 1;    // must be TTSLOTS-1
	constexpr unsigned int NODE_CHUNK_SIZE = 2000;    // this parameter may be increased for larger benchmarks to minimize the number of allocations
	constexpr unsigned int LIST_CHUNK_SIZE = 2000;
	constexpr unsigned short MAXN = 128;                       // max no. of inputs

	enum ComputeMatrixPropertiesMode {
		Disabled, Enabled, Recompute
	};

    typedef struct Node *NodePtr;

    struct Edge {
	    NodePtr p;
	    Complex w;
    };

    struct Node {
	    NodePtr next;         // link for unique table and available space chain
	    Edge e[NEDGE];     // edges out of this node
	    Complex normalizationFactor; // stores normalization factor
	    unsigned int ref;       // reference count
	    short v;        // variable index (nonterminal) value (-1 for terminal)
	    bool ident, symm; // special matrices flags
	    ComputeMatrixPropertiesMode computeMatrixProperties; // indicates whether to compute matrix properties
	    unsigned int reuseCount;
    };

    // list definitions for breadth first traversals (e.g. printing)
    typedef struct ListElement *ListElementPtr;

    struct ListElement {
	    NodePtr p;
	    ListElementPtr next;
	    int w;
    };

	struct NodeSetElement {
		intptr_t id = 0;
		short v = -1;
		NodeSetElement(intptr_t id, short v): id(id), v(v) {};
	};

    // computed table definitions
    // compute table entry kinds
    enum CTkind {
        ad,
        mult,
        fid,
        transp,
        conjTransp,
        kron,
        renormalize,
        noise,
        noNoise,
        none
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

	enum DynamicReorderingStrategy {
		None, Sifting, Random, Window3
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
	    // Three types since different possibilities for complex numbers  (caused by caching)
	    // weights of operands and result are from complex table (e.g., transpose, conjugateTranspose)
		std::array<std::array<CTentry1, CTSLOTS>, static_cast<int>(Mode::ModeCount)> CTable1{};

	    // weights of operands are from complex table, weight of result from cache/ZERO (e.g., mult)
		std::array<std::array<CTentry2, CTSLOTS>, static_cast<int>(Mode::ModeCount)> CTable2{};

	    // weights of operands and result are from cache/ZERO (e.g., add)
		std::array<std::array<CTentry3, CTSLOTS>, static_cast<int>(Mode::ModeCount)> CTable3{};

	    // Toffoli gate table
	    std::array<TTentry, TTSLOTS> TTable{ };
	    // Identity matrix table
	    std::array<Edge, MAXN> IdTable{ };

	    unsigned int currentNodeGCLimit = GCLIMIT1;        // current garbage collection limit
	    unsigned int currentComplexGCLimit = CN::GCLIMIT1; // current complex garbage collection limit
		std::array<int, MAXN> active{ };                   // number of active nodes for each variable
	    unsigned long nodecount = 0;                       // node count in unique table
	    unsigned long peaknodecount = 0;                   // records peak node count in unique table
        // mostly for debugging in reordering

        std::array<unsigned long, 10> nOps{};              // operation counters
	    std::array<unsigned long, 10> CTlook{}, CThit{};   // counters for gathering compute table hit stats

	    std::vector<ListElementPtr> allocated_list_chunks;
	    std::vector<NodePtr> allocated_node_chunks;

		Mode mode{Mode::Vector};
	    std::unordered_set<NodePtr> visited{NODECOUNT_BUCKETS}; // 2e6
	    ComputeMatrixPropertiesMode computeMatrixProperties = Enabled; // enable/disable computation of matrix properties

	    /// private helper routines
	    void initComputeTable();
	    NodePtr getNode();


	    Edge multiply2(Edge& x, Edge& y, unsigned short var);
	    ComplexValue innerProduct(Edge x, Edge y, int var);
	    Edge trace(Edge a, short v, const std::bitset<MAXN>& eliminate);
	    Edge kronecker2(Edge x, Edge y);

	    void checkSpecialMatrices(NodePtr p);

        static uintptr_t UThash(NodePtr p);
	    Edge UTlookup(Edge e, bool keep_node = false);
        std::string UTcheck(Edge e) const;


        Edge UT_update_node(Edge e, std::size_t previous_hash, Edge in);

	    static inline unsigned long CThash(const Edge& a, const Edge& b, CTkind which);
	    static inline unsigned long CThash2(NodePtr a, const ComplexValue& aw, NodePtr b, const ComplexValue& bw, CTkind which);
	    static inline unsigned short TThash(unsigned short n, unsigned short t, const short line[]);

	    unsigned int nodeCount(const Edge& e, std::unordered_set<NodePtr>& v) const;
	    ComplexValue getVectorElement(Edge e, unsigned long long int element);
	    ListElementPtr newListElement();

    public:
        // edges pointing to zero and one DD constants
        constexpr static Edge DDone{ terminalNode, ComplexNumbers::ONE };
        constexpr static Edge DDzero{ terminalNode, ComplexNumbers::ZERO };

        // The following variables are supposed to be read only. Tread carefully if you change them!

        unsigned long long node_allocations = 0;       // Number of nodes allocated by getNode()
        unsigned long activeNodeCount = 0;             // number of active nodes
        unsigned long unnormalizedNodes = 0;           // active nodes that need renormalization
        unsigned int node_substitutions = 0;           // number of nodes substituted during reordering
        unsigned int node_collapses = 0;               // number of nodes collapses during reordering
        unsigned int exchange_base_cases = 0;          // number of nodes substituted during reordering
        unsigned long maxActive = 0;                   // maximum number of active nodes
        unsigned long gc_calls{};                      // number of calls to the garbage collector
        unsigned long gc_runs{};                       // number of times the GC actually ran
        unsigned long UTcol{}, UTmatch{}, UTlookups{}; // counter for collisions / matches in hash tables
        ComplexNumbers cn;                             // instance of the complex number handler
        std::array<unsigned short, MAXN> varOrder{ };  // variable order initially 0,1,... from bottom up | Usage: varOrder[level] := varible at a certain level
        std::array<unsigned short, MAXN> invVarOrder{ };// inverse of variable order (inverse permutation) | Usage: invVarOrder[variable] := level of a certain variable

        Package();
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
	    Edge makeIdent(short x, short y);
	    Edge makeGateDD(const Matrix2x2& mat, unsigned short n, const short *line);
	    Edge makeGateDD(const std::array<ComplexValue,NEDGE>& mat, unsigned short n, const std::array<short,MAXN>& line);

        Edge CTlookup(const Edge& a, const Edge& b, CTkind which);
        void CTinsert(const Edge& a, const Edge& b, const Edge& r, CTkind which);

	    // operations on DDs
	    Edge multiply(Edge x, Edge y);
	    Edge add(Edge x, Edge y);
        Edge add2(Edge x, Edge y);
	    Edge transpose(const Edge& a);
	    Edge conjugateTranspose(Edge a);
	    Edge normalize(Edge& e, bool cached);
	    Edge partialTrace(Edge a, const std::bitset<MAXN>& eliminate);
	    ComplexValue trace(Edge a);
		ComplexValue innerProduct(Edge x, Edge y);
	    fp fidelity(Edge x, Edge y);
	    Edge kronecker(Edge x, Edge y);
	    Edge extend(Edge e, unsigned short h = 0, unsigned short l = 0);

		// functions for dynamic reordering
		void recomputeMatrixProperties(Edge in);
		void markForMatrixPropertyRecomputation(Edge in);
		void resetNormalizationFactor(Edge in, Complex defaultValue);
		Edge renormalize(Edge in);
		Edge renormalize2(Edge in);
	    void reuseNonterminal(short v, const Edge *edge, NodePtr p, Edge in);
	    void exchange(unsigned short i, unsigned short j);
        dd::Edge exchange2(unsigned short i, unsigned short j, std::map<unsigned short, unsigned short> &varMap, Edge in);
	    void exchangeBaseCase(unsigned short i, Edge in);
	    void exchangeBaseCase2(NodePtr p, unsigned short index, Edge in);
	    Edge dynamicReorder(Edge in, std::map<unsigned short, unsigned short>& varMap, DynamicReorderingStrategy strat = None);
	    std::tuple<Edge, unsigned int, unsigned int> sifting(Edge in, std::map<unsigned short, unsigned short>& varMap);
		Edge random(Edge in, std::map<unsigned short, unsigned short> &varMap, std::mt19937_64 &mt);
		Edge window3(Edge in, std::map<unsigned short, unsigned short>& varMap);


		// utility
        /// Traverse DD and return product of edge weights along the way
		ComplexValue getValueByPath(Edge e, std::string elements);
        /// Calculate the size of the DD pointed to by e
        unsigned int size(const Edge& e);

	    // reference counting and garbage collection
        void incRef(Edge &e);
	    void decRef(Edge &e);
	    void garbageCollect(bool force = false);

	    // checks
	    static inline bool isTerminal(const Edge& e) {
		    return e.p == terminalNode;
	    }
	    static inline bool equals(const Edge& a, const Edge& b) {
		    return a.p == b.p && ComplexNumbers::equals(a.w, b.w);
	    }

	    // Toffoli table insertion and lookup
	    void TTinsert(unsigned short n, unsigned short m, unsigned short t, const short line[], const Edge& e);
	    Edge TTlookup(unsigned short n, unsigned short m, unsigned short t, const short line[]);

	    // statistics
        void statistics();
        static void printInformation();

        // printing and GraphViz(dot) export
        void printVector(const Edge& e);
        void printActive(int n);
        void printDD(const Edge& e, unsigned int limit);
        void printUniqueTable(unsigned short n);

        void toDot(Edge e, std::ostream& oss, bool isVector = false);
        void export2Dot(Edge basic, const std::string& outputFilename, bool isVector = false, bool show = true);

        // debugging
        void debugnode(NodePtr p) const;
        std::string debugnode_line(NodePtr p) const;
        bool is_locally_consistent_dd(Edge e);
        bool is_locally_consistent_dd2(Edge e);
        bool is_globally_consistent_dd(Edge e);

        void fill_consistency_counter(Edge edge, std::map<ComplexTableEntry*, long>& weight_map, std::map<NodePtr , unsigned long>& node_map);
        void check_consistency_counter(Edge edge, const std::map<ComplexTableEntry*, long>& weight_map, const std::map<NodePtr , unsigned long>& node_map);

        void substitute_node(short index, Edge original, Edge substitute, Edge in);
        void substitute_node_ut(short index, Edge original, Edge substitute, Edge in);
        void substitute_node_dd(short index, Edge parent, Edge original, Edge substitute, Edge in);

        void check_node_is_really_gone(NodePtr pNode, Edge in);
    };
}
#endif
