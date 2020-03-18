/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_H
#define DDpackage_H

#include <unordered_set>
#include <vector>
#include <array>
#include <bitset>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>

#include "DDcomplex.h"

using CN = dd::ComplexNumbers;

namespace dd {
	const std::string DDversion = "IIC-DD package v1.1";
	constexpr unsigned short RADIX = 2;                     // logic radix
	constexpr unsigned short NEDGE = RADIX * RADIX;   // max no. of edges = RADIX^2

	// General package configuration parameters
	constexpr unsigned int GCLIMIT1 = 250000;                // first garbage collection limit
	constexpr unsigned int GCLIMIT_INC = 0;                  // garbage collection limit increment
	constexpr unsigned int MAXREFCNT = 4000000;     // max reference count (saturates at this value)
	constexpr unsigned int NODECOUNT_BUCKETS = 2000000;
	constexpr unsigned short NBUCKET = 32768;                  // no. of hash table buckets; must be a power of 2
	constexpr unsigned short HASHMASK = NBUCKET - 1;  // must be nbuckets-1
	constexpr unsigned short CTSLOTS = 16384;         // no. of computed table slots
	constexpr unsigned short CTMASK = CTSLOTS - 1;    // must be CTSLOTS-1
	constexpr unsigned short TTSLOTS = 2048;          // Toffoli table slots
	constexpr unsigned short TTMASK = TTSLOTS - 1;    // must be TTSLOTS-1
	constexpr unsigned short CHUNK_SIZE = 2000;
	constexpr unsigned short MAXN = 225;                       // max no. of inputs

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
        ad,
        mult,
        fid,
        transp,
        conjTransp,
        kron,
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

    class Package {

    	static Node terminal;
	    constexpr static Node* terminalNode{&terminal};        // pointer to terminal node


        NodePtr nodeAvail{};                 // pointer to available space chain
	    ListElementPtr listAvail{ };           // pointer to available list elements for breadth first searches

	    // Unique Tables (one per input variable)
	    std::array<std::array<NodePtr, NBUCKET>, MAXN> Unique{ };
	    // Three types since different possibilities for complex numbers  (caused by caching)
	    // weights of operands and result are from complex table (e.g., transpose, conjugateTranspose)
	    std::array<CTentry1, CTSLOTS> CTable1{ };
	    // weights of operands are from complex table, weight of result from cache/ZERO (e.g., mult)
	    std::array<CTentry2, CTSLOTS> CTable2{ };
	    // weights of operands and result are from cache/ZERO (e.g., add)
	    std::array<CTentry3, CTSLOTS> CTable3{ };
	    // Toffoli gate table
	    std::array<TTentry, TTSLOTS> TTable{ };
	    // Identity matrix table
	    std::array<Edge, MAXN> IdTable{ };

	    unsigned int currentNodeGCLimit;              // current garbage collection limit
	    unsigned int currentComplexGCLimit;         // current complex garbage collection limit
	    std::array<int, MAXN> active{ };              // number of active nodes for each variable
	    unsigned long nodecount = 0;                // node count in unique table
	    unsigned long peaknodecount = 0;            // records peak node count in unique table

	    std::array<unsigned long, 7> nOps{};                     // operation counters
	    std::array<unsigned long, 7> CTlook{}, CThit{};      // counters for gathering compute table hit stats
        unsigned long UTcol=0, UTmatch=0, UTlookups=0;  // counter for collisions / matches in hash tables

	    std::vector<ListElementPtr> allocated_list_chunks;
	    std::vector<NodePtr> allocated_node_chunks;

	    bool forceMatrixNormalization = false;

	    /// private helper routines
	    void initComputeTable();
	    NodePtr getNode();

        Edge add2(Edge x, Edge y);
	    Edge multiply2(Edge& x, Edge& y, unsigned short var);
	    ComplexValue fidelity(Edge x, Edge y, int var);
	    Edge trace(Edge a, short v, const std::bitset<MAXN>& eliminate);
	    Edge kronecker2(Edge x, Edge y);

	    void checkSpecialMatrices(Edge &e);
	    Edge UTlookup(Edge& e);
	    Edge CTlookup(const Edge& a, const Edge& b, CTkind which);
	    void CTinsert(const Edge& a, const Edge& b, const Edge& r, CTkind which);

	    static inline unsigned long CThash(const Edge& a, const Edge& b, const CTkind which) {
		    const uintptr_t node_pointer = ((uintptr_t) a.p + (uintptr_t) b.p) >> 3u;
		    const uintptr_t weights = (uintptr_t) a.w.i + (uintptr_t) a.w.r + (uintptr_t) b.w.i + (uintptr_t) b.w.r;
		    return (node_pointer + weights + (uintptr_t) which) & CTMASK;
	    }

	    static inline unsigned long CThash2(NodePtr a, const ComplexValue& aw, NodePtr b, const ComplexValue& bw, const CTkind which) {
		    const uintptr_t node_pointer = ((uintptr_t) a + (uintptr_t) b) >> 3u;
		    const uintptr_t weights = (uintptr_t) (aw.r * 1000) + (uintptr_t) (aw.i * 2000) + (uintptr_t) (bw.r * 3000) + (uintptr_t) (bw.i * 4000);
		    return (node_pointer + weights + (uintptr_t) which) & CTMASK;
	    }
	    static unsigned short TThash(unsigned short n, unsigned short t, const short line[]);

	    unsigned int nodeCount(Edge e, std::unordered_set<NodePtr>& visited) const;
	    ComplexValue getVectorElement(Edge e, unsigned long long int element);
	    ListElementPtr newListElement();
	    void toDot(Edge e, std::ostream& oss, bool isVector = false);

    public:
        constexpr static Edge DDone{ terminalNode, ComplexNumbers::ONE };
        constexpr static Edge DDzero{ terminalNode, ComplexNumbers::ZERO };            // edges pointing to zero and one DD constants
        unsigned long activeNodeCount = 0;             // number of active nodes
		unsigned long maxActive = 0;
        unsigned long gc_calls{};
        unsigned long gc_runs{};
        ComplexNumbers cn;
        std::array<unsigned short, MAXN> varOrder{ };    // variable order initially 0,1,... from bottom up | Usage: varOrder[level] := varible at a certain level
        std::array<unsigned short, MAXN> invVarOrder{ };// inverse of variable order (inverse permutation) | Usage: invVarOrder[variable] := level of a certain variable



        Package();
        ~Package();

        void useMatrixNormalization(bool use) { forceMatrixNormalization = use; }

        // DD creation
        static inline Edge makeTerminal(const Complex& w) { return { terminalNode, w }; }

	    Edge makeNonterminal(short v, const Edge *edge, bool cached = false);

	    inline Edge makeNonterminal(const short v, const std::array<Edge, NEDGE>& edge, bool cached = false) {
	    	return makeNonterminal(v, edge.data(), cached);
	    };
	    Edge makeZeroState(unsigned short n);
	    Edge makeBasisState(unsigned short n, const std::bitset<64>& state);
	    Edge makeIdent(short x, short y);
	    Edge makeGateDD(const Matrix2x2& mat, unsigned short n, const short *line);
	    Edge makeGateDD(const std::array<ComplexValue,NEDGE>& mat, unsigned short n, const std::array<short,MAXN>& line);

	    // operations on DDs
	    Edge multiply(Edge x, Edge y);
	    Edge add(Edge x, Edge y);
	    Edge transpose(const Edge& a);
	    Edge conjugateTranspose(Edge a);
	    Edge normalize(Edge& e, bool cached);
	    Edge partialTrace(Edge a, const std::bitset<MAXN>& eliminate);
	    ComplexValue trace(Edge a);
	    fp fidelity(Edge x, Edge y);
	    Edge kronecker(Edge x, Edge y);
	    Edge extend(Edge e, unsigned short h = 0, unsigned short l = 0);

	    unsigned int size(Edge e) const;

		/**
		 * Get a single element of the vector or matrix represented by the dd with root edge e
		 * @param dd package where the dd lives
		 * @param e edge pointing to the root node
		 * @param elements string {0, 1, 2, 3}^n describing which outgoing edge should be followed
		 *                 (for vectors 0 is the 0-successor and 2 is the 1-successor due to the shared representation)
		 *                 If string is longer than required, the additional characters are ignored.
		 * @return the complex value of the specified element
		 */
		ComplexValue getValueByPath(Edge e, std::string elements);

	    // reference counting and garbage collection
	    void incRef(Edge& e);
	    void decRef(Edge& e);
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

	    // printing
	    void printVector(Edge e);
	    void printActive(int n);
	    void printDD(Edge e, unsigned int limit);
	    void export2Dot(Edge basic, const char *outputFilename, bool isVector = false, bool show = true);

	    // statistics and info
	    void statistics();
	    static void printInformation();

	    // debugging - not normally used
	    void debugnode(NodePtr p) const;
	};
}
#endif
