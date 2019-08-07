/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */


#ifndef DDpackage_H
#define DDpackage_H

#include "DDcomplex.h"

namespace dd_package {
    const std::string DDversion = "IIC-DD package v1.0";

    constexpr int MAXSTRLEN = 23;
    constexpr int MAXN = 300;                       // max no. of inputs
    constexpr int MAXRADIX = 2;                     // max logic radix
    constexpr int MAXNEDGE = MAXRADIX * MAXRADIX;   // max no. of edges = MAXRADIX^2
    constexpr int GCLIMIT1 = 250000;                // first garbage collection limit
    constexpr int GCLIMIT_INC = 0;                  // garbage collection limit increment
                                                    // added to garbage collection limit after each collection
    constexpr int MAXND = 4;                        // max n for display purposes
    constexpr int MAXDIM = 16;                      // max dimension of matrix for printing, (should be 2^MAXND)
    constexpr int NBUCKET = 32768;                  // no. of hash table buckets; must be a power of 2
    constexpr unsigned int HASHMASK = NBUCKET - 1;  // must be nbuckets-1
    constexpr unsigned int CTSLOTS = 16384;         // no. of computed table slots
    constexpr unsigned int CTMASK = CTSLOTS - 1;    // must be CTSLOTS-1
    constexpr unsigned int TTSLOTS = 2048;          // Toffoli table slots
    constexpr unsigned int TTMASK = TTSLOTS - 1;    // must be TTSLOTS-1
    constexpr unsigned int MAXREFCNT = 4000000;     // max reference count (saturates at this value)
    constexpr unsigned int MAXPL = 65536;           // max size for a permutation recording

    typedef struct DDnode *DDnodePtr;

    struct DDedge {
        DDnodePtr p;
        complex w;
    };

    struct DDnode {
        DDnodePtr next;         // link for unique table and available space chain
        unsigned int ref;       // reference count
        unsigned char v;        // variable index (nonterminal) value (-1 for terminal)
        complex renormFactor;   // factor that records renormalization factor
        char ident, diag, block, symm, c01; // flag to mark if vertex heads a QMDD for a special matrix
        char computeSpecialMatricesFlag;    // flag to mark whether SpecialMatrices are to be computed
        DDedge e[MAXNEDGE];     // edges out of this node
    };

// list definitions for breadth first traversals (e.g. printing)  
    typedef struct ListElement *ListElementPtr;

    struct ListElement {
        int w, cnt;
        int line[MAXN];
        DDnodePtr p;
        ListElementPtr next;
    };

    // computed table definitions
    // compute table entry kinds
    enum CTkind {
        add,
        mult,
        kronecker,
        fidelity,
        reduce,
        transpose,
        conjugateTranspose,
        transform,
        c0,
        c1,
        c2,
        none,
        norm,
        createHdmSign,
        findCmnSign,
        findBin,
        reduceHdm,
        renormalize
    };

    //computed table entry
    struct CTentry1// computed table entry defn
    {
        DDedge a, b, r;     // a and b are arguments, r is the result
        CTkind which;       // type of operation
    };

    struct CTentry2// computed table entry defn
    {
        DDedge a, b;     // a and b are arguments, r is the result
        DDnodePtr r;
        complex_value rw;
        CTkind which;       // type of operation
    };

    struct CTentry3// computed table entry defn
    {
        DDnodePtr a, b, r;     // a and b are arguments, r is the result
        complex_value aw, bw, rw;
        CTkind which;       // type of operation
    };


    struct TTentry // Toffoli table entry defn
    {
        int n, m, t, line[MAXN];
        DDedge e;
    };

    struct CircuitLine {
        char input[MAXSTRLEN];
        char output[MAXSTRLEN];
        char variable[MAXSTRLEN];
        char ancillary;
        char garbage;

    };

    struct DDrevlibDescription // circuit description structure
    {
        int n, ngates, qcost, nancillary, ngarbage;
        DDedge e, totalDC;
        CircuitLine line[MAXN];
        char version[MAXSTRLEN];
        char inperm[MAXN], outperm[MAXN];
        char ngate, cgate, tgate, fgate, pgate, vgate, kind[7], dc[5], name[32], no[8], modified;
    };

    typedef complex_value DD_matrix[MAXRADIX][MAXRADIX];

    // matrices for different operations
    extern const DD_matrix Nm, Sm, Hm, Zm;

    // Global Variables
    extern const int Radix;                 // radix (default is 2)
    extern const int Nedge;                 // no. of edges (default is 4)
    extern DDnodePtr Avail;                 // pointer to available space chain
    extern ListElementPtr Lavail;           // pointer to available list elements for breadth first searchess
    extern DDnodePtr DDterminalNode;        // pointer to terminal node
    extern DDedge DDone, DDzero;            // edges pointing to zero and complex_one DD constants
    extern long DDorder[MAXN];              // variable order initially 0,1,... from bottom up | Usage: DDorder[level] := varible at a certain level
    extern long DDinverseOrder[MAXN];       // inverse of variable order (inverse permutation) | Usage: DDinverseOrder[variable] := level of a certain variable
    extern long DDnodecount;                // counts active nodes
    extern long DDpeaknodecount;            // records peak node count in unique table
    extern long Nop[6];                     // operation counters
    extern long CTlook[20], CThit[20];      // counters for gathering compute table hit stats
    extern long UTcol, UTmatch, UTlookups;  // counter for collisions / matches in hash tables
    extern int GCcurrentLimit;              // current garbage collection limit
    extern int ComplexCurrentLimit;         // current garbage collection limit
    extern int ActiveNodeCount;             // number of active nodes
    extern int Active[MAXN];                // number of active nodes for each variable
    extern int GCswitch;                    // set switch to 1 to enable garbage collection
    extern int Smode;                       // S mode switch for spectral transformation; Smode==1 0->+1 1->-1; Smode==0 0->0 1->1
    extern int RMmode;                      // Select RM transformation mode forces mod Radix arithmetic
    extern int MultMode;                    // set to 1 for matrix - vector multiplication
    extern DDedge DDnullEdge;               // set in DDinit routine
    extern int PermList[MAXPL];             // array for recording a permutation
    extern int RenormalizationNodeCount;    // number of active nodes that need renormalization (used in DDdecRef)
    extern int blockMatrixCounter;          // number of active nodes that represent block matrices (used in DDincRef, DDdecRef)
    extern int globalComputeSpecialMatricesFlag; // default value for computeSpecialMatricesFlag of newly created nodes (used in DDmakeNonterminal)

    // Unique Tables (one per input variable)
    extern DDnodePtr Unique[MAXN][NBUCKET];

    // Three types since different possibilities for complex numbers  (caused by caching)
    // weights of operands and result are from complex table (e.g., transpose, conjugateTranspose)
    extern CTentry1 CTable1[CTSLOTS];
    // weights of operands are from complex table, weight of result from cache/COMPLEX_ZERO (e.g., mult)
    extern CTentry2 CTable2[CTSLOTS];
    // weights of operands and result are from cache/COMPLEX_ZERO (e.g., add)
    extern CTentry3 CTable3[CTSLOTS];

    // Toffoli gate table
    extern TTentry TTable[TTSLOTS];

    // Identity matrix table
    extern DDedge DDid[MAXN];

    // Variable labels
    extern int Nlabel;        // number of labels
    extern char Label[MAXN][MAXSTRLEN];  // label table


    // checks if an edge points to the terminal node
    inline bool DDterminal(const DDedge e) {
        return e.p == DDterminalNode;
    }

    // checks if two edges are equal
    inline bool DDedgeEqual(const DDedge a, const DDedge b) {
        return a.p==b.p && a.w==b.w;
    }

    void DDdebugnode(DDnodePtr); // for debugging purposes - not normally used
    ListElementPtr DDnewListElement();

    void DDprint(DDedge, unsigned int);

    void DD2dot(DDedge, std::ostream &, DDrevlibDescription);

    DDedge DDmultiply(DDedge, DDedge);

    DDedge DDadd(DDedge, DDedge);

    void DDdecRef(DDedge);

    void DDincRef(DDedge);

    DDedge DDident(int, int);

    DDedge DDmvlgate(const DD_matrix, int, const int[]);

    void TTinsert(int, int, int, const int[], DDedge);

    DDedge TTlookup(int, int, int, const int[]);

    void DDgarbageCollect();

    DDedge DDtranspose(DDedge);

    void DDmatrixPrint2(DDedge);

    DDedge DDnormalize(DDedge, bool);

    void DDcheckSpecialMatrices(DDedge);

    DDedge CTlookup(DDedge, DDedge, CTkind);

    void CTinsert(DDedge, DDedge, DDedge, CTkind);

    void DDinitComputeTable();

    DDedge DDutLookup(DDedge);

    DDedge DDmakeNonterminal(short, DDedge[MAXNEDGE], bool);

    DDedge DDmakeTerminal(complex w);

    void DDinit(bool verbose);

    void DDdotExport(DDedge basic, const char *outputFilename, DDrevlibDescription circ, bool show);

    void DDdotExportVector(DDedge basic, const char *outputFilename);

    void DDdotExportMatrix(DDedge basic, const char *outputFilename);

    void DDstatistics();

    void DDprintActive(int n);

    DDedge DDzeroState(int n);

    void DDprintVector(DDedge e);

    unsigned int DDsize(DDedge e);

    long double DDfidelity(DDedge x, DDedge y);

    int CacheSize();
}
#endif
