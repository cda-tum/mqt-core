/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */


#include "DDpackage.h"
#include <limits>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>

namespace dd_package {
    constexpr unsigned int NODECOUNT_BUCKETS = 2000000;
    const complex_value complex_one = {1, 0};
    const complex_value complex_zero = {0, 0};
    const long double sqrt_2 = 1.0L / std::sqrt(2.0L);

    // NOT operations
    const DD_matrix Nm = {{complex_zero, complex_one},
                          {complex_one,  complex_zero}};
    // phase shift S
    const DD_matrix Sm = {{complex_one,  complex_zero},
                          {complex_zero, {0, 1}}};
    // Hadamard
    const DD_matrix Hm = {{{sqrt_2, 0}, {sqrt_2,  0}},
                          {{sqrt_2, 0}, {-sqrt_2, 0}}};
    // phase shift Z = S*S
    const DD_matrix Zm = {{complex_one,  complex_zero},
                          {complex_zero, {-1, 0}}};

    const int Radix = MAXRADIX;                 // radix (default is 2)
    const int Nedge = MAXNEDGE;                 // no. of edges (default is 4)
    int GCswitch = 1;                           // set switch to 1 to enable garbage collection
    int Smode = 1;                              // S mode switch for spectral transformation
    int RMmode = 0;                             // Select RM transformation mode
    int MultMode = 0;                           // set to 1 for matrix - vector multiplication

    int RenormalizationNodeCount = 0;           // number of active nodes that need renormalization (used in DDdecRef)
    int blockMatrixCounter = 0;                 // number of active nodes that represent block matrices (used in DDincRef, DDdecRef)
    int globalComputeSpecialMatricesFlag = 1;   // default value for computeSpecialMatricesFlag of newly created nodes (used in DDmakeNonterminal)

    DDnodePtr Avail;                            // pointer to available space chain
    ListElementPtr Lavail;                      // pointer to available list elements for breadth first searchess
    DDnodePtr DDterminalNode;                   // pointer to terminal node
    DDedge DDone, DDzero;                       // edges pointing to zero and complex_one DD constants
    long DDorder[MAXN];                         // variable order initially 0,1,... from bottom up | Usage: DDorder[level] := varible at a certain level
    long DDinverseOrder[MAXN];                  // inverse of variable order (inverse permutation) | Usage: DDinverseOrder[variable] := level of a certain variable
    long DDnodecount;                           // counts active nodes
    long DDpeaknodecount;                       // records peak node count in unique table
    long Nop[6];                                // operation counters
    long CTlook[20], CThit[20];                 // counters for gathering compute table hit stats
    long UTcol, UTmatch, UTlookups;             // counter for collisions / matches in hash tables
    int GCcurrentLimit;                         // current garbage collection limit
    int ComplexCurrentLimit;                    // current garbage collection limit
    int ActiveNodeCount;                        // number of active nodes
    int Active[MAXN];                           // number of active nodes for each variable
    DDedge DDnullEdge;                          // set in DDinit routine
    int PermList[MAXPL];                        // array for recording a permutation
    DDnodePtr Unique[MAXN][NBUCKET];
    CTentry1 CTable1[CTSLOTS];
    CTentry2 CTable2[CTSLOTS];
    CTentry3 CTable3[CTSLOTS];
    TTentry TTable[TTSLOTS];
    DDedge DDid[MAXN];
    int Nlabel;        // number of labels
    char Label[MAXN][MAXSTRLEN];  // label table

    int CacheSize() {
        complex_table_entry *p = ComplexCache_Avail;
        int size = 0;

        intptr_t min = std::numeric_limits<intptr_t>::max();
        intptr_t max = std::numeric_limits<intptr_t>::min();

        while (p != nullptr && size <= 1800) {
            if (p->ref != 0) {
                std::cerr << "Entry with refcount != 0 in Cache!\n";
                std::cerr << (intptr_t) p << " " << p->ref << " " << p->val << " " << (intptr_t) p->next << "\n";
            }
            if (((intptr_t) p) < min) {min = (intptr_t) p;}
            if (((intptr_t) p) > max) {max = (intptr_t) p;}

            p = p->next;
            size++;
        }
        if (size > 1800) {
            p = ComplexCache_Avail;
            for (int i = 0; i < 10; i++) {
                std::cout << i << ": " << (uintptr_t) p << "\n";
                p = p->next;
            }
            std::cerr << "Error in Cache!\n" << std::flush;
            std::exit(1);
        }
        std::cout << "Min ptr in cache: " << min << ", max ptr in cache: " << max << "\n";
        return size;
    }

    // for debugging purposes - not normally used
    void DDdebugnode(DDnodePtr p) {
       if (p == DDzero.p) {
            std::cout << "terminal\n";
            return;
        }
        std::cout  <<"Debug node" << (intptr_t) p << "\n";
        std::cout << "node v " << (int) DDorder[p->v] <<" (" << (int) p->v << ") edges (w,p) ";
        for (auto const & i : p->e) {
            std::cout << i.w << " " << (intptr_t) i.p <<" || ";
        }
        std::cout << "ref " << p->ref << "\n";
    }

    ListElementPtr DDnewListElement() {
        ListElementPtr r;

        if (Lavail != nullptr) {   // get node from avail chain if possible
            r = Lavail;
            Lavail = Lavail->next;
        } else {            // otherwise allocate 2000 new nodes
            r = new ListElement[2000];
            ListElementPtr r2 = r + 1;
            Lavail = r2;
            for (int i = 0; i < 1998; i++, r2++) {
                r2->next = r2+1;
            }
            r2->next = nullptr;
        }
        return r;
    }

    // a slightly better DD print utility
    void DDprint(DDedge e, unsigned int limit) {
        ListElementPtr first, q, lastq, pnext;
        uint64_t n, i, j;

        first = DDnewListElement();
        first->p = e.p;
        first->next = nullptr;

        first->w = 0;
        first->cnt = 1;
        n = 0;
        i = 0;
        std::cout << "top edge weight " << e.w << "\n";
        pnext = first;

        while (pnext != nullptr) {
            std::cout << pnext->cnt << " " << pnext->p->ref;
            std::cout << (pnext->p->block ? 'B' : ' ');
            std::cout << (pnext->p->diag ? 'D' : ' ');
            std::cout << (pnext->p->ident ? 'I' : ' ');
            std::cout << (pnext->p->symm ? 'S' : ' ');

            if (pnext->p->renormFactor != COMPLEX_ONE)
                std::cout << "R=" << pnext->p->renormFactor;
            else
                std::cout << "    ";
            std::cout << i << "|  (" << pnext->p->v << ")";

            std::cout << "[";
            if (pnext->p != DDzero.p) {
                for (j = 0; j < Nedge; j++) {
                    if (pnext->p->e[j].p == nullptr) {
                        std::cout << "NULL ";
                    } else {
                        if (!DDterminal(pnext->p->e[j])) {
                            q = first->next;
                            lastq = first;
                            while (q != nullptr && pnext->p->e[j].p != q->p) {
                                lastq = q;
                                q = q->next;
                            }
                            if (q == nullptr) {
                                q = DDnewListElement();
                                q->p = pnext->p->e[j].p;
                                q->next = nullptr;
                                q->w = n = n + 1;
                                q->cnt = 1;
                                lastq->next = q;
                            } else
                                q->cnt = q->cnt + 1;
                            std::cout << " " << q->w << ":";
                        } else {
                            std::cout << "   T:";
                        }
                        std::cout << " (" << pnext->p->e[j].w << ") ";
                    }
                }
            }

            std::cout << "] " << (intptr_t) pnext->p << "\n";
            i++;
            if (i == limit) {
                std::cout << "Printing terminated after " << limit << " vertices\n";
                return;
            }
            pnext = pnext->next;
        }
    }

    complex_value GetElementOfVector(DDedge e, unsigned long long element) {
        if (DDterminal(e)) {
            complex_value ret;
            ret.r = ret.i = 0;
            return ret;
        }
        complex l;
        l.r = ComplexCache_Avail;
        l.i = l.r->next;
        l.r->val = 1;
        l.i->val = 0;
        do {
            Cmul(l, l, e.w);
            long tmp = (element >> DDinverseOrder[e.p->v]) & 1;
            e = e.p->e[2 * tmp];
        } while (!DDterminal(e));
        Cmul(l, l, e.w);
        complex_value ret;
        ret.r = l.r->val;
        ret.i = l.i->val;

        return ret;
    }


    void DDprintVector(DDedge e) {
        unsigned long long element = 2 << DDinverseOrder[e.p->v];
        for (unsigned long long i = 0; i < element; i++) {
            complex_value amplitude = GetElementOfVector(e, i);
            for (int j = DDinverseOrder[e.p->v]; j >= 0; j--) {
                std::cout << ((i >> j) & 1u);
            }
            std::cout << ": " << amplitude << "\n";

        }
        std::cout << std::flush;
    }


    /*
     * DD2dot export. Nodes representing special matrices (symmetric/identity) are coloured green/red.
     */
    void DD2dot(DDedge e, std::ostream &oss, DDrevlibDescription circ) {
        /* first part of dot file*/
        std::ostringstream nodes;
        /*second part of dot file*/
        std::ostringstream edges;

        edges << "\n";
        /*Initialize Graph*/
        nodes << "digraph \"DD\" {\n"
              << "graph [center=true, ordering=out];\n"
              << "node [shape=circle, center=true];\n"
              << "\"T\" [ shape = box, label=\"1\" ];\n";
        /* Define Nodes */
        ListElementPtr first, q, lastq, pnext;
        first = DDnewListElement();
        first->p = e.p;
        first->next = nullptr;
        first->w = 0;
        first->cnt = 1;

        uint64_t n=0, i=0;

        nodes << "\"R\"";
        //füge Kante zwischen helper node und neuem Knoten hinzu
        if (e.w == COMPLEX_ONE) {
            nodes << " [label=\"\", shape=point];\n";
            edges << "\"R\" -> \"0\"\n";
        } else {
            nodes << " [label=\"\", shape=point];\n";
            edges << "\"R\" -> \"0\" [label=\"(" << e.w << ")\" ];\n";
        }


        pnext = first;
        while (pnext != nullptr) {
            /* Zeichne Knoten*/
            if (pnext->p->ident)
                nodes << "\"" << i << "\" " << "[ label=\""
                      << circ.line[((int) pnext->p->v)].variable
                      << "\" ,style=filled, fillcolor=red ];\n";
            else if (pnext->p->symm)
                nodes << "\"" << i << "\" " << "[ label=\""
                      << circ.line[((int) pnext->p->v)].variable
                      << "\" ,style=filled, fillcolor=green ];\n";
            else
                nodes << "\"" << i << "\" " << "[ label=\""
                      << circ.line[((int) pnext->p->v)].variable
                      << "\" ,style=filled, fillcolor=lightgray ];\n";

            if (pnext->p != DDzero.p) {
                edges << "{rank=same;";
                for (unsigned int k = 0; k < MAXNEDGE; k++) {
                    if (MultMode == 1) {
                        if (k % MAXRADIX != 0) continue;
                    }
                    edges << " \"" << i << "h" << k << "\"";
                }
                edges << "}\n";

                for (int j = 0; j < Nedge; j++) {
                    if (MultMode == 1) {
                        if (j % MAXRADIX != 0) {
                            continue;
                        }
                    }
                    if (pnext->p->e[j].p == nullptr);
                    else {
                        if (!DDterminal(pnext->p->e[j])) {
                            q = first->next;
                            lastq = first;
                            while (q != nullptr && pnext->p->e[j].p != q->p) {
                                lastq = q;
                                q = q->next;
                            }
                            if (q == nullptr) {
                                q = DDnewListElement();
                                q->p = pnext->p->e[j].p;
                                q->next = nullptr;
                                q->w = n = n + 1;
                                q->cnt = 1;
                                lastq->next = q;
                            } else {
                                q->cnt = q->cnt + 1;
                            }
                            nodes << "\"" << i << "h" << j << "\" ";

                            //connect helper node

                            edges << "\"" << i << "\" -> \"" << i << "h" << j << "\" [arrowhead=none";

                            switch (j) {
                                case 0:
                                    edges << ",color=darkgreen";
                                    break;
                                case 1:
                                    edges << ",color=blue";
                                    break;
                                case 2:
                                    edges << ",color=red";
                                    break;
                                case 3:
                                    edges << ",color=gold";
                                    break;
                                default:
                                    break;
                            }
                            edges << "];\n";
                            //füge Kante zwischen helper node und neuem Knoten hinzu
                            if (pnext->p->e[j].w == COMPLEX_ONE) {
                                nodes << " [label=\"\", shape=point];\n";
                                edges << "\"" << i << "h" << j << "\" -> \"" << q->w << "\";\n";
                            } else {
                                nodes << " [label=\"\", shape=point];\n";
                                edges << "\"" << i << "h" << j << "\" -> \"" << q->w
                                      << "\" [label=\" (" << pnext->p->e[j].w << ")\" ];\n";
                            }

                        } else {
                            nodes << "\"" << i << "h" << j << "\" " << " [label=\"\", shape=point ";
                            edges << "\"" << i << "\" -> \"" << i << "h" << j << "\" [arrowhead=none";
                            switch (j) {
                                case 0:
                                    edges << ",color=darkgreen";
                                    break;
                                case 1:
                                    edges << ",color=blue";
                                    break;
                                case 2:
                                    edges << ",color=red";
                                    break;
                                case 3:
                                    edges << ",color=gold";
                                    break;
                                default:
                                    break;
                            }
                            edges << "];\n";
                            //connect helper node
                            if (pnext->p->e[j].w == COMPLEX_ZERO) {
                                nodes << ", fillcolor=red, color=red";
                            } else if (pnext->p->e[j].w == COMPLEX_ONE) {
                                edges << "\"" << i << "h" << j << "\"-> \"T\";\n";
                            } else {
                                edges << "\"" << i << "h" << j << "\"-> \"T\" [label= \"(" << pnext->p->e[j].w << ")\", ];\n";
                            }
                            nodes << "];\n";

                        }
                    }
                }
            }
            i++;
            pnext = pnext->next;
        }
        oss << nodes.str() << edges.str() << "\n}\n" << std::flush;
    }

    // export a DD in .dot format to the file specified by outputFilename
    // and call DOT->SVG export (optional, requires dot package)
    void DDdotExport(DDedge basic, const char *outputFilename, DDrevlibDescription circ, bool show) {
        std::ofstream init(outputFilename);
        DD2dot(basic, init, circ);
        init.close();

        if (show) {
            std::ostringstream oss;
            oss << "dot -Tsvg " << outputFilename << " -o " << outputFilename << ".svg";
            auto str = oss.str(); // required to avoid immediate deallocation of temporary
            std::system(str.c_str());
        }
    }

    void DDdotExportVector(DDedge basic, const char *outputFilename) {
        DDrevlibDescription circ{};

        for (long int i = 0; i <= DDinverseOrder[basic.p->v]; i++) {
            snprintf(circ.line[i].variable, MAXSTRLEN, "x[%ld]", DDinverseOrder[basic.p->v] - i);
        }
        int MultMode_old = MultMode;
        MultMode = 1;

        DDdotExport(basic, outputFilename, circ, true);

        MultMode = MultMode_old;
    }

    void DDdotExportMatrix(DDedge basic, const char *outputFilename) {
        DDrevlibDescription circ{};

        for (long i = 0; i <= DDinverseOrder[basic.p->v]; i++) {
            snprintf(circ.line[i].variable, MAXSTRLEN, "x[%ld]", DDinverseOrder[basic.p->v] - i);
        }
        int MultMode_old = MultMode;
        MultMode = 0;

        DDdotExport(basic, outputFilename, circ, true);

        MultMode = MultMode_old;
    }

    DDedge DDzeroState(int n) {
        DDedge f = DDone;
        DDedge edges[4];
        edges[1] = edges[2] = edges[3] = DDzero;

        for (int p = 0; p < n; p++) {
            edges[0] = f;
            f = DDmakeNonterminal(p, edges, false);
        }
        return f;
    }

    DDedge DDnormalize(DDedge e, bool cached) {
        int max = -1;
        long double sum = 0.0;
        long double div = 0.0;

        for (int i = 0; i < Nedge; i++) {
            if ((e.p->e[i].p == nullptr || Ceq(e.p->e[i].w, COMPLEX_ZERO))) {
                continue;
            }

            if (max == -1) {
                max = i;
                sum = div = CmagSquared(e.p->e[i].w);
            } else {
                sum += CmagSquared(e.p->e[i].w);
            }
        }
        if (max == -1) {
            if (cached) {
                for (auto & i : e.p->e) {
                    if (i.p != nullptr && i.w != COMPLEX_ZERO) {
                        i.w.i->next = ComplexCache_Avail;
                        ComplexCache_Avail = i.w.r;
                    }
                }
            }
            return DDzero;
        }

        if (e.p->e[1].w != COMPLEX_ZERO || e.p->e[3].w != COMPLEX_ZERO) {
            sum /= 2;
        }

        sum = std::sqrt(sum / div);

        if (cached && e.p->e[max].w != COMPLEX_ONE) {
            e.w = e.p->e[max].w;
            e.w.r->val *= sum;
            e.w.i->val *= sum;
        } else {
            complex c;
            c.r = ComplexCache_Avail;
            c.i = ComplexCache_Avail->next;
            c.r->val = CVAL(e.p->e[max].w.r) * sum;
            c.i->val = CVAL(e.p->e[max].w.i) * sum;
            e.w = Clookup(c);
            if (e.w == COMPLEX_ZERO) {
                return DDzero;
            }
        }

        for (int j = 0; j < Nedge; j++) {
            if (max == j) {
                complex c;
                c.r = ComplexCache_Avail;
                c.i = ComplexCache_Avail->next;
                c.r->val = 1.0 / sum;
                c.i->val = 0.0;
                e.p->e[j].w = Clookup(c);
                if (e.p->e[j].w == COMPLEX_ZERO) {
                    e.p->e[j] = DDzero;
                }
            } else if (e.p->e[j].p != nullptr && e.p->e[j].w != COMPLEX_ZERO) {
                if (cached) {
                    e.p->e[j].w.i->next = ComplexCache_Avail;
                    ComplexCache_Avail = e.p->e[j].w.r;
                    Cdiv(e.p->e[j].w, e.p->e[j].w, e.w);
                    e.p->e[j].w = Clookup(e.p->e[j].w);
                    if (e.p->e[j].w == COMPLEX_ZERO) {
                        e.p->e[j] = DDzero;
                    }
                } else {
                    complex c;
                    c.r = ComplexCache_Avail;
                    c.i = ComplexCache_Avail->next;
                    Cdiv(c, e.p->e[j].w, e.w);
                    e.p->e[j].w = Clookup(c);
                    if (e.p->e[j].w == COMPLEX_ZERO) {
                        e.p->e[j] = DDzero;
                    }
                }
            }
        }
        return e;
    }

    //  check if e points to a block, identity, diagonal, symmetric or 0/1-matrix and
    //  marks top node if it does
    void DDcheckSpecialMatrices(DDedge e) {
        // only perform checks if flag is set
        if (!e.p->computeSpecialMatricesFlag)
            return;

        e.p->ident = 0;       // assume not identity
        e.p->diag = 0;           // assume not diagonal
        e.p->block = 0;           // assume not block
        e.p->symm = 1;           // assume symmetric
        e.p->c01 = 1;           // assume 0/1-matrix

        /****************** CHECK IF 0-1 MATRIX ***********************/

        for (int i = 0; i < Nedge; i++) {  // check if 0-1 matrix
            if ((e.p->e[i].w != COMPLEX_ONE && e.p->e[i].w != COMPLEX_ZERO) || (!e.p->e[i].p->c01)) {
                e.p->c01 = 0;
                break;
            }
        }
        /****************** CHECK IF Symmetric MATRIX *****************/

        for (int i = 0; i < Radix; i++) {  // check if symmetric matrix (on diagonal)
            if (!(e.p->e[Radix * i + i].p->symm)) {
                e.p->symm = 0;
                break;
            }
        }

        for (int i = 0; e.p->symm && i < Radix - 1; i++) { // check off diagonal entries for transpose properties
            for (int j = i + 1; j < Radix; j++) {
                DDedge t = DDtranspose(e.p->e[i * Radix + j]);
                if (!DDedgeEqual(t, e.p->e[j * Radix + i])) {
                    e.p->symm = 0;
                    break;
                }
            }
        }

        int w = DDinverseOrder[e.p->v];
        if (w != 0) {
            w = DDorder[w - 1];
        }
        // w:= variable complex_one level below current level or 0 if already at the bottom

        /****************** CHECK IF Block MATRIX ***********************/

        for (int i = 0; i < Radix; i++) { // check off diagonal entries
            for (int j = 0; j < Radix; j++) {
                if (e.p->e[i * Radix + j].p == nullptr || (i != j && e.p->e[i * Radix + j].w != COMPLEX_ZERO)) {
                    return;
                }
            }
        }
        e.p->block = 1;

        /****************** CHECK IF Diagonal MATRIX ***********************/
        // will only reach this point if block == 1
        e.p->diag = 1;
        for (int i = 0; i < Radix; i++) { // check diagonal entries to verify matrix is diagonal
            // necessary condition: edge points to a diagonal matrix
            e.p->diag = e.p->e[i * Radix + i].p->diag;
            int j = Radix * i + i;

            // skipped variable: edge pointing to terminal with non-zero weight from level > 0
            if ((DDterminal(e.p->e[j])) && e.p->e[j].w != COMPLEX_ZERO && DDinverseOrder[e.p->v] != 0) {
                e.p->diag = 0;
            }
            // skipped variable: edge pointing to an irregular level (non-terminal)
            if ((!DDterminal(e.p->e[j])) && e.p->e[j].p->v != w) {
                e.p->diag = 0;
            }

            if (!e.p->diag){
                return;
            }
        }

        /****************** CHECK IF Identity MATRIX ***********************/
        // will only reach this point if diag == 1
        for (int i = 0; i < Radix; i++) { // check diagonal entries
            int j = Radix * i + i;
            // if skipped variable, then matrix cannot be diagonal (and we will not reach this point)!
            if (e.p->e[j].w != COMPLEX_ONE || e.p->e[j].p->ident == 0){
                return;
            }
        }
        e.p->ident = 1;
    }

    DDedge DDutLookup(DDedge e) {
    //  lookup a node in the unique table for the appropriate variable - if not found insert it
    //  only normalized nodes shall be stored.
        if (DDterminal(e)) // there is a unique terminal node
        {
            e.p = DDzero.p;
            return e;
        }

        UTlookups++;

        uintptr_t key = 0;
        // note hash function shifts pointer values so that order is important
        // suggested by Dr. Nigel Horspool and helps significantly
        for (unsigned int i = 0; i < Nedge; i++)
            key += ((uintptr_t) (e.p->e[i].p) >> i) + ((uintptr_t) (e.p->e[i].w.r) >> i) +
                   ((uintptr_t) (e.p->e[i].w.i) >> (i + 1));
        key = key & HASHMASK;

        unsigned int v = e.p->v;
        DDnodePtr p = Unique[v][key]; // find pointer to appropriate collision chain
        while (p != nullptr)    // search for a match
        {
            if (memcmp(e.p->e, p->e, Nedge * sizeof(DDedge)) == 0) {
                // Match found
                e.p->next = Avail;    // put node pointed to by e.p on avail chain
                Avail = e.p;

                // NOTE: reference counting is to be adjusted by function invoking the table lookup
                UTmatch++;        // record hash table match

                e.p = p;// and set it to point to node found (with weight unchanged)

                if (p->renormFactor != COMPLEX_ONE) {
                    std::cout << "Debug: table lookup found a node with active renormFactor with v="
                        << p->v << "(id=" << (uintptr_t) p << ").\n";
                    if (p->ref != 0)
                        std::cout << "was active!";
                    else
                        std::cout << "was inactive!";
                    std::exit(1);
                }
                return e;
            }

            UTcol++;        // record hash collision
            p = p->next;
        }
        e.p->next = Unique[v][key]; // if end of chain is reached, this is a new node
        Unique[v][key] = e.p;       // add it to front of collision chain

        DDnodecount++;          // count that it exists
        if (DDnodecount > DDpeaknodecount)
            DDpeaknodecount = DDnodecount;

        if (!DDterminal(e))
            DDcheckSpecialMatrices(e); // check if it is identity or diagonal if nonterminal

        return e;                // and return
    }

    // set compute table to empty and
    // set toffoli gate table to empty and
    // set identity table to empty
    void DDinitComputeTable() {
        for (unsigned int i = 0; i < CTSLOTS; i++) {
            CTable1[i].r.p = nullptr;
            CTable1[i].which = none;
            CTable2[i].r = nullptr;
            CTable2[i].which = none;
            CTable3[i].r = nullptr;
            CTable3[i].which = none;
        }
        for (auto & i : TTable) {
            i.e.p = nullptr;
        }
        for (auto & i : DDid) {
            i.p = nullptr;
        }
        DDnullEdge.p = nullptr;
        DDnullEdge.w = COMPLEX_ZERO;
    }

    void DDgarbageCollect()
// a simple garbage collector that removes nodes with 0 ref count from the unique
// tables placing them on the available space chain
    {
        if (DDnodecount < GCcurrentLimit && ComplexCount < ComplexCurrentLimit)
            return; // do not collect if below GCcurrentLimit node count
        int count = 0;
        int counta = 0;
        for (auto & variable : Unique) {
            for (auto & bucket : variable) {
                DDnodePtr lastp = nullptr;
                DDnodePtr p = bucket;
                while (p != nullptr) {
                    if (p->ref == 0) {
                        if (p == DDterminalNode)
                            std::cerr << "error in garbage collector\n";
                        count++;
                        DDnodePtr nextp = p->next;
                        if (lastp == nullptr)
                            bucket = p->next;
                        else
                            lastp->next = p->next;
                        p->next = Avail;
                        Avail = p;
                        p = nextp;
                    } else {
                        lastp = p;
                        p = p->next;
                        counta++;
                    }
                }
            }
        }
        GCcurrentLimit += GCLIMIT_INC;
        DDnodecount = counta;
        garbageCollectComplexTable();
        DDinitComputeTable(); // IMPORTANT sets compute table to empty after garbage collection
    }

    // get memory space for a node
    DDnodePtr DDgetNode() {
        DDnodePtr r;

        if (Avail != nullptr)    // get node from avail chain if possible
        {
            r = Avail;
            Avail = Avail->next;
        } else {            // otherwise allocate 2000 new nodes
            r = new DDnode[2000];
            DDnodePtr r2 = r+1;
            Avail = r2;
            for (int i = 0; i < 1998; i++, r2++) {
                r2->next = r2+1;
            }
            r2->next = nullptr;
        }
        r->next = nullptr;
        r->ref = 0;            // set reference count to 0
        r->ident = r->diag = r->block = 0;        // mark as not identity or diagonal
        return r;
    }

    // increment reference counter for node e points to
    // and recursively increment reference counter for
    // each child if this is the first reference
    //
    // a ref count saturates and remains unchanged if it has reached
    // MAXREFCNT
    void DDincRef(DDedge e) {
        complexIncRef(e.w);
        if (DDterminal(e))
            return;

        if (e.p->ref == MAXREFCNT) {
            std::cout << "MAXREFCNT reached\n\n\ne.w=" << e.w << "\n";
            DDdebugnode(e.p);
            return;
        }
        e.p->ref++;

        if (e.p->ref == 1) {
            if (!DDterminal(e))
                for (int i = 0; i < Nedge; i++)
                    if (e.p->e[i].p != nullptr) {
                        DDincRef(e.p->e[i]);
                    }
            Active[e.p->v]++;
            ActiveNodeCount++;

            /******* Part added for sifting purposes ********/
            if (e.p->block)
                blockMatrixCounter++;
            /******* by Niemann, November 2012 ********/

        }
    }

    // decrement reference counter for node e points to
    // and recursively decrement reference counter for
    // each child if this is the last reference
    //
    // a ref count saturates and remains unchanged if it has reached
    // MAXREFCNT
    void DDdecRef(DDedge e) {
        complexDecRef(e.w);

        if (DDterminal(e))
            return;

        if (e.p->ref == MAXREFCNT)
            return;


        if (e.p->ref == 0) // ERROR CHECK
        {
            std::cerr <<"error in decref " << e.p->ref << "n";
            DDdebugnode(e.p);
            std::exit(8);
        }
        e.p->ref--;

        if (e.p->ref == 0) {
            if (!DDterminal(e)) {
                for (auto & i : e.p->e) {
                    if (i.p != nullptr) {
                        DDdecRef(i);
                    }
                }
            }
            Active[e.p->v]--;
            if (Active[e.p->v] < 0) {
                std::cerr << "ERROR in decref\n";
                std::exit(1);
            }
            ActiveNodeCount--;

            /******* Part added for sifting purposes ********/
            if (e.p->renormFactor != COMPLEX_ONE) {
                RenormalizationNodeCount--;
                e.p->renormFactor = COMPLEX_ONE;
            }
            if (e.p->block)
                blockMatrixCounter--;
            /******* by Niemann, November 2012 ********/
        }
    }

    // counting number of unique nodes in a QMDD
    unsigned int DDnodeCount(const DDedge e, std::unordered_set<DDnodePtr>& visited) {
        visited.insert(e.p);

        unsigned int sum = 1;
        if (!DDterminal(e)) {
            for (const auto & edge : e.p->e) {
                if (edge.p != nullptr && !visited.count(edge.p)) {
                    sum += DDnodeCount(edge, visited);
                }
            }
        }
        return sum;
    }

    // counts number of unique nodes in a QMDD
    unsigned int DDsize(const DDedge e) {
        std::unordered_set<DDnodePtr> visited(NODECOUNT_BUCKETS); // 2e6
        visited.max_load_factor(10);
        visited.clear();
        return DDnodeCount(e, visited);
    }

    void DDradixPrint(int p, int n)
    // prints p as an n bit Radix number
    // with leading 0's and no CR
    {
        int buffer[MAXN];
        for (int i = 0; i < n; i++) {
            buffer[i] = p % Radix;
            p = p / Radix;
        }
        for (int i = n - 1; i >= 0; i--)
            std::cout << buffer[i];
    }

    inline unsigned long CThash(const DDedge a, const DDedge b, const CTkind which) {
        const uintptr_t node_pointer = ((uintptr_t)a.p+(uintptr_t)b.p)>>3u;
        const uintptr_t weights = (uintptr_t)a.w.i+(uintptr_t)a.w.r+(uintptr_t)b.w.i+(uintptr_t)b.w.r;
        return (node_pointer+weights+(uintptr_t)which) & CTMASK;
    }

    inline unsigned long CThash2(DDnodePtr a, const complex_value aw, DDnodePtr b, const complex_value bw, const CTkind which) {
        const uintptr_t node_pointer = ((uintptr_t)a+(uintptr_t)b)>>3u;
        const uintptr_t weights = (uintptr_t)(aw.r*1000)+(uintptr_t)(aw.i*2000)+(uintptr_t)(bw.r*3000)+(uintptr_t)(bw.i*4000);
        return (node_pointer+weights+(uintptr_t)which) & CTMASK;
    }


    DDedge CTlookup(DDedge a, DDedge b, CTkind which) {
    // Lookup a computation in the compute table
    // return NULL if not a match else returns result of prior computation
        DDedge r;
        r.p = nullptr;
        CTlook[which]++;

        if (which == mult || which == fidelity) {
            const unsigned long i = CThash(a, b, which);

            if (CTable2[i].which != which) return (r);
            if (CTable2[i].a.p != a.p || !(CTable2[i].a.w == a.w)) return (r);
            if (CTable2[i].b.p != b.p || !(CTable2[i].b.w == b.w)) return (r);

            CThit[which]++;
            r.p = CTable2[i].r;

            complex c;
            c.r = ComplexCache_Avail;
            c.i = ComplexCache_Avail->next;
            c.r->val = CTable2[i].rw.r;
            c.i->val = CTable2[i].rw.i;

            if (Ceq(c, COMPLEX_ZERO)) {
                return DDzero;
            } else {
                ComplexCache_Avail = c.i->next;
                r.w = c;
            }

            return r;
        } else if (which == add) {

            complex_value aw;
            aw.r = a.w.r->val;
            aw.i = a.w.i->val;

            complex_value bw;
            bw.r = b.w.r->val;
            bw.i = b.w.i->val;

            const unsigned long i = CThash2(a.p, aw, b.p, bw, which);

            if (CTable3[i].which != which) return (r);
            if (CTable3[i].a != a.p || !Ceq(CTable3[i].aw, aw)) return (r);
            if (CTable3[i].b != b.p || !Ceq(CTable3[i].bw, bw)) return (r);

            CThit[which]++;
            r.p = CTable3[i].r;

            complex c;
            c.r = ComplexCache_Avail;
            c.i = ComplexCache_Avail->next;
            c.r->val = CTable3[i].rw.r;
            c.i->val = CTable3[i].rw.i;

            if (Ceq(c, COMPLEX_ZERO)) {
                return DDzero;
            } else {
                ComplexCache_Avail = c.i->next;
                r.w = c;
            }

            return r;
        } else if (which == conjugateTranspose || which == transpose) {
            const unsigned long i = CThash(a, b, which);

            if (CTable1[i].which != which) return (r);
            if (CTable1[i].a.p != a.p || !(CTable1[i].a.w == a.w)) return (r);
            if (CTable1[i].b.p != b.p || !(CTable1[i].b.w == b.w)) return (r);

            CThit[which]++;
            return CTable1[i].r;

        } else {
            std::cerr << "Undefined kind in CTlookup: " << which << "\n";
            std::exit(1);
        }
    }

    // put an entry into the compute table
    void CTinsert(DDedge a, DDedge b, DDedge r, CTkind which) {
        if (which == mult || which == fidelity) {
            const unsigned long i = CThash(a, b, which);

            CTable2[i].a = a;
            CTable2[i].b = b;
            CTable2[i].which = which;
            CTable2[i].r = r.p;
            CTable2[i].rw.r = r.w.r->val;
            CTable2[i].rw.i = r.w.i->val;
        } else if (which == add) {
            complex_value aw, bw;
            aw.r = a.w.r->val;
            aw.i = a.w.i->val;
            bw.r = b.w.r->val;
            bw.i = b.w.i->val;

            const unsigned long i = CThash2(a.p, aw, b.p, bw, which);

            CTable3[i].a = a.p;
            CTable3[i].aw = aw;
            CTable3[i].b = b.p;
            CTable3[i].bw = bw;
            CTable3[i].r = r.p;
            CTable3[i].rw.r = r.w.r->val;
            CTable3[i].rw.i = r.w.i->val;
            CTable3[i].which = which;

        } else if (which == conjugateTranspose || which == transpose) {
            const unsigned long i = CThash(a, b, which);

            CTable1[i].a = a;
            CTable1[i].b = b;
            CTable1[i].which = which;
            CTable1[i].r = r;
        } else {
            std::cerr << "Undefined kind in CTinsert: " << which << "\n";
            std::exit(1);
        }

    }

    unsigned int TThash(unsigned int n, int t, const int line[]) {
        unsigned int i = t;
        for (unsigned int j = 0; j < n; j++){
            if (line[j] == 1){
                i = i << (3 + j);
            }
        }
        return i & TTMASK;
    }

    DDedge TTlookup(int n, int m, int t, const int line[]) {
        DDedge r;
        r.p = nullptr;
        const unsigned int i = TThash(n, t, line);

        if (TTable[i].e.p == nullptr || TTable[i].t != t || TTable[i].m != m || TTable[i].n != n) {
            return r;
        }
        if (0 == memcmp(TTable[i].line, line, n * sizeof(int))) {
            return TTable[i].e;
        }
        return r;
    }

    void TTinsert(int n, int m, int t, const int line[], DDedge e) {
        const unsigned int i = TThash(n, t, line);
        TTable[i].n = n;
        TTable[i].m = m;
        TTable[i].t = t;
        memcpy(TTable[i].line, line, n * sizeof(int));
        TTable[i].e = e;
    }

    // recursively scan an QMDD putting values in entries of mat
    // v is the variable index
    void DDfillmat(complex mat[MAXDIM][MAXDIM], DDedge a, int r, int c, int dim, short v, const char vtype[]) {
        if (a.p == nullptr) {
            return;
        }

        if (v == -1) { // terminal node case
            if (r >= MAXDIM || c >= MAXDIM) {
                std::cerr << "out of bounds, r=" << r << ", c=" << c << "\n";
                return;
            }
            complex co;
            co.r = ComplexCache_Avail;
            co.i = ComplexCache_Avail->next;
            ComplexCache_Avail = co.i->next;

            co.r->val = CVAL(a.w.r);
            co.i->val = CVAL(a.w.i);

            mat[r][c] = co;
        } else {
            bool expand = (DDterminal(a)) || v != DDinverseOrder[a.p->v];
            for (int i = 0; i < Nedge; i++) {
                if (expand) {
                    DDfillmat(mat, a, r + (i / Radix) * dim / Radix,
                              c + (i % Radix) * dim / Radix, dim / Radix, v - 1,
                              vtype);
                } else {
                    DDedge e = a.p->e[i];

                    complex co;
                    co.r = ComplexCache_Avail;
                    co.i = ComplexCache_Avail->next;
                    ComplexCache_Avail = co.i->next;

                    Cmul(co, a.w, e.w);
                    e.w = co;

                    DDfillmat(mat, e, r + (i / Radix) * dim / Radix,c + (i % Radix) * dim / Radix,
                            dim / Radix, v - 1, vtype);

                    co.i->next = ComplexCache_Avail;
                    ComplexCache_Avail = co.r;
                }
            }
        }
    }

    void DDpermPrint(DDedge e, int row, int col) {
        if (DDterminal(e)) {
            if (e.w != COMPLEX_ONE)
                std::cerr << "error in permutation printing\n";
            else
                PermList[col] = row;
        } else
            for (int i = 0; i < Nedge; i++)
                if (e.p->e[i].p != nullptr && e.p->e[i].w != COMPLEX_ZERO)
                    DDpermPrint(e.p->e[i], row * Radix + i / Radix,col * Radix + i % Radix);
    }

/***************************************

 Public Routines

 ***************************************/

    // make a DD nonterminal node and return an edge pointing to it
    // node is not recreated if it already exists
    DDedge DDmakeNonterminal(short v, DDedge edge[MAXNEDGE], bool cached) {
        DDedge e;
        e.p = DDgetNode();  // get space and form node
        e.w = COMPLEX_ONE;
        e.p->v = v;
        e.p->renormFactor = COMPLEX_ONE;
        e.p->computeSpecialMatricesFlag = globalComputeSpecialMatricesFlag;

        memcpy(e.p->e, edge, Nedge * sizeof(DDedge));
        e = DDnormalize(e, cached); // normalize it
        e = DDutLookup(e);  // look it up in the unique tables
        return e;          // return result
    }

    // make a terminal - actually make an edge with appropriate weight
    // as there is only complex_one terminal DDone
    DDedge DDmakeTerminal(complex w) {
        DDedge e;
        e.p = DDterminalNode;
        e.w = w;
        return e;
    }
    // initialize DD package - must be called before other routines are used
    void DDinit(const bool verbose) {
        if (verbose) {
            std::cout << DDversion
                << "\n  compiled: " << __DATE__ << " " << __TIME__
                << "\n  edge size: " << sizeof(DDedge) << " bytes"
                << "\n  node size: " << sizeof(DDnode) << " bytes (with edges: " << sizeof(DDnode) + MAXNEDGE * sizeof(DDedge) << " bytes)"
                << "\n  max variables: " << MAXN
                << "\n  UniqueTable buckets: " << NBUCKET
                << "\n  ComputeTable slots: " << CTSLOTS
                << "\n  ToffoliTable slots: " << TTSLOTS
                << "\n  garbage collection limit: " << GCLIMIT1
                << "\n  garbage collection increment: " << GCLIMIT_INC
                << "\n" << std::flush;
        }

        complexInit();       // init complex number package
        DDinitComputeTable();  // init computed table to empty

        GCcurrentLimit = GCLIMIT1; // set initial garbage collection limit

        ComplexCurrentLimit = 100000;

        UTcol = UTmatch = UTlookups = 0;

        DDnodecount = 0;            // zero node counter
        DDpeaknodecount = 0;
        Nlabel = 0;                        // zero variable label counter
        Nop[0] = Nop[1] = Nop[2] = 0;        // zero op counter
        CTlook[0] = CTlook[1] = CTlook[2] = CThit[0] = CThit[1] = CThit[2] = 0;    // zero CTable counters
        Avail = nullptr;                // set available node list to empty
        Lavail = nullptr;                // set available element list to empty
        DDterminalNode = DDgetNode();// create terminal node - note does not go in unique table
        DDterminalNode->ident = 1;
        DDterminalNode->diag = 1;
        DDterminalNode->block = 0;
        DDterminalNode->symm = 1;
        DDterminalNode->c01 = 1;
        DDterminalNode->renormFactor = COMPLEX_ONE;
        DDterminalNode->computeSpecialMatricesFlag = 0;
        for (auto & i : DDterminalNode->e) {
            i.p = nullptr;
            i.w = COMPLEX_ZERO;
        }
        DDterminalNode->v = -1;

        DDzero = DDmakeTerminal(COMPLEX_ZERO);
        DDone = DDmakeTerminal(COMPLEX_ONE);


        for (auto & variable : Unique) {
            for (auto & bucket : variable) {
                bucket = nullptr; // set unique tables to empty
            }
        }
        for (int i = 0; i < MAXN; i++) //  set initial variable order to 0,1,2... from bottom up
        {
            DDorder[i] = DDinverseOrder[i] = i;
            Active[i] = 0;
        }
        ActiveNodeCount = 0;
        if (verbose) {
            std::cout << "DD initialization complete\n----------------------------------------------------------\n";
        }
    }

    // adds two matrices represented by DD
    // the two DD should have the same variable set and ordering
    DDedge DDadd2(DDedge x, DDedge y) {
        if (x.p == nullptr) {
            return y;  // handles partial matrices i.e.
        }
        if (y.p == nullptr) {
            return x;  // column and row vetors
        }
        Nop[add]++;

        if (x.w == COMPLEX_ZERO) {
            if (y.w == COMPLEX_ZERO) {
                return y;
            }
            complex c;
            c.r = ComplexCache_Avail;
            c.i = ComplexCache_Avail->next;
            ComplexCache_Avail = c.i->next;
            c.r->val = y.w.r->val;
            c.i->val = y.w.i->val;
            y.w = c;

            return y;
        }
        if (y.w == COMPLEX_ZERO) {
            complex c;
            c.r = ComplexCache_Avail;
            c.i = ComplexCache_Avail->next;
            ComplexCache_Avail = c.i->next;
            c.r->val = x.w.r->val;
            c.i->val = x.w.i->val;
            x.w = c;

            return x;
        }
        if (x.p == y.p) {
            DDedge r = y;

            complex result;
            result.r = ComplexCache_Avail;
            result.i = result.r->next;

            Cadd(result, x.w, y.w);
            if (Ceq(result, COMPLEX_ZERO)) {
                return DDzero;
            }
            ComplexCache_Avail = result.i->next;
            r.w = result;

            return r;
        }

        DDedge r = CTlookup(x, y, add);
        if (r.p != nullptr) {
            return (r);
        }

        int w;
        if (DDterminal(x)) {
            w = y.p->v;
        } else {
            w = x.p->v;
            if (!DDterminal(y) && DDinverseOrder[y.p->v] > DDinverseOrder[w]) {
                w = y.p->v;
            }
        }

        DDedge e1, e2, e[MAXNEDGE];
        for (int i = 0; i < Nedge; i++) {
            if (!DDterminal(x) && x.p->v == w) {
                e1 = x.p->e[i];

                if (e1.w != COMPLEX_ZERO) {
                    complex c;
                    c.r = ComplexCache_Avail;
                    c.i = ComplexCache_Avail->next;
                    ComplexCache_Avail = c.i->next;
                    Cmul(c, e1.w, x.w);
                    e1.w = c;
                }
            } else {
                if ((!MultMode) || (i % Radix == 0)) {
                    e1 = x;
                    if (y.p->e[i].p == nullptr) {
                        e1 = DDnullEdge;
                    }
                } else {
                    e1.p = nullptr;
                    e1.w = COMPLEX_ZERO;
                }
            }
            if (!DDterminal(y) && y.p->v == w) {
                e2 = y.p->e[i];

                if (e2.w != COMPLEX_ZERO) {
                    complex c;
                    c.r = ComplexCache_Avail;
                    c.i = ComplexCache_Avail->next;
                    ComplexCache_Avail = c.i->next;
                    Cmul(c, e2.w, y.w);
                    e2.w = c;
                }
            } else {
                if ((!MultMode) || (i % Radix == 0)) {
                    e2 = y;
                    if (x.p->e[i].p == nullptr) {
                        e2 = DDnullEdge;
                    }
                } else {
                    e2.p = nullptr;
                    e2.w = COMPLEX_ZERO;
                }
            }

            e[i] = DDadd2(e1, e2);

            if (!DDterminal(x) && x.p->v == w && e1.w != COMPLEX_ZERO) {
                e1.w.i->next = ComplexCache_Avail;
                ComplexCache_Avail = e1.w.r;
            }

            if (!DDterminal(y) && y.p->v == w && e2.w != COMPLEX_ZERO) {
                e2.w.i->next = ComplexCache_Avail;
                ComplexCache_Avail = e2.w.r;
            }
        }

        r = DDmakeNonterminal(w, e, true);

        CTinsert(x, y, r, add);

        return r;
    }

    DDedge DDadd(DDedge x, DDedge y) {
        DDedge result = DDadd2(x, y);

        if (result.w != COMPLEX_ZERO) {
            complex c = Clookup(result.w);
            result.w.i->next = ComplexCache_Avail;
            ComplexCache_Avail = result.w.r;
            result.w = c;
        }
        return result;
    }

    // new multiply routine designed to handle missing variables properly
    // var is number of variables
    DDedge DDmultiply2(DDedge x, DDedge y, const int var) {
        if (x.p == nullptr)
            return x;
        if (y.p == nullptr)
            return y;

        Nop[mult]++;

        if (x.w == COMPLEX_ZERO || y.w == COMPLEX_ZERO)  {
            return DDzero;
        }

        if (var == 0) {
            complex result;
            result.r = ComplexCache_Avail;
            result.i = result.r->next;
            ComplexCache_Avail = result.i->next;
            Cmul(result, x.w, y.w);
            return DDmakeTerminal(result);
        }

        const complex xweight = x.w;
        const complex yweight = y.w;
        x.w = COMPLEX_ONE;
        y.w = COMPLEX_ONE;

        DDedge r = CTlookup(x, y, mult);
        if (r.p != nullptr) {
            if (r.w != COMPLEX_ZERO) {
                Cmul(r.w, r.w, xweight);
                Cmul(r.w, r.w, yweight);
            }
            return r;
        }

        const int w = DDorder[var - 1];

        if (x.p->v == w && x.p->v == y.p->v) {
            if (x.p->ident) {
                r = y;
                CTinsert(x, y, r, mult);

                complex result;
                result.r = ComplexCache_Avail;
                result.i = result.r->next;
                ComplexCache_Avail = result.i->next;

                Cmul(result, xweight, yweight);
                r.w = result;


                return r;
            }
            if (y.p->ident) {
                r = x;
                CTinsert(x, y, r, mult);

                complex result;
                result.r = ComplexCache_Avail;
                result.i = result.r->next;
                ComplexCache_Avail = result.i->next;

                Cmul(result, xweight, yweight);
                r.w = result;

                return r;
            }
        }

        DDedge e[MAXNEDGE];
        for (int i = 0; i < Nedge; i += Radix) {
            for (int j = 0; j < Radix; j++) {
                e[i + j] = DDzero;
                for (int k = 0; k < Radix; k++) {
                    DDedge e1, e2;
                    if (!DDterminal(x) && x.p->v == w) {
                        e1 = x.p->e[i + k];
                    } else {
                        e1 = x;
                    }
                    if (!DDterminal(y) && y.p->v == w) {
                        e2 = y.p->e[j + Radix * k];
                    } else {
                        e2 = y;
                    }

                    DDedge m = DDmultiply2(e1, e2, var - 1);

                    if (k == 0 || e[i + j].w == COMPLEX_ZERO) {
                        e[i + j] = m;
                    } else if (m.w != COMPLEX_ZERO) {
                        DDedge old_e = e[i + j];

                        e[i + j] = DDadd2(e[i + j], m);

                        old_e.w.i->next = ComplexCache_Avail;
                        ComplexCache_Avail = old_e.w.r;
                        m.w.i->next = ComplexCache_Avail;
                        ComplexCache_Avail = m.w.r;
                    }
                }
            }
        }
        r = DDmakeNonterminal(w, e, true);

        CTinsert(x, y, r, mult);
        if (r.w != COMPLEX_ZERO) {
            Cmul(r.w, r.w, xweight);
            Cmul(r.w, r.w, yweight);

        }
        return r;
    }

    DDedge DDmultiply(DDedge x, DDedge y) {
        int var = 0;
        if (!DDterminal(x) && (DDinverseOrder[x.p->v] + 1) > var) {
            var = DDinverseOrder[x.p->v] + 1;
        }
        if (!DDterminal(y) && (DDinverseOrder[y.p->v] + 1) > var) {
            var = DDinverseOrder[y.p->v] + 1;
        }

        DDedge e = DDmultiply2(x, y, var);

        if (e.w != COMPLEX_ZERO) {
            complex c = Clookup(e.w);
            e.w.i->next = ComplexCache_Avail;
            ComplexCache_Avail = e.w.r;
            e.w = c;
        }

        return e;
    }

    DDedge DDtranspose(DDedge a)
    // returns a pointer to the transpose of the matrix a points to
    {
        if (a.p == nullptr || DDterminal(a) || a.p->symm) {
            return a;         // NULL pointer // terminal / or symmetric case   ADDED by Niemann Nov. 2012
        }

        DDedge r = CTlookup(a, a, transpose);     // check in compute table
        if (r.p != nullptr) {
            return (r);
        }

        DDedge e[MAXNEDGE];
        for (int i = 0; i < Radix; i++) { // transpose submatrices and rearrange as required
            for (int j = i; j < Radix; j++) {
                e[i * Radix + j] = DDtranspose(a.p->e[j * Radix + i]);
                if (i != j) {
                    e[j * Radix + i] = DDtranspose(a.p->e[i * Radix + j]);
                }
            }
        }

        r = DDmakeNonterminal(a.p->v, e, false);           // create new top vertex
        complex c;
        c.r = ComplexCache_Avail;
        c.i = ComplexCache_Avail->next;
        Cmul(c, r.w, a.w);              // adjust top weight
        r.w = Clookup(c);

        CTinsert(a, a, r, transpose);      // put in compute table
        return (r);
    }

    DDedge DDconjugateTranspose(DDedge a)
    // returns a pointer to the conjugate transpose of the matrix pointed to by a
    {
        if (a.p == nullptr)
            return a;          // NULL pointer
        if (DDterminal(a)) {              // terminal case
            a.w = Cconjugate(a.w);
            return a;
        }

        DDedge r = CTlookup(a, a, conjugateTranspose);  // check if in compute table
        if (r.p != nullptr) {
            return r;
        }

        DDedge e[MAXNEDGE];
        for (int i = 0; i < Radix; i++)    // conjugate transpose submatrices and rearrange as required
            for (int j = i; j < Radix; j++) {
                e[i * Radix + j] = DDconjugateTranspose(a.p->e[j * Radix + i]);
                if (i != j)
                    e[j * Radix + i] = DDconjugateTranspose(
                            a.p->e[i * Radix + j]);
            }
        r = DDmakeNonterminal(a.p->v, e, false);    // create new top node

        complex c;
        c.r = ComplexCache_Avail;
        c.i = ComplexCache_Avail->next;
        Cmul(c, r.w, Cconjugate(a.w));  // adjust top weight including conjugate
        r.w = Clookup(c);

        CTinsert(a, a, r, conjugateTranspose); // put it in the compute table
        return r;
    }

    DDedge DDident(const int x, const int y)
    // build a DD for the identity matrix for variables x to y (x<y)
    {
        DDedge f, edge[MAXNEDGE];

        if (y < 0)
            return DDone;

        if (x == 0 && DDid[y].p != nullptr) {
            return (DDid[y]);
        }
        if (y >= 1 && (f = DDid[y - 1]).p != nullptr) {
            for (int i = 0; i < Radix; i++) {
                for (int j = 0; j < Radix; j++) {
                    if (i == j)
                        edge[i * Radix + j] = f;
                    else
                        edge[i * Radix + j] = DDzero;
                }
            }
            DDedge e = DDmakeNonterminal(DDorder[y], edge, false);
            DDid[y] = e;
            return e;
        }
        for (int i = 0; i < Radix; i++) {
            for (int j = 0; j < Radix; j++) {
                if (i == j)
                    edge[i * Radix + j] = DDone;
                else
                    edge[i * Radix + j] = DDzero;
            }
        }
        DDedge e = DDmakeNonterminal(DDorder[x], edge, false);
        for (int k = x + 1; k <= y; k++) {
            for (int i = 0; i < Radix; i++)
                for (int j = 0; j < Radix; j++)
                    if (i == j)
                        edge[i * Radix + j] = e;
                    else
                        edge[i * Radix + j] = DDzero;
            e = DDmakeNonterminal(DDorder[k], edge, false);
        }
        if (x == 0)
            DDid[y] = e;
        return e;
    }

    // build matrix representation for a single gate on a circuit with n lines
    // line is the vector of connections
    // -1 not connected
    // 0...Radix-1 indicates a control by that value
    // Radix indicates the line is the target
    DDedge DDmvlgate(const DD_matrix mat, int n, const int line[]) {
        DDedge em[MAXNEDGE], fm[MAXNEDGE];
        int w, z;
        complex c;
        c.r = ComplexCache_Avail;
        c.i = ComplexCache_Avail->next;

        for (int i = 0; i < Radix; i++) {
            for (int j = 0; j < Radix; j++) {
                if (mat[i][j].r == 0.0 && mat[i][j].i == 0.0) {
                    em[i * Radix + j] = DDzero;
                } else {
                    c.r->val = mat[i][j].r;
                    c.i->val = mat[i][j].i;
                    em[i * Radix + j] = DDmakeTerminal(Clookup(c));
                }
            }
        }

        DDedge e = DDone;
        for (z = 0; line[w = DDorder[z]] < Radix; z++) { //process lines below target
            if (line[w] >= 0) { //  control line below target in DD
                for (int i1 = 0; i1 < Radix; i1++) {
                    for (int i2 = 0; i2 < Radix; i2++) {
                        DDedge f;
                        int i = i1 * Radix + i2;
                        if (i1 == i2) {
                            f = e;
                        } else {
                            f = DDzero;
                        }
                        for (int k = 0; k < Radix; k++) {
                            for (int j = 0; j < Radix; j++) {
                                int t = k * Radix + j;
                                if (k == j) {
                                    if (k == line[w]) {
                                        fm[t] = em[i];
                                    } else {
                                        fm[t] = f;
                                    }
                                } else {
                                    fm[t] = DDzero;
                                }
                            }
                        }
                        em[i] = DDmakeNonterminal(w, fm, false);
                    }
                }
            } else { // not connected
                for (int i = 0; i < Nedge; i++) {
                    for (int i1 = 0; i1 < Radix; i1++) {
                        for (int i2 = 0; i2 < Radix; i2++) {
                            if (i1 == i2) {
                                fm[i1 + i2 * Radix] = em[i];
                            } else {
                                fm[i1 + i2 * Radix] = DDzero;
                            }
                        }
                    }
                    em[i] = DDmakeNonterminal(w, fm, false);
                }
            }
            e = DDident(0, z);
        }
        e = DDmakeNonterminal(DDorder[z], em, false);  // target line
        for (z++; z < n; z++) { // go through lines above target
            if (line[w = DDorder[z]] >= 0) { //  control line above target in DD
                DDedge temp = DDident(0, z - 1);
                for (int i = 0; i < Radix; i++) {
                    for (int j = 0; j < Radix; j++) {
                        if (i == j) {
                            if (i == line[w]) {
                                em[i * Radix + j] = e;
                            } else {
                                em[i * Radix + j] = temp;
                            }
                        } else {
                            em[i * Radix + j] = DDzero;
                        }
                    }
                }
                e = DDmakeNonterminal(w, em, false);
            } else { // not connected
                for (int i1 = 0; i1 < Radix; i1++) {
                    for (int i2 = 0; i2 < Radix; i2++) {
                        if (i1 == i2) {
                            fm[i1 + i2 * Radix] = e;
                        } else {
                            fm[i1 + i2 * Radix] = DDzero;
                        }
                    }
                }
                e = DDmakeNonterminal(w, fm, false);
            }
        }
        return e;
    }

    // for building 0 or 1 control binary gates
    // c is the control variable
    // t is the target variable
    DDedge DDgate(DD_matrix mat, int n, int control, int target) {
        int line[MAXN];
        for (int i = 0; i < n; i++) {
            line[i] = -1;
        }
        if (control >= 0){
            line[control] = Radix - 1;
        }
        line[target] = Radix;
        return DDmvlgate(mat, n, line);
    }

    // a 0-1 matrix is printed more compactly
    //
    // Note: 0 entry and 1 entry in complex value table always denote
    // the values 0 and 1 respectively, so it is sufficient to print the index
    // for a 0-1 matrix.
    //
    // v is the variable index for the top vertex
    void DDmatrixPrint(DDedge a, short v, const char vtype[], std::ostream &os) {
        complex mat[MAXDIM][MAXDIM];
        complex cTabPrint[MAXDIM * MAXDIM + 2];
        int cTabEntries = 0;
        bool cTabPrintFlag = false;
        int n;

        if (DDterminal(a)) {
            n = 0;
        } else {
            n = v + 1;
        }
        int m = 1;
        for (int i = 0; i < n; i++) {
            m *= Radix;
        }

        if (n > MAXND) {
            std::cerr << "Matrix is too big to print. No. of vars=" << n << "\n";
            std::exit(1);
        }

        DDfillmat(mat, a, 0, 0, m, v, vtype); // convert to matrix


        cTabPrint[0] = COMPLEX_ZERO;
        cTabPrint[1] = COMPLEX_ONE;
        cTabEntries = 2;

        for (int i = 0; i < m; i++) {          // display matrix
            for (int j = 0; j < m; j++) {
                int k = 0;
                while (k < cTabEntries && !Ceq(cTabPrint[k], mat[i][j])) {
                    k++;
                }

                if (k == cTabEntries) {
                    cTabEntries++;
                    cTabPrint[k] = mat[i][j];
                }
                cTabPrintFlag = true;
                if (k < 10) {
                    os << " ";
                }
                os << k << " ";
                if (j == m / 2 - 1) {
                    os << "|";
                }
            }
            os << "\n";
            if (i == m / 2 - 1) {
                for (int j = 0; j < m; j++) {
                    os << " --";
                }
                os << "\n";
            }
        }
        if (cTabPrintFlag) {
            os << "ComplexTable values: "; //(0): 0; (1): 1; ";

            for (int i = 0; i < cTabEntries; i++) {
                os << "(" << i << "):" << cTabPrint[i] << "; ";
            }
        }

        os << "\n";

        for (int i = 0; i < (1 << n); i++) {
            for (int j = 0; j < (1 << n); j++) {
                mat[i][j].i->next = ComplexCache_Avail;
                ComplexCache_Avail = mat[i][j].r;
            }
        }
    }

    void DDmatrixPrint(DDedge a, short v, char vtype[]) {
        DDmatrixPrint(a, v, vtype, std::cout);
    }

    void DDmatrixPrint2(DDedge a, std::ostream &os) {
        char v[MAXN];
        int i;

        if (DDterminal(a)) {
            os << a.w << "\n";
        } else {
            for (i = 0; i < MAXN; i++) {
                v[i] = 0;
            }
            DDmatrixPrint(a, a.p->v, v, os);
        }
    }

    void DDmatrixPrint2(DDedge a, std::ostream &os, short n) {
        if (DDterminal(a)) {
            os << a.w << "\n";
        } else {
            char v[MAXN]{};
            DDmatrixPrint(a, n, v, os);
        }
    }

    void DDmatrixPrint2(DDedge a) {
        if (DDterminal(a)) {
            std::cout << a.w << "\n";
        } else {
            char v[MAXN]{};
            DDmatrixPrint(a, DDinverseOrder[a.p->v], v);
        }
    }

    // displays DD package statistics
    void DDstatistics() {
        std::cout << "\nDD statistics:"
            << "\n  Current # nodes in UniqueTable: " << DDnodecount
            << "\n  Total compute table lookups: " << CTlook[0] + CTlook[1] + CTlook[2]
            << "\n  Number of operations:"
            << "\n    add:  " <<  Nop[add]
            << "\n    mult: " <<  Nop[mult]
            << "\n    kron: " <<  Nop[kronecker]
            << "\n  Compute table hit ratios (hits/looks/ratio):"
            << "\n    adds: " << CThit[add] << " / " << CTlook[add] << " / " << (double) CThit[add] / (double)CTlook[add]
            << "\n    mult: " << CThit[mult] << " / " << CTlook[mult] << " / " << (double) CThit[mult] / (double)CTlook[mult]
            << "\n    kron: " << CThit[kronecker] << " / " << CTlook[kronecker] << " / " << (double) CThit[kronecker] / (double)CTlook[kronecker]
            << "\n  UniqueTable:"
            << "\n    Collisions: " << UTcol
            << "\n    Matches:    " << UTmatch
            << "\n" << std::flush;
    }

    // print number of active nodes for variables 0 to n-1
    void DDprintActive(int n) {
        std::cout << "#printActive: " << ActiveNodeCount << ". ";
        for (int i = 0; i < n; i++) {
            std::cout << " " << Active[i] << " ";
        }
        std::cout << "\n";
    }

    complex_value DDfidelity(DDedge x, DDedge y, int var) {
        if (x.p == nullptr || y.p == nullptr || x.w == COMPLEX_ZERO || y.w == COMPLEX_ZERO)  // the 0 case
        {
            return {0.0,0.0};
        }

        if (var == 0) {
            complex result;
            result.r = ComplexCache_Avail;
            result.i = result.r->next;
            Cmul(result, x.w, y.w);
            return {result.r->val, result.i->val};
        }

        complex xweight = x.w;
        complex yweight = y.w;
        x.w = COMPLEX_ONE;
        y.w = COMPLEX_ONE;

        DDedge r = CTlookup(x, y, fidelity);
        if (r.p != nullptr) {
            r.w.i->next = ComplexCache_Avail;
            ComplexCache_Avail = r.w.r;

            Cmul(r.w, r.w, xweight);
            Cmul(r.w, r.w, yweight);
            return {r.w.r->val, r.w.i->val};
        }

        long w = DDorder[var - 1];
        complex_value sum{0.0, 0.0};
        DDedge e1, e2;

        for (int i = 0; i < Nedge; i++) {
            if (!DDterminal(x) && x.p->v == w) {
                e1 = x.p->e[i];
            } else {
                e1 = x;
            }
            if (!DDterminal(y) && y.p->v == w) {
                e2 = y.p->e[i];
                e2.w = Cconjugate(e2.w);
            } else {
                e2 = y;
            }
            complex_value cv = DDfidelity(e1, e2, var - 1);

            sum.r += cv.r;
            sum.i += cv.i;
        }

        r = DDzero;
        r.w.r = ComplexCache_Avail;
        r.w.i = ComplexCache_Avail->next;
        r.w.r->val = sum.r;
        r.w.i->val = sum.i;

        CTinsert(x, y, r, fidelity);
        Cmul(r.w, r.w, xweight);
        Cmul(r.w, r.w, yweight);

        return {r.w.r->val, r.w.i->val};
    }

    long double DDfidelity(DDedge x, DDedge y) {
        long w = DDinverseOrder[x.p->v];
        if(DDinverseOrder[y.p->v] > w) {
            w = DDinverseOrder[y.p->v];
        }

        complex c;
        c.r = ComplexCache_Avail;
        c.i = ComplexCache_Avail->next;

        c.r->val = CVAL(y.w.r);
        c.i->val = CVAL(y.w.i);

        long double norm = CmagSquared(c);

        c.r->val /= std::sqrt(norm);
        c.i->val /= std::sqrt(norm);

        complex c2 = Clookup(c);

        y.w = Cconjugate(c2);
        return DDfidelity(x, y, w+1).r;
    }
}