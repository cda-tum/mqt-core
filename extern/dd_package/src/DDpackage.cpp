/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDpackage.h"

#include <cstdlib>
#include <iomanip>

namespace dd {

    Node Package::terminal{
            nullptr,
            {{nullptr, CN::ZERO}, {nullptr, CN::ZERO}, {nullptr, CN::ZERO}, {nullptr, CN::ZERO}},
            0,
            -1,
            true,
            true,
    };
    constexpr Edge Package::DDzero;
    constexpr Edge Package::DDone;

    Package::Package() : cn(ComplexNumbers()) {
		// Initialization of the compute table during construction should not be necessary as it is default initialized
		// initComputeTable();  // init computed table to empty
    }

    Package::~Package() {
        for (auto chunk : allocated_list_chunks) {
            delete[] chunk;
        }
        for (auto chunk : allocated_node_chunks) {
            delete[] chunk;
        }
    }

    ListElementPtr Package::newListElement() {
        ListElementPtr r;

        if (listAvail != nullptr) {   // get node from avail chain if possible
            r = listAvail;
            listAvail = listAvail->next;
        } else {            // otherwise allocate 2000 new nodes
            r = new ListElement[LIST_CHUNK_SIZE];
            allocated_list_chunks.push_back(r);
            ListElementPtr r2 = r + 1;
            listAvail = r2;
            for (unsigned int i = 0; i < LIST_CHUNK_SIZE - 2; i++, r2++) {
                r2->next = r2 + 1;
            }
            r2->next = nullptr;
        }
        return r;
    }

    ComplexValue Package::getVectorElement(const Edge& e, const unsigned long long element) {
        if (isTerminal(e)) {
            return {0, 0};
        }
        Complex l = cn.getTempCachedComplex(1, 0);
        auto r = e;
        do {
            CN::mul(l, l, r.w);
            auto tmp = (element >> r.p->v) & 1u;
            r = r.p->e[2 * tmp];
        } while (!isTerminal(r));
        CN::mul(l, l, r.w);

        return {CN::val(l.r), CN::val(l.i)};
    }

    void Package::toDot(const Edge& e, std::ostream &oss, bool isVector) {
        /* first part of dot file*/
        std::ostringstream nodes;
        /*second part of dot file*/
        std::ostringstream edges;

        edges << "\n";
        /*Initialize Graph*/
        nodes << "digraph \"DD\" {\n"
              << "graph [center=true, ordering=out];\n"
              << "node [shape=oval, center=true];\n"
              << "\"T\" [ shape = box, label=\"1\" ];\n";
        /* Define Nodes */
        ListElementPtr first, q, lastq, pnext;
        first = newListElement();
        first->p = e.p;
        first->next = nullptr;
        first->w = 0;

        unsigned short n = 0, i = 0;

        nodes << "\"R\"";
        //füge Kante zwischen helper node und neuem Knoten hinzu
        if (CN::equalsOne(e.w)) {
            nodes << " [label=\"\", shape=point];\n";
            edges << "\"R\" -> \"0\"\n";
        } else {
            nodes << " [label=\"\", shape=point];\n";
            auto ref_r = CN::get_sane_pointer(e.w.r)->ref;
            auto ref_i = CN::get_sane_pointer(e.w.i)->ref;
            edges << R"("R" -> "0" [label="()" << e.w << ") " << ref_r << " " << ref_i << "\" ];\n";
        }


        pnext = first;
        while (pnext != nullptr) {
            /* Zeichne Knoten*/
            nodes << "\"" << i << "\" " << "[ label=\""
                  << "q" << pnext->p->v << " " << pnext->p->ref
                  << "\" ,style=filled, fillcolor=lightgray ];\n";

            if (pnext->p != DDzero.p) {
                edges << "{rank=same;";
                for (unsigned int k = 0; k < NEDGE; k++) {
                    if (isVector && k % RADIX != 0) continue;
                    edges << " \"" << i << "h" << k << "\"";
                }
                edges << "}\n";

                for (int j = 0; j < NEDGE; j++) {
                    if (isVector && j % RADIX != 0) continue;
                    if (pnext->p->e[j].p == nullptr);
                    else {
                        if (!isTerminal(pnext->p->e[j])) {
                            q = first->next;
                            lastq = first;
                            while (q != nullptr && pnext->p->e[j].p != q->p) {
                                lastq = q;
                                q = q->next;
                            }
                            if (q == nullptr) {
                                q = newListElement();
                                q->p = pnext->p->e[j].p;
                                q->next = nullptr;
                                q->w = n = n + 1;
                                lastq->next = q;
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
                            if (CN::equalsOne(pnext->p->e[j].w)) {
                                nodes << " [label=\"\", shape=point];\n";
                                edges << "\"" << i << "h" << j << "\" -> \"" << q->w << "\";\n";
                            } else {
                                nodes << " [label=\"\", shape=point];\n";
                                auto ref_r = CN::get_sane_pointer(pnext->p->e[j].w.r)->ref;
                                auto ref_i = CN::get_sane_pointer(pnext->p->e[j].w.i)->ref;
                                edges << "\"" << i << "h" << j << "\" -> \"" << q->w
                                      << "\" [label=\" (" << pnext->p->e[j].w << ") " << ref_r << " " << ref_i << "\" ];\n";
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
                            if (CN::equalsZero(pnext->p->e[j].w)) {
                                nodes << ", fillcolor=red, color=red";
                            } else if (CN::equalsOne(pnext->p->e[j].w)) {
                                edges << "\"" << i << "h" << j << "\"-> \"T\";\n";
                            } else {
                                edges << "\"" << i << "h" << j << R"("-> "T" [label= "()" << pnext->p->e[j].w
                                      << ")\", ];\n";
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
    void Package::export2Dot(const Edge& basic, const std::string &outputFilename, bool isVector, bool show) {
        std::ofstream init(outputFilename);
        toDot(basic, init, isVector);
        init.close();

        if (show) {
            std::ostringstream oss;
            oss << "dot -Tsvg " << outputFilename << " -o "
                << outputFilename.substr(0, outputFilename.find_last_of('.')) << ".svg";
            auto str = oss.str(); // required to avoid immediate deallocation of temporary
            static_cast<void>(!std::system(str.c_str())); // cast and ! just to suppress the unused result warning
        }
    }

    Edge Package::makeZeroState(unsigned short n) {
        Edge f = DDone;
        Edge edges[4];
        edges[1] = edges[2] = edges[3] = DDzero;

        for (short p = 0; p < n; p++) {
            edges[0] = f;
            f = makeNonterminal(p, edges);
        }
        return f;
    }

    // create DD for basis state |q_n-1 q_n-2 ... q1 q0>
    Edge Package::makeBasisState(unsigned short n, const std::bitset<MAXN> &state) {
        Edge f = DDone;
        Edge edges[4];
        edges[1] = edges[3] = DDzero;

        for (short p = 0; p < n; ++p) {
            if (state[p] == 0) {
                edges[0] = f;
                edges[2] = DDzero;
            } else {
                edges[0] = DDzero;
                edges[2] = f;
            }
            f = makeNonterminal(p, edges);
        }
        return f;
    }

    Edge Package::makeBasisState(unsigned short n, const std::vector<BasisStates> &state) {
        if (state.size() < n) {
            std::cerr << "Insufficient qubit states provided. Requested " << n << ", but received " << state.size()
                      << std::endl;
            exit(1);
        }

        Edge f = DDone;
        Edge edges[4];
        edges[1] = edges[3] = DDzero;

        for (unsigned short p = 0; p < n; ++p) {
            switch (state[p]) {
                case BasisStates::zero:
                    edges[0] = f;
                    edges[2] = DDzero;
                    break;
                case BasisStates::one:
                    edges[0] = DDzero;
                    edges[2] = f;
                    break;
                case BasisStates::plus:
                    edges[0].p = f.p;
                    edges[0].w = cn.lookup(SQRT_2, 0);
                    edges[2].p = f.p;
                    edges[2].w = cn.lookup(SQRT_2, 0);
                    break;
                case BasisStates::minus:
                    edges[0].p = f.p;
                    edges[0].w = cn.lookup(SQRT_2, 0);
                    edges[2].p = f.p;
                    edges[2].w = cn.lookup(-SQRT_2, 0);
                    break;
                case BasisStates::right:
                    edges[0].p = f.p;
                    edges[0].w = cn.lookup(SQRT_2, 0);
                    edges[2].p = f.p;
                    edges[2].w = cn.lookup(0, SQRT_2);
                    break;
                case BasisStates::left:
                    edges[0].p = f.p;
                    edges[0].w = cn.lookup(SQRT_2, 0);
                    edges[2].p = f.p;
                    edges[2].w = cn.lookup(0, -SQRT_2);
                    break;
            }

            f = makeNonterminal(static_cast<short>(p), edges);
        }
        return f;
    }

    Edge Package::normalize(const Edge &e, bool cached) {
        int argmax = -1;

        bool zero[] = {CN::equalsZero(e.p->e[0].w),
                       CN::equalsZero(e.p->e[1].w),
                       CN::equalsZero(e.p->e[2].w),
                       CN::equalsZero(e.p->e[3].w)};

        for (int i = 0; i < NEDGE; i++) {
            if (zero[i] && e.p->e[i].w != CN::ZERO) {
                cn.releaseCached(e.p->e[i].w);
                e.p->e[i] = DDzero;
            }
        }

        /// --- Matrix treatment ---
        if (mode == Mode::Matrix || !zero[1] || !zero[3]) {
            fp max = 0.L;
            Complex maxc = ComplexNumbers::ONE;
            // determine max amplitude
            for (int i = 0; i < NEDGE; ++i) {
                if (zero[i]) continue;
                if (argmax == -1) {
                    argmax = i;
                    max = ComplexNumbers::mag2(e.p->e[i].w);
                    maxc = e.p->e[i].w;
                } else {
                    auto mag = ComplexNumbers::mag2(e.p->e[i].w);
                    if (mag - max > CN::TOLERANCE) {
                        argmax = i;
                        max = mag;
                        maxc = e.p->e[i].w;
                    }
                }
            }

            // all equal to zero - make sure to release cached numbers approximately zero, but not exactly zero
            if (argmax == -1) {
                if (cached) {
                    for (auto const &i : e.p->e) {
                        if (i.w != CN::ZERO) {
                            cn.releaseCached(i.w);
                        }
                    }
                } else if (&e.p != &DDzero.p) {
                    // If it is not a cached variable, I have to put it pack into the chain
                    e.p->next = nodeAvail;
                    nodeAvail = e.p;
                }
                return DDzero;
            }

            auto r = e;
            // divide each entry by max
            for (int i = 0; i < NEDGE; ++i) {
                if (i == argmax) {
                    if (cached) {
                        if (r.w == ComplexNumbers::ONE)
                            r.w = maxc;
                        else
                            CN::mul(r.w, r.w, maxc);
                    } else {
                        if (r.w == ComplexNumbers::ONE) {
                            r.w = maxc;
                        } else {
                            auto c = cn.getTempCachedComplex();
                            CN::mul(c, r.w, maxc);
                            r.w = cn.lookup(c);
                        }
                    }
                    r.p->e[i].w = ComplexNumbers::ONE;
                } else {
                    if (zero[i]) {
                        if (cached && r.p->e[i].w != ComplexNumbers::ZERO)
                            cn.releaseCached(r.p->e[i].w);
                        r.p->e[i] = DDzero;
                        continue;
                    }
                    if (cached && !zero[i] && r.p->e[i].w != ComplexNumbers::ONE) {
                        cn.releaseCached(r.p->e[i].w);
                    }
                    if (CN::equalsOne(r.p->e[i].w))
                        r.p->e[i].w = ComplexNumbers::ONE;
                    auto c = cn.getTempCachedComplex();
                    CN::div(c, r.p->e[i].w, maxc);
                    r.p->e[i].w = cn.lookup(c);
                }
            }
            return r;
        }

        /// --- Vector treatment ---
        fp sum = 0.L;
        fp div = 0.L;

        for (int i = 0; i < NEDGE; ++i) {
            if (e.p->e[i].p == nullptr || zero[i]) {
                continue;
            }

            if (argmax == -1) {
                argmax = i;
                sum = div = ComplexNumbers::mag2(e.p->e[i].w);
            } else {
                sum += ComplexNumbers::mag2(e.p->e[i].w);
            }
        }

        if (argmax == -1) {
            if (cached) {
                for (auto &i : e.p->e) {
                    if (i.p == nullptr && i.w != CN::ZERO) {
                        cn.releaseCached(i.w);
                    }
                }
            } else if (&e.p != &DDzero.p) {
                // If it is not a cached variable, I have to put it pack into the chain
                e.p->next = nodeAvail;
                nodeAvail = e.p;
            }
            return DDzero;
        }

        sum = std::sqrt(sum / div);

        auto r = e;
        if (cached && r.p->e[argmax].w != CN::ONE) {
            r.w = r.p->e[argmax].w;
            r.w.r->val *= sum;
            r.w.i->val *= sum;
        } else {
            r.w = cn.lookup(ComplexNumbers::val(r.p->e[argmax].w.r) * sum, ComplexNumbers::val(r.p->e[argmax].w.i) * sum);
            CN::incRef(r.w);
            if (CN::equalsZero(r.w)) {
                return DDzero;
            }
        }

        for (int j = 0; j < NEDGE; j++) {
            if (j == argmax) {
                r.p->e[j].w = cn.lookup((fp) 1.0L / sum, 0);
                CN::incRef(r.p->e[j].w);
                if (r.p->e[j].w == CN::ZERO)
                    r.p->e[j] = DDzero;
            } else if (r.p->e[j].p != nullptr && !zero[j]) {
                if (cached) {
                    cn.releaseCached(r.p->e[j].w);
                    CN::div(r.p->e[j].w, r.p->e[j].w, r.w);
                    r.p->e[j].w = cn.lookup(r.p->e[j].w);
                    if (r.p->e[j].w == CN::ZERO) {
                        r.p->e[j] = DDzero;
                    }
                } else {
                    Complex c = cn.getTempCachedComplex();
                    CN::div(c, r.p->e[j].w, r.w);
                    r.p->e[j].w = cn.lookup(c);
                    CN::incRef(r.p->e[j].w);
                    if (r.p->e[j].w == CN::ZERO) {
                        r.p->e[j] = DDzero;
                    }
                }
            }
        }
        return r;
    }


    std::size_t Package::UThash(NodePtr p) {
        std::uintptr_t key = 0;
        // note hash function shifts pointer values so that order is important
        // suggested by Dr. Nigel Horspool and helps significantly
        for (unsigned int i = 0; i < NEDGE; i++) {
            key += ((reinterpret_cast<std::uintptr_t>(p->e[i].p) >> i)
                    + (reinterpret_cast<std::uintptr_t>(p->e[i].w.r) >> i)
                    + (reinterpret_cast<std::uintptr_t>(p->e[i].w.i) >> (i + 1))) & HASHMASK;
            key &= HASHMASK;
        }
        return key;
    }

    //  lookup a node in the unique table for the appropriate variable - if not found insert it
    //  only normalized nodes shall be stored.
    Edge Package::UTlookup(const Edge& e, bool keep_node) {
        // there is a unique terminal node
        if (isTerminal(e)) {
            return e;
        }
        UTlookups++;

        std::size_t key = UThash(e.p);

        unsigned short v = e.p->v;
        assert(v - 1 == e.p->e[0].p->v || isTerminal(e.p->e[0]));
        assert(v - 1 == e.p->e[1].p->v || isTerminal(e.p->e[1]));
        assert(v - 1 == e.p->e[2].p->v || isTerminal(e.p->e[2]));
        assert(v - 1 == e.p->e[3].p->v || isTerminal(e.p->e[3]));

        NodePtr p = Unique[v][key]; // find pointer to appropriate collision chain
        while (p != nullptr)    // search for a match
        {
            if (std::memcmp(e.p->e, p->e, NEDGE * sizeof(Edge)) == 0) {
                // Match found
                if (e.p != p && !keep_node) {
                    e.p->next = nodeAvail;    // put node pointed to by e.p on avail chain
                    nodeAvail = e.p;
                }

                // NOTE: reference counting is to be adjusted by function invoking the table lookup
                UTmatch++;        // record hash table match
                assert(CN::ONE.r->val == 1);
                assert(CN::ONE.i->val == 0);
                assert(p->v == e.p->v);
                assert(v - 1 == e.p->e[0].p->v || isTerminal(e.p->e[0]));
                assert(v - 1 == e.p->e[1].p->v || isTerminal(e.p->e[1]));
                assert(v - 1 == e.p->e[2].p->v || isTerminal(e.p->e[2]));
                assert(v - 1 == e.p->e[3].p->v || isTerminal(e.p->e[3]));

	            auto r = e;
	            r.p = p;// and set it to point to node found (with weight unchanged)
                return r;
            }

            UTcol++;        // record hash collision
            p = p->next;
        }
        e.p->next = Unique[v][key]; // if end of chain is reached, this is a new node
        Unique[v][key] = e.p;       // add it to front of collision chain

        nodecount++;          // count that it exists
        if (nodecount > peaknodecount)
            peaknodecount = nodecount;

        checkSpecialMatrices(e.p);
        return e;                // and return
    }

    std::string Package::UTcheck(const Edge& e) const {
        if (isTerminal(e)) {
            return "terminal";
        }

        const unsigned short v = e.p->v;

        for (std::size_t key = 0; key < Unique[v].size(); key++) {
            NodePtr p = Unique[v][key];
            while (p != nullptr) {
                if (e.p == p) {
                    if (UThash(p) == key) {
                        return std::to_string(key); // correct bucket
                    } else {
                        return "!" + std::to_string(key); // wrong bucket
                    }
                }
                p = p->next;
            }
        }
        return "not_found";
    }

    // set compute table to empty and
    // set toffoli gate table to empty and
    // set identity table to empty
    void Package::initComputeTable() {
        for (unsigned int i = 0; i < CTSLOTS; i++) {
            for (auto &table : CTable1) {
                table[i].r.p = nullptr;
                table[i].which = none;
            }
            for (auto &table : CTable2) {
                table[i].r = nullptr;
                table[i].which = none;
            }
            for (auto &table : CTable3) {
                table[i].r = nullptr;
                table[i].which = none;
            }
        }
        for (auto &table : OperationTable) {
            table.r = nullptr;
        }

        for (auto &i : TTable) {
            i.e.p = nullptr;
        }
        for (auto &i : IdTable) {
            i.p = nullptr;
        }
    }

    // a simple garbage collector that removes nodes with 0 ref count from the unique
    // tables placing them on the available space chain
    void Package::garbageCollect(bool force) {
        gc_calls++;
        if (!force && nodecount < currentNodeGCLimit && cn.count < currentComplexGCLimit) {
            return; // do not collect if below current limits
        }
        gc_runs++;

        int count = 0;
        int counta = 0;
        for (auto &variable : Unique) {
            for (auto &bucket : variable) {
                NodePtr lastp = nullptr;
                NodePtr p = bucket;
                while (p != nullptr) {
                    if (p->ref == 0) {
                        if (p == terminalNode) {
                            throw std::runtime_error("Tried to collect a terminal node.");
                        }
                        count++;
                        NodePtr nextp = p->next;
                        if (lastp == nullptr)
                            bucket = p->next;
                        else
                            lastp->next = p->next;
                        p->next = nodeAvail;
                        nodeAvail = p;
                        p = nextp;
                    } else {
                        lastp = p;
                        p = p->next;
                        counta++;
                    }
                }
            }
        }
        currentNodeGCLimit += GCLIMIT_INC;
        nodecount = counta;
        cn.garbageCollect(); // NOTE: this cleans all complex values with ref-count 0
        currentComplexGCLimit += ComplexNumbers::GCLIMIT_INC;
        initComputeTable(); // IMPORTANT sets compute table to empty after garbage collection
    }

    // get memory space for a node
    NodePtr Package::getNode() {
        NodePtr r;
        if (nodeAvail != nullptr)    // get node from avail chain if possible
        {
            r = nodeAvail;
            nodeAvail = nodeAvail->next;
        } else {            // otherwise allocate new nodes
            r = new Node[NODE_CHUNK_SIZE];
            node_allocations += NODE_CHUNK_SIZE;
            allocated_node_chunks.push_back(r);
            NodePtr r2 = r + 1;
            nodeAvail = r2;
            for (unsigned int i = 0; i < NODE_CHUNK_SIZE - 2; i++, r2++) {
                r2->next = r2 + 1;
            }
            r2->next = nullptr;
        }
        r->next = nullptr;
        r->ref = 0; // nodes in nodeAvailable may have non-zero ref count
        r->ident = r->symm = false; // mark as not identity or symmetric
        return r;
    }

    // increment reference counter for node e points to
    // and recursively increment reference counter for
    // each child if this is the first reference
    //
    // a ref count saturates and remains unchanged if it has reached
    // MAXREFCNT
    void Package::incRef(const Edge &e) {
        dd::ComplexNumbers::incRef(e.w);
        if (isTerminal(e))
            return;

        if (e.p->ref == MAXREFCNT) {
            std::clog << "[WARN] MAXREFCNT reached for e.w=" << e.w << ". Weight will never be collected.\n";
            debugnode(e.p);
            return;
        }

        e.p->ref++;

        if (e.p->ref == 1) {
            for (auto &edge : e.p->e) {
                if (edge.p != nullptr) {
                    incRef(edge);
                }
            }

            active[e.p->v]++;
            activeNodeCount++;
            maxActive = std::max(maxActive, activeNodeCount);
        }
    }

    // decrement reference counter for node e points to
    // and recursively decrement reference counter for
    // each child if this is the last reference
    //
    // a ref count saturates and remains unchanged if it has reached
    // MAXREFCNT
    void Package::decRef(const Edge &e) {
        dd::ComplexNumbers::decRef(e.w);

        if (isTerminal(e)) return;
        if (e.p->ref == MAXREFCNT) return;

        if (e.p->ref == 0) // ERROR CHECK
        {
            debugnode(e.p);
            throw std::runtime_error("In decref: ref==0 before decref\n");
        }

        e.p->ref--;

        if (e.p->ref == 0) {
            for (auto &edge : e.p->e) {
                if (edge.p != nullptr) {
                    decRef(edge);
                }
            }
            active[e.p->v]--;
            activeNodeCount--;
        }
    }

    // counting number of unique nodes in a DD
    unsigned int Package::nodeCount(const Edge &e, std::unordered_set<NodePtr> &v) const {
        v.insert(e.p);

        unsigned int sum = 1;
        if (!isTerminal(e)) {
            for (const auto &edge : e.p->e) {
                if (edge.p != nullptr && !v.count(edge.p)) {
                    sum += nodeCount(edge, v);
                }
            }
        }
        return sum;
    }

    // counts number of unique nodes in a DD
    unsigned int Package::size(const Edge &e) {
		static std::unordered_set<NodePtr> visited{NODECOUNT_BUCKETS}; // 2e6
	    visited.max_load_factor(10);
        visited.clear();
        return nodeCount(e, visited);
    }


    inline unsigned long Package::CThash(const Edge &a, const Edge &b, const CTkind which) {
        const std::uintptr_t node_pointer = (reinterpret_cast<std::uintptr_t>(a.p) + reinterpret_cast<std::uintptr_t>(b.p)) >> 3u;

        const std::uintptr_t weights = reinterpret_cast<std::uintptr_t>(a.w.i)
                                       + reinterpret_cast<std::uintptr_t>(a.w.r)
                                       + reinterpret_cast<std::uintptr_t>(b.w.i)
                                       + reinterpret_cast<std::uintptr_t>(b.w.r);

        return (node_pointer + weights + which) & CTMASK;
    }

    inline unsigned long Package::CThash2(NodePtr a, const ComplexValue &aw, NodePtr b, const ComplexValue &bw, const CTkind which) {
        const unsigned long node_pointer = (reinterpret_cast<std::uintptr_t>(a) + reinterpret_cast<std::uintptr_t>(b)) >> 3u;
        const auto weights = static_cast<unsigned long>(aw.r * 1000 + aw.i * 2000 + bw.r * 3000 + bw.i * 4000);
        return (node_pointer + weights + which) & CTMASK;
    }

    Edge Package::OperationLookup(const unsigned int operationType, const short *line, const unsigned short nQubits) {
        operationLook++;
        Edge r{nullptr, {nullptr, nullptr}};
        const unsigned long i = OperationHash(operationType, line, nQubits);
        if (OperationTable[i].r == nullptr) return r;
        if (OperationTable[i].operationType != operationType) return r;
        if (OperationTable[i].r->v != nQubits - 1) return r;

        if (std::memcmp(OperationTable[i].line, line, (nQubits) * sizeof(short)) != 0) return r;

        r.p = OperationTable[i].r;
        if (std::fabs(OperationTable[i].rw.r) < CN::TOLERANCE && std::fabs(OperationTable[i].rw.i) < CN::TOLERANCE) {
            return DDzero;
        } else {
            r.w = cn.lookup(OperationTable[i].rw.r, OperationTable[i].rw.i);
        }
        operationCThit++;
        return r;
    }

    void Package::OperationInsert(const unsigned int operationType, const short *line, const Edge &result, const unsigned short nQubits) {
        const unsigned long i = OperationHash(operationType, line, nQubits);
        std::memcpy(OperationTable[i].line, line, (nQubits) * sizeof(short));
        OperationTable[i].operationType = operationType;
        OperationTable[i].r = result.p;
        OperationTable[i].rw.r = CN::val(result.w.r);
        OperationTable[i].rw.i = CN::val(result.w.i);
    }

    unsigned long Package::OperationHash(const unsigned int operationType, const short *line, const unsigned short nQubits) {
        unsigned long i = operationType;
        for (unsigned short j = 0; j <= nQubits; j++) {
//            i = (i << 5u) + (4 * j) + (line[j] * 4);
            i = (i << 3u) + i * j + line[j];
        }
        return i & OperationMASK;
    }

    Edge Package::CTlookup(const Edge &a, const Edge &b, const CTkind which) {
        // Lookup a computation in the compute table
        // return NULL if not a match else returns result of prior computation
        Edge r{nullptr, {nullptr, nullptr}};
        CTlook[which]++;

        if (which == mult || which == fid || which == kron) {
            std::array<CTentry2, CTSLOTS> &table = CTable2.at(mode);
            const unsigned long i = CThash(a, b, which);

            if (table[i].which != which) return r;
            if (!equals(table[i].a, a)) return r;
            if (!equals(table[i].b, b)) return r;

            CThit[which]++;
            r.p = table[i].r;

            if (std::fabs(table[i].rw.r) < CN::TOLERANCE && std::fabs(table[i].rw.i) < CN::TOLERANCE) {
                return DDzero;
            } else {
                r.w = cn.getCachedComplex(table[i].rw.r, table[i].rw.i);
            }

            return r;
        } else if (which == ad) {
            std::array<CTentry3, CTSLOTS> &table = CTable3.at(mode);
            ComplexValue aw{a.w.r->val, a.w.i->val};
            ComplexValue bw{b.w.r->val, b.w.i->val};
            const unsigned long i = CThash2(a.p, aw, b.p, bw, which);

            if (table[i].which != which) return r;
            if (table[i].a != a.p || !CN::equals(table[i].aw, aw)) return r;
            if (table[i].b != b.p || !CN::equals(table[i].bw, bw)) return r;

            CThit[which]++;
            r.p = table[i].r;

            if (std::fabs(table[i].rw.r) < CN::TOLERANCE && std::fabs(table[i].rw.i) < CN::TOLERANCE) {
                return DDzero;
            } else {
                r.w = cn.getCachedComplex(table[i].rw.r, table[i].rw.i);
            }

            return r;
        } else if (which == conjTransp || which == transp) {
            std::array<CTentry1, CTSLOTS> &table = CTable1.at(mode);
            const unsigned long i = CThash(a, b, which);

            if (table[i].which != which) return r;
            if (!equals(table[i].a, a)) return r;
            if (!equals(table[i].b, b)) return r;

            CThit[which]++;
            return table[i].r;
        } else {
            throw std::runtime_error("Undefined kind in CTlookup: " + std::to_string(which));
        }
    }

    // put an entry into the compute table
    void Package::CTinsert(const Edge &a, const Edge &b, const Edge &r, const CTkind which) {
        if (which == mult || which == fid || which == kron) {
            if (CN::equalsZero(a.w) || CN::equalsZero(b.w)) {
                std::cerr << "[WARN] CTinsert: Edge with near zero weight a.w=" << a.w << "  b.w=" << b.w << "\n";
            }
            assert(((std::uintptr_t) r.w.r & 1u) == 0 && ((std::uintptr_t) r.w.i & 1u) == 0);
            std::array<CTentry2, CTSLOTS> &table = CTable2.at(mode);
            const unsigned long i = CThash(a, b, which);

            table[i].a = a;
            table[i].b = b;
            table[i].which = which;
            table[i].r = r.p;
            table[i].rw.r = r.w.r->val;
            table[i].rw.i = r.w.i->val;
        } else if (which == ad) {
            std::array<CTentry3, CTSLOTS> &table = CTable3.at(mode);
            ComplexValue aw{a.w.r->val, a.w.i->val};
            ComplexValue bw{b.w.r->val, b.w.i->val};
            const unsigned long i = CThash2(a.p, aw, b.p, bw, which);

            assert(((std::uintptr_t) r.w.r & 1u) == 0 && ((std::uintptr_t) r.w.i & 1u) == 0);

            table[i].a = a.p;
            table[i].aw = aw;
            table[i].b = b.p;
            table[i].bw = bw;
            table[i].r = r.p;
            table[i].rw.r = r.w.r->val;
            table[i].rw.i = r.w.i->val;
            table[i].which = which;

        } else if (which == conjTransp || which == transp) {
            std::array<CTentry1, CTSLOTS> &table = CTable1.at(mode);
            const unsigned long i = CThash(a, b, which);

            table[i].a = a;
            table[i].b = b;
            table[i].which = which;
            table[i].r = r;
        } else {
            throw std::runtime_error("Undefined kind in CTinsert: " + std::to_string(which));
        }
    }

    inline unsigned short Package::TThash(const unsigned short n, const unsigned short t, const short line[]) {
        unsigned long i = t;
        for (unsigned short j = 0; j < n; j++) {
            if (line[j] == 1) {
                i = i << (3u + j);
            }
        }
        return i & TTMASK;
    }

    Edge Package::TTlookup(const unsigned short n, const unsigned short m, const unsigned short t, const short line[]) {
        Edge r{};
        r.p = nullptr;
        const unsigned short i = TThash(n, t, line);

        if (TTable[i].e.p == nullptr || TTable[i].t != t || TTable[i].m != m || TTable[i].n != n) {
            return r;
        }
        if (0 == std::memcmp(TTable[i].line, line, n * sizeof(short))) {
            return TTable[i].e;
        }
        return r;
    }

    void Package::TTinsert(unsigned short n, unsigned short m, unsigned short t, const short *line, const Edge &e) {
        const unsigned short i = TThash(n, t, line);
        TTable[i].n = n;
        TTable[i].m = m;
        TTable[i].t = t;
        std::memcpy(TTable[i].line, line, n * sizeof(short));
        TTable[i].e = e;
    }

    // make a DD nonterminal node and return an edge pointing to it
    // node is not recreated if it already exists
    Edge Package::makeNonterminal(const short v, const Edge *edge, const bool cached) {
        Edge e{getNode(), CN::ONE};
        assert(e.p->ref == 0);
        e.p->v = v;
        assert(e.p->v == v);
        assert(v - 1 == edge[0].p->v || isTerminal(edge[0]));
        assert(v - 1 == edge[1].p->v || isTerminal(edge[1]));
        assert(v - 1 == edge[2].p->v || isTerminal(edge[2]));
        assert(v - 1 == edge[3].p->v || isTerminal(edge[3]));

        std::memcpy(e.p->e, edge, NEDGE * sizeof(Edge));
        assert(e.p->v == v);
        e = normalize(e, cached); // normalize it
        assert(e.p->v == v || isTerminal(e));
//        e = UTlookup(e, true);  // why do you keep the node? This generates a memory leak!
        e = UTlookup(e, false);  // look it up in the unique tables

        assert(e.p->v == v || isTerminal(e));
        return e;          // return result
    }

    Edge Package::partialTrace(const Edge& a, const std::bitset<MAXN> &eliminate) {
        [[maybe_unused]] const auto before = cn.cacheCount;
        const auto result = trace(a, eliminate);
        [[maybe_unused]] const auto after = cn.cacheCount;
        assert(before == after);
        return result;
    }

    ComplexValue Package::trace(const Edge& a) {
        auto eliminate = std::bitset<MAXN>{}.set();
        [[maybe_unused]] const auto before = cn.cacheCount;
        Edge res = partialTrace(a, eliminate);
        [[maybe_unused]] const auto after = cn.cacheCount;
        assert(before == after);
        return {ComplexNumbers::val(res.w.r), ComplexNumbers::val(res.w.i)};
    }

    // adds two matrices represented by DD
    // the two DD should have the same variable set and ordering
    Edge Package::add2(const Edge& x, const Edge& y) {
        if (x.p == nullptr) {
            return y;  // handles partial matrices i.e.
        }
        if (y.p == nullptr) {
            return x;  // column and row vetors
        }
        nOps[ad]++;

        if (x.w == CN::ZERO) {
            if (y.w == CN::ZERO) {
                return y;
            }
            auto r = y;
            r.w = cn.getCachedComplex(CN::val(y.w.r), CN::val(y.w.i));
            return r;
        }
        if (y.w == CN::ZERO) {
            auto r = x;
        	r.w = cn.getCachedComplex(CN::val(x.w.r), CN::val(x.w.i));
            return r;
        }
        if (x.p == y.p) {
            Edge r = y;
            r.w = cn.addCached(x.w, y.w);
            if (CN::equalsZero(r.w)) {
                cn.releaseCached(r.w);
                return DDzero;
            }
            return r;
        }

        Edge r = CTlookup(x, y, ad);
        if (r.p != nullptr) {
            return r;
        }

        short w;
        if (isTerminal(x)) {
            w = y.p->v;
        } else {
            w = x.p->v;
            if (!isTerminal(y) && y.p->v > w) {
                w = y.p->v;
            }
        }

        Edge e1{}, e2{}, e[NEDGE];
        for (int i = 0; i < NEDGE; i++) {
            if (!isTerminal(x) && x.p->v == w) {
                e1 = x.p->e[i];

                if (e1.w != CN::ZERO) {
                    e1.w = cn.mulCached(e1.w, x.w);
                }
            } else {
                e1 = x;
                if (y.p->e[i].p == nullptr) {
                    e1 = {nullptr, CN::ZERO};
                }
            }
            if (!isTerminal(y) && y.p->v == w) {
                e2 = y.p->e[i];

                if (e2.w != CN::ZERO) {
                    e2.w = cn.mulCached(e2.w, y.w);
                }
            } else {
                e2 = y;
                if (x.p->e[i].p == nullptr) {
                    e2 = {nullptr, CN::ZERO};
                }
            }

            e[i] = add2(e1, e2);

            if (!isTerminal(x) && x.p->v == w && e1.w != CN::ZERO) {
                cn.releaseCached(e1.w);
            }

            if (!isTerminal(y) && y.p->v == w && e2.w != CN::ZERO) {
                cn.releaseCached(e2.w);
            }
        }

        r = makeNonterminal(w, e, true);

        CTinsert(x, y, r, ad);

        return r;
    }

    Edge Package::add(const Edge& x, const Edge& y) {
        [[maybe_unused]] const auto before = cn.cacheCount;
        Edge result = add2(x, y);

        if (result.w != CN::ZERO) {
            cn.releaseCached(result.w);
            result.w = cn.lookup(result.w);
        }
        [[maybe_unused]] const auto after = cn.cacheCount;
        assert(after == before);
        return result;
    }

    // new multiply routine designed to handle missing variables properly
    // var is number of variables
    Edge Package::multiply2(const Edge& x, const Edge& y, unsigned short var) {
        if (x.p == nullptr)
            return x;
        if (y.p == nullptr)
            return y;

        nOps[mult]++;

        if (x.w == CN::ZERO || y.w == CN::ZERO) {
            return DDzero;
        }

        if (var == 0) {
            return makeTerminal(cn.mulCached(x.w, y.w));
        }

        auto xCopy = x;
        xCopy.w = CN::ONE;
        auto yCopy = y;
        yCopy.w = CN::ONE;

        Edge r = CTlookup(xCopy, yCopy, mult);
        if (r.p != nullptr) {
            if (r.w != CN::ZERO) {
                CN::mul(r.w, r.w, x.w);
                CN::mul(r.w, r.w, y.w);
                if (CN::equalsZero(r.w)) {
                    cn.releaseCached(r.w);
                    return DDzero;
                }
            }
            return r;
        }

        const auto w = static_cast<short>(var - 1);

        if (x.p->v == w && x.p->v == y.p->v) {
            if (x.p->ident) {
                if (y.p->ident) {
                    r = makeIdent(0, w);
                } else {
                    r = yCopy;
                }
                CTinsert(xCopy, yCopy, r, mult);
                r.w = cn.mulCached(x.w, y.w);
                if (CN::equalsZero(r.w)) {
                    cn.releaseCached(r.w);
                    return DDzero;
                }
                return r;
            }
            if (y.p->ident) {
                r = xCopy;
                CTinsert(xCopy, yCopy, r, mult);
                r.w = cn.mulCached(x.w, y.w);

                if (dd::ComplexNumbers::equalsZero(r.w)) {
                    cn.releaseCached(r.w);
                    return DDzero;
                }
                return r;
            }
        }

        Edge e1{}, e2{}, e[NEDGE];
        for (int i = 0; i < NEDGE; i += RADIX) {
            for (int j = 0; j < RADIX; j++) {
                e[i + j] = DDzero;
                for (int k = 0; k < RADIX; k++) {
                    if (!isTerminal(x) && x.p->v == w) {
                        e1 = x.p->e[i + k];
                    } else {
                        e1 = xCopy;
                    }
                    if (!isTerminal(y) && y.p->v == w) {
                        e2 = y.p->e[j + RADIX * k];
                    } else {
                        e2 = yCopy;
                    }

                    Edge m = multiply2(e1, e2, var - 1);

                    if (k == 0 || e[i + j].w == CN::ZERO) {
                        e[i + j] = m;
                    } else if (m.w != CN::ZERO) {
                        Edge old_e = e[i + j];

                        e[i + j] = add2(e[i + j], m);

                        cn.releaseCached(old_e.w);
                        cn.releaseCached(m.w);
                    }
                }
            }
        }
        r = makeNonterminal(w, e, true);

        CTinsert(xCopy, yCopy, r, mult);

        if (r.w != CN::ZERO && (x.w != CN::ONE || y.w != CN::ONE)) {
            if (r.w == CN::ONE) {
                r.w = cn.mulCached(x.w, y.w);
            } else {
                CN::mul(r.w, r.w, x.w);
                CN::mul(r.w, r.w, y.w);
            }
            if (CN::equalsZero(r.w)) {
                cn.releaseCached(r.w);
                return DDzero;
            }
        }
        return r;
    }

    Edge Package::multiply(const Edge& x, const Edge& y) {
        [[maybe_unused]] const auto before = cn.cacheCount;
        unsigned short var = 0;
        if (!isTerminal(x)) {
            var = x.p->v + 1;
        }
        if (!isTerminal(y) && (y.p->v + 1) > var) {
            var = y.p->v + 1;
        }

        Edge e = multiply2(x, y, var);

        if (e.w != ComplexNumbers::ZERO && e.w != ComplexNumbers::ONE) {
            cn.releaseCached(e.w);
            e.w = cn.lookup(e.w);
        }
        [[maybe_unused]] const auto after = cn.cacheCount;
        assert(before == after);

        return e;
    }

    // returns a pointer to the transpose of the matrix a points to
    Edge Package::transpose(const Edge &a) {
        if (a.p == nullptr || isTerminal(a) || a.p->symm) {
            return a;
        }

        Edge r = CTlookup(a, a, transp);     // check in compute table
        if (r.p != nullptr) {
            return r;
        }

        r = makeNonterminal(a.p->v, {transpose(a.p->e[0]), transpose(a.p->e[2]), transpose(a.p->e[1]), transpose(a.p->e[3])});           // create new top vertex
        // adjust top weight
        Complex c = cn.getTempCachedComplex();
        CN::mul(c, r.w, a.w);
        r.w = cn.lookup(c);

        CTinsert(a, a, r, transp);      // put in compute table
        return r;
    }

    // returns a pointer to the conjugate transpose of the matrix pointed to by a
    Edge Package::conjugateTranspose(const Edge& a) {
        if (a.p == nullptr)
            return a;          // NULL pointer
        if (isTerminal(a)) {              // terminal case
            auto r = a;
        	r.w = dd::ComplexNumbers::conj(a.w);
            return r;
        }

        Edge r = CTlookup(a, a, conjTransp);  // check if in compute table
        if (r.p != nullptr) {
            return r;
        }

        Edge e[NEDGE];
        // conjugate transpose submatrices and rearrange as required
        e[0] = conjugateTranspose(a.p->e[0]);
        e[1] = conjugateTranspose(a.p->e[2]);
        e[2] = conjugateTranspose(a.p->e[1]);
        e[3] = conjugateTranspose(a.p->e[3]);
        r = makeNonterminal(a.p->v, e);    // create new top node

        Complex c = cn.getTempCachedComplex();
        CN::mul(c, r.w, dd::ComplexNumbers::conj(a.w));  // adjust top weight including conjugate
        r.w = cn.lookup(c);

        CTinsert(a, a, r, conjTransp); // put it in the compute table
        return r;
    }

    // build a DD for the identity matrix for variables x to y (x<y)
    Edge Package::makeIdent(short x, short y) {
        if (y < 0)
            return DDone;

        if (x == 0 && IdTable[y].p != nullptr) {
            return IdTable[y];
        }
        if (y >= 1 && (IdTable[y - 1]).p != nullptr) {
            IdTable[y] = makeNonterminal(y, {IdTable[y - 1], DDzero, DDzero, IdTable[y - 1]});
            return IdTable[y];
        }

        Edge e = makeNonterminal(x, {DDone, DDzero, DDzero, DDone});
        for (int k = x + 1; k <= y; k++) {
            e = makeNonterminal(k, {e, DDzero, DDzero, e});
        }
        if (x == 0)
            IdTable[y] = e;
        return e;
    }

    // build matrix representation for a single gate on a circuit with n lines
    // line is the vector of connections
    // -1 not connected
    // 0...1 indicates a control by that value
    // 2 indicates the line is the target
    Edge Package::makeGateDD(const Matrix2x2 &mat, unsigned short n, const short *line) {
        Edge em[NEDGE], fm[NEDGE];
        short z = 0;

        for (int i = 0; i < RADIX; i++) {
            for (int j = 0; j < RADIX; j++) {
                if (mat[i][j].r == 0.0 && mat[i][j].i == 0.0) {
                    em[i * RADIX + j] = DDzero;
                } else {
                    em[i * RADIX + j] = makeTerminal(cn.lookup(mat[i][j]));
                }
            }
        }

        Edge e = DDone;
        Edge f{};
        for (z = 0; line[z] < RADIX; z++) { //process lines below target
            if (line[z] >= 0) { //  control line below target in DD
                for (int i1 = 0; i1 < RADIX; i1++) {
                    for (int i2 = 0; i2 < RADIX; i2++) {
                        int i = i1 * RADIX + i2;
                        if (i1 == i2) {
                            f = e;
                        } else {
                            f = DDzero;
                        }
                        for (int k = 0; k < RADIX; k++) {
                            for (int j = 0; j < RADIX; j++) {
                                int t = k * RADIX + j;
                                if (k == j) {
                                    if (k == line[z]) {
                                        fm[t] = em[i];
                                    } else {
                                        fm[t] = f;
                                    }
                                } else {
                                    fm[t] = DDzero;
                                }
                            }
                        }
                        em[i] = makeNonterminal(z, fm);
                    }
                }
            } else { // not connected
                for (auto &edge : em) {
                    for (int i1 = 0; i1 < RADIX; ++i1) {
                        for (int i2 = 0; i2 < RADIX; ++i2) {
                            if (i1 == i2) {
                                fm[i1 + i2 * RADIX] = edge;
                            } else {
                                fm[i1 + i2 * RADIX] = DDzero;
                            }
                        }
                    }
                    edge = makeNonterminal(z, fm);
                }
            }
            e = makeIdent(0, z);
        }
        e = makeNonterminal(z, em);  // target line
        for (z++; z < n; z++) { // go through lines above target
            if (line[z] >= 0) { //  control line above target in DD
                Edge temp = makeIdent(0, static_cast<short>(z - 1));
                for (int i = 0; i < RADIX; i++) {
                    for (int j = 0; j < RADIX; j++) {
                        if (i == j) {
                            if (i == line[z]) {
                                em[i * RADIX + j] = e;
                            } else {
                                em[i * RADIX + j] = temp;
                            }
                        } else {
                            em[i * RADIX + j] = DDzero;
                        }
                    }
                }
                e = makeNonterminal(z, em);
            } else { // not connected
                for (int i1 = 0; i1 < RADIX; i1++) {
                    for (int i2 = 0; i2 < RADIX; i2++) {
                        if (i1 == i2) {
                            fm[i1 + i2 * RADIX] = e;
                        } else {
                            fm[i1 + i2 * RADIX] = DDzero;
                        }
                    }
                }
                e = makeNonterminal(z, fm);
            }
        }
        return e;
    }

    Edge Package::makeGateDD(const std::array<ComplexValue, NEDGE> &mat, unsigned short n,
                             const std::array<short, MAXN> &line) {
        std::array<Edge, NEDGE> em{};
        short z = 0;

        for (int i = 0; i < NEDGE; ++i) {
            if (mat[i].r == 0 && mat[i].i == 0) {
                em[i] = DDzero;
            } else {
                em[i] = makeTerminal(cn.lookup(mat[i]));
            }
        }

        //process lines below target
        for (z = 0; line[z] < RADIX; z++) {
            for (int i1 = 0; i1 < RADIX; i1++) {
                for (int i2 = 0; i2 < RADIX; i2++) {
                    int i = i1 * RADIX + i2;
                    if (line[z] == 0) { // neg. control
                        em[i] = makeNonterminal(z, {em[i], DDzero, DDzero,
                                                    (i1 == i2) ? makeIdent(0, static_cast<short>(z - 1)) : DDzero});
                    } else if (line[z] == 1) { // pos. control
                        em[i] = makeNonterminal(z,
                                                {(i1 == i2) ? makeIdent(0, static_cast<short>(z - 1)) : DDzero, DDzero, DDzero, em[i]});
                    } else { // not connected
                        em[i] = makeNonterminal(z, {em[i], DDzero, DDzero, em[i]});
                    }
                }
            }
        }

        // target line
        Edge e = makeNonterminal(z, em);

        //process lines above target
        for (z++; z < n; z++) {
            if (line[z] == 0) { //  neg. control
                e = makeNonterminal(z, {e, DDzero, DDzero, makeIdent(0, static_cast<short>(z - 1))});
            } else if (line[z] == 1) { // pos. control
                e = makeNonterminal(z, {makeIdent(0, static_cast<short>(z - 1)), DDzero, DDzero, e});
            } else { // not connected
                e = makeNonterminal(z, {e, DDzero, DDzero, e});
            }
        }
        return e;
    }

    ComplexValue Package::innerProduct(const Edge& x, const Edge& y, int var) {
        if (x.p == nullptr || y.p == nullptr || CN::equalsZero(x.w) || CN::equalsZero(y.w)) { // the 0 case
            return {0.0, 0.0};
        }

        if (var == 0) {
            Complex c = cn.getTempCachedComplex();
            CN::mul(c, x.w, y.w);
            return {c.r->val, c.i->val};
        }

        auto xCopy = x;
        xCopy.w = CN::ONE;
        auto yCopy = y;
        yCopy.w = CN::ONE;

        Edge r = CTlookup(xCopy, yCopy, fid);
        if (r.p != nullptr) {
            if (r.w != CN::ZERO && r.w != CN::ONE) {
                cn.releaseCached(r.w);
            } else {
                // r.w == CN::ZERO/CN::ONE
                // In the mul statement below, the result is written to r.w
                r.w = cn.getTempCachedComplex(0,0);
            }
            CN::mul(r.w, r.w, x.w);
            CN::mul(r.w, r.w, y.w);
            return {CN::val(r.w.r), CN::val(r.w.i)};
        }

        short w = var - 1;
        ComplexValue sum{0.0, 0.0};

        Edge e1{}, e2{};
        for (int i = 0; i < NEDGE; i++) {
            if (!isTerminal(x) && x.p->v == w) {
                e1 = x.p->e[i];
            } else {
                e1 = xCopy;
            }
            if (!isTerminal(y) && y.p->v == w) {
                e2 = y.p->e[i];
                e2.w = CN::conj(e2.w);
            } else {
                e2 = yCopy;
            }
            ComplexValue cv = innerProduct(e1, e2, var - 1);

            sum.r += cv.r;
            sum.i += cv.i;
        }

        r = DDzero;
        r.w = cn.getTempCachedComplex(sum.r, sum.i);

        CTinsert(xCopy, yCopy, r, fid);
        CN::mul(r.w, r.w, x.w);
        CN::mul(r.w, r.w, y.w);

        return {CN::val(r.w.r), CN::val(r.w.i)};
    }

    ComplexValue Package::innerProduct(const Edge& x, const Edge& y) {
        if (x.p == nullptr || y.p == nullptr || CN::equalsZero(x.w) || CN::equalsZero(y.w)) { // the 0 case
            return {0, 0};
        }

        [[maybe_unused]] const auto before = cn.cacheCount;
        short w = x.p->v;
        if (y.p->v > w) {
            w = y.p->v;
        }
        const ComplexValue ip = innerProduct(x, y, w + 1);

        [[maybe_unused]] const auto after = cn.cacheCount;
        assert(after == before);
        return ip;
    }

    fp Package::fidelity(const Edge& x, const Edge& y) {
        const ComplexValue fid = innerProduct(x, y);
        return fid.r * fid.r + fid.i * fid.i;
    }

    Edge Package::kronecker(const Edge& x, const Edge& y) {
        Edge e = kronecker2(x, y);

        if (e.w != CN::ZERO && e.w != CN::ONE) {
            cn.releaseCached(e.w);
            e.w = cn.lookup(e.w);
        }

        return e;
    }

    Edge Package::kronecker2(const Edge& x, const Edge& y) {

        if (CN::equalsZero(x.w))
            return DDzero;

        nOps[kron]++;

        if (isTerminal(x)) {
            Edge r = y;
            r.w = cn.mulCached(x.w, y.w);
            return r;
        }

        Edge r = CTlookup(x, y, kron);
        if (r.p != nullptr)
            return r;

        if (x.p->ident) {
            r = makeNonterminal(static_cast<short>(y.p->v + 1), {y, DDzero, DDzero, y});
            for (int i = 0; i < x.p->v; ++i) {
                r = makeNonterminal(static_cast<short>(r.p->v + 1), {r, DDzero, DDzero, r});
            }

            r.w = cn.getCachedComplex(CN::val(y.w.r), CN::val(y.w.i));
            CTinsert(x, y, r, kron);
            return r;
        }

        Edge e0 = kronecker2(x.p->e[0], y);
        Edge e1 = kronecker2(x.p->e[1], y);
        Edge e2 = kronecker2(x.p->e[2], y);
        Edge e3 = kronecker2(x.p->e[3], y);

        r = makeNonterminal(static_cast<short>(y.p->v + x.p->v + 1), {e0, e1, e2, e3}, true);
        CN::mul(r.w, r.w, x.w);
        CTinsert(x, y, r, kron);
        return r;
    }

    Edge Package::extend(const Edge& e, unsigned short h, unsigned short l) {
        Edge f = (l > 0) ? kronecker(e, makeIdent(0, static_cast<short>(l - 1))) : e;
        Edge g = (h > 0) ? kronecker(makeIdent(0, static_cast<short>(h - 1)), f) : f;
        return g;
    }


    Edge Package::trace(const Edge& a, const std::bitset<MAXN> &eliminate, unsigned short alreadyEliminated) {
        short v = a.p->v;

        if (CN::equalsZero(a.w)) return DDzero;

	    if (eliminate.none()) return a;

	    // Base case
        if (v == -1) {
            if (isTerminal(a)) return a;
            std::cerr << "Expected terminal node in trace." << std::endl;
            exit(1);
        }

        if (eliminate[v]) {
	        auto elims = alreadyEliminated+1;
	        Edge r = DDzero;

            auto t0 = trace(a.p->e[0], eliminate, elims);
            r = add2(r, t0);
            auto r1 = r;

            auto t1 = trace(a.p->e[3], eliminate, elims);
            r = add2(r, t1);
            auto r2 = r;

            if (r.w == CN::ONE) {
		        r.w = a.w;
	        } else {
		        auto c = cn.getTempCachedComplex();
		        CN::mul(c, r.w, a.w);
		        r.w = cn.lookup(c); // better safe than sorry. this may result in complex values with magnitude > 1 in the complex table
	        }

	        if (r1.w != CN::ZERO) {
                cn.releaseCached(r1.w);
	        }

            if (r2.w != CN::ZERO) {
                cn.releaseCached(r2.w);
            }

            return r;
        } else {
        	auto adjustedV = static_cast<short>(a.p->v - (eliminate.count() - alreadyEliminated));
            Edge r = makeNonterminal(adjustedV, {trace(a.p->e[0], eliminate, alreadyEliminated),
                                              trace(a.p->e[1], eliminate, alreadyEliminated),
                                              trace(a.p->e[2], eliminate, alreadyEliminated),
                                              trace(a.p->e[3], eliminate, alreadyEliminated)}, false);
            if (r.w == CN::ONE) {
	            r.w = a.w;
            } else {
	            auto c = cn.getTempCachedComplex();
	            CN::mul(c, r.w, a.w);
	            r.w = cn.lookup(c);
            }
            return r;
        }
    }

    /**
     * Get a single element of the vector or matrix represented by the dd with root edge e
     * @param dd package where the dd lives
     * @param e edge pointing to the root node
     * @param elements string {0, 1, 2, 3}^n describing which outgoing edge should be followed
     *                 (for vectors 0 is the 0-successor and 2 is the 1-successor due to the shared representation)
     *                 If string is longer than required, the additional characters are ignored.
     * @return the complex value of the specified element
     */
    ComplexValue Package::getValueByPath(const Edge& e, std::string elements) {
        if (dd::Package::isTerminal(e)) {
            return {dd::ComplexNumbers::val(e.w.r), dd::ComplexNumbers::val(e.w.i)};
        }

        dd::Complex c = cn.getTempCachedComplex(1, 0);
        auto r = e;
        do {
            dd::ComplexNumbers::mul(c, c, r.w);
            int tmp = elements.at(r.p->v) - '0';
            assert(tmp >= 0 && tmp <= dd::NEDGE);
            r = r.p->e[tmp];
        } while (!dd::Package::isTerminal(r));
        dd::ComplexNumbers::mul(c, c, r.w);

        return {dd::ComplexNumbers::val(c.r), dd::ComplexNumbers::val(c.i)};
    }

    void Package::checkSpecialMatrices(NodePtr p) {
        p->ident = false;       // assume not identity
        p->symm = false;           // assume symmetric

        /****************** CHECK IF Symmetric MATRIX *****************/
        if (!p->e[0].p->symm || !p->e[3].p->symm) return;
        if (!equals(transpose(p->e[1]), p->e[2])) return;
        p->symm = true;

        /****************** CHECK IF Identity MATRIX ***********************/
        if (!(p->e[0].p->ident) || (p->e[1].w) != CN::ZERO || (p->e[2].w) != CN::ZERO || (p->e[0].w) != CN::ONE ||
            (p->e[3].w) != CN::ONE || !(p->e[3].p->ident))
            return;
        p->ident = true;
    }

    void Package::reset() {
        Unique = {}; // TODO: Return nodes from Unique to NodeAvail
        activeNodeCount = 0;
        maxActive = 0;
        for (unsigned short q = 0; q < MAXN; ++q)
            active[q] = 0;
        initComputeTable();
    }

}
