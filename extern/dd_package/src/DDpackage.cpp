/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDpackage.h"

namespace dd {

	Node Package::terminal{ nullptr, {{ nullptr, CN::ZERO}, { nullptr, CN::ZERO }, { nullptr, CN::ZERO }, { nullptr, CN::ZERO }}, 0, -1, true, true};
	constexpr Edge Package::DDzero;
	constexpr Edge Package::DDone;

    void Package::debugnode(NodePtr p) const {
       if (p == DDzero.p) {
            std::cout << "terminal\n";
            return;
        }
        std::cout << "Debug node" << (intptr_t) p << "\n";
        std::cout << "node v " << (int) varOrder[p->v] << " (" << (int) p->v << ") edges (w,p) ";
        for (auto const & i : p->e) {
            std::cout << i.w << " " << (intptr_t) i.p <<" || ";
        }
        std::cout << "ref " << p->ref << "\n" << std::flush;
    }

    ListElementPtr Package::newListElement() {
        ListElementPtr r;

        if (listAvail != nullptr) {   // get node from avail chain if possible
            r = listAvail;
	        listAvail = listAvail->next;
        } else {            // otherwise allocate 2000 new nodes
            r = new ListElement[CHUNK_SIZE];
            allocated_list_chunks.push_back(r);
            ListElementPtr r2 = r + 1;
	        listAvail = r2;
            for (int i = 0; i < CHUNK_SIZE-2; i++, r2++) {
                r2->next = r2+1;
            }
            r2->next = nullptr;
        }
        return r;
    }

    // a slightly better DD print utility
    void Package::printDD(Edge e, unsigned int limit) {
        ListElementPtr first, q, lastq, pnext;
        unsigned short n = 0, i = 0;

        first = newListElement();
        first->p = e.p;
        first->next = nullptr;
        first->w = 0;

        std::cout << "top edge weight " << e.w << "\n";
        pnext = first;

        while (pnext != nullptr) {
            std::cout << pnext->p->ref << " ";

            std::cout << i << " \t|\t(" << pnext->p->v << ") \t";

            std::cout << "[";
            if (pnext->p != DDzero.p) {
	            for (auto& edge : pnext->p->e) {
		            if (edge.p == nullptr) {
                        std::cout << "NULL ";
                    } else {
			            if (!isTerminal(edge)) {
                            q = first->next;
                            lastq = first;
				            while (q != nullptr && edge.p != q->p) {
                                lastq = q;
                                q = q->next;
                            }
                            if (q == nullptr) {
                                q = newListElement();
	                            q->p = edge.p;
                                q->next = nullptr;
                                q->w = n = n + 1;
                                lastq->next = q;
                            }
                            std::cout << " " << q->w << ":\t";
                        } else {
                            std::cout << "  T:\t";
                        }
			            std::cout << " (" << edge.w << ") ";
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

    ComplexValue Package::getVectorElement(Edge e, const unsigned long long element) {
        if (isTerminal(e)) {
            return {0,0};
        }
        Complex l = cn.getTempCachedComplex(1, 0);
        do {
            CN::mul(l, l, e.w);
            auto tmp = (element >> invVarOrder[e.p->v]) & 1u;
            e = e.p->e[2 * tmp];
        } while (!isTerminal(e));
        CN::mul(l, l, e.w);

        return {l.r->val, l.i->val};
    }

    void Package::printVector(Edge e) {
        unsigned long long element = 2u << invVarOrder[e.p->v];
        for (unsigned long long i = 0; i < element; i++) {
            ComplexValue amplitude = getVectorElement(e, i);
            for (int j = invVarOrder[e.p->v]; j >= 0; j--) {
                std::cout << ((i >> j) & 1u);
            }
            std::cout << ": " << amplitude << "\n";
        }
        std::cout << std::flush;
    }


    void Package::toDot(Edge e, std::ostream& oss, bool isVector) {
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
        first = newListElement();
        first->p = e.p;
        first->next = nullptr;
        first->w = 0;

        unsigned short n=0, i=0;

        nodes << "\"R\"";
        //füge Kante zwischen helper node und neuem Knoten hinzu
        if (CN::equalsOne(e.w)) {
            nodes << " [label=\"\", shape=point];\n";
            edges << "\"R\" -> \"0\"\n";
        } else {
            nodes << " [label=\"\", shape=point];\n";
            edges << R"("R" -> "0" [label="()" << e.w << ")\" ];\n";
        }


        pnext = first;
        while (pnext != nullptr) {
            /* Zeichne Knoten*/
            nodes << "\"" << i << "\" " << "[ label=\""
            << "q" << pnext->p->v
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
                            if (CN::equalsZero(pnext->p->e[j].w)) {
                                nodes << ", fillcolor=red, color=red";
                            } else if (CN::equalsOne(pnext->p->e[j].w)) {
                                edges << "\"" << i << "h" << j << "\"-> \"T\";\n";
                            } else {
                                edges << "\"" << i << "h" << j << R"("-> "T" [label= "()" << pnext->p->e[j].w << ")\", ];\n";
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
    void Package::export2Dot(Edge basic, const char *outputFilename, bool isVector, bool show) {
        std::ofstream init(outputFilename);
	    toDot(basic, init, isVector);
        init.close();

        if (show) {
            std::ostringstream oss;
            oss << "dot -Tsvg " << outputFilename << " -o " << outputFilename << ".svg";
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
    Edge Package::makeBasisState(unsigned short n, const std::bitset<64>& state) {
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

    Edge Package::normalize(Edge& e, bool cached) {
        int argmax = -1;

	    bool zero[] = { CN::equalsZero(e.p->e[0].w),
                        CN::equalsZero(e.p->e[1].w),
                        CN::equalsZero(e.p->e[2].w),
                        CN::equalsZero(e.p->e[3].w) };

	    for (int i=0; i < NEDGE; i++) {
	        if (zero[i] && e.p->e[i].w != CN::ZERO) {
	            cn.releaseCached(e.p->e[i].w);
                e.p->e[i] = DDzero;
	        }
	    }

	    /// --- Matrix treatment ---
	    if (forceMatrixNormalization || !zero[1] || !zero[3]) {
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
                    for (auto const & i : e.p->e) {
                        if (i.w != CN::ZERO){
                            cn.releaseCached(i.w);
                        }
                    }
                }
                return DDzero;
            }

		    // divide each entry by max
		    for (int i = 0; i < NEDGE; ++i) {
			    if (i == argmax) {
			    	if (cached) {
					    if (e.w == ComplexNumbers::ONE)
						    e.w = maxc;
					    else
                            CN::mul(e.w, e.w, maxc);
				    } else {
			    		if (e.w == ComplexNumbers::ONE) {
			    			e.w = maxc;
			    		} else {
						    auto c = cn.getTempCachedComplex();
                            CN::mul(c, e.w, maxc);
						    e.w = cn.lookup(c);
					    }
			    	}
				    e.p->e[i].w = ComplexNumbers::ONE;
			    } else {
			    	if (zero[i]) {
			    		if (cached && e.p->e[i].w != ComplexNumbers::ZERO)
			    			cn.releaseCached(e.p->e[i].w);
			    		e.p->e[i] = DDzero;
			    		continue;
			    	}
					if (cached && !zero[i] && e.p->e[i].w != ComplexNumbers::ONE) {
						cn.releaseCached(e.p->e[i].w);
					}
				    if (CN::equalsOne(e.p->e[i].w))
					    e.p->e[i].w = ComplexNumbers::ONE;
				    auto c = cn.getTempCachedComplex();
                    CN::div(c, e.p->e[i].w, maxc);
				    e.p->e[i].w = cn.lookup(c);
			    }
		    }
		    return e;
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
	            for (auto & i : e.p->e) {
		            if (i.p == nullptr && i.w != CN::ZERO) {
			            cn.releaseCached(i.w);
		            }
	            }
            }
            return DDzero;
        }

        sum = std::sqrt(sum / div);

        if (cached && e.p->e[argmax].w != CN::ONE) {
            e.w = e.p->e[argmax].w;
            e.w.r->val *= sum;
            e.w.i->val *= sum;
        } else {
            e.w = cn.lookup(ComplexNumbers::val(e.p->e[argmax].w.r) * sum, ComplexNumbers::val(e.p->e[argmax].w.i) * sum);
            if (CN::equalsZero(e.w)) {
                return DDzero;
            }
        }

        for (int j = 0; j < NEDGE; j++) {
	        if (j == argmax) {
		        e.p->e[j].w = cn.lookup((fp)1.0L / sum, 0);
		        if (e.p->e[j].w == CN::ZERO)
		        	e.p->e[j] = DDzero;
	        } else if (e.p->e[j].p != nullptr && !zero[j]) {
		        if (cached) {
			        cn.releaseCached(e.p->e[j].w);
                    cn.div(e.p->e[j].w, e.p->e[j].w, e.w);
                    e.p->e[j].w = cn.lookup(e.p->e[j].w);
                    if (e.p->e[j].w == CN::ZERO) {
                        e.p->e[j] = DDzero;
                    }
                } else {
       	            Complex c = cn.getTempCachedComplex();
                    cn.div(c, e.p->e[j].w, e.w);
                    e.p->e[j].w = cn.lookup(c);
                    if (e.p->e[j].w == CN::ZERO) {
                        e.p->e[j] = DDzero;
                    }
                }
            }
        }
        return e;
    }

	//  lookup a node in the unique table for the appropriate variable - if not found insert it
	//  only normalized nodes shall be stored.
	Edge Package::UTlookup(Edge& e) {
		// there is a unique terminal node
		if (isTerminal(e)) {
            e.p = DDzero.p;
            return e;
        }
        UTlookups++;

        std::uintptr_t key = 0;
        // note hash function shifts pointer values so that order is important
        // suggested by Dr. Nigel Horspool and helps significantly
        for (unsigned int i = 0; i < NEDGE; i++) {
            key += ((std::uintptr_t) (e.p->e[i].p) >> i)
                   + ((std::uintptr_t) (e.p->e[i].w.r) >> i)
                   + ((std::uintptr_t) (e.p->e[i].w.i) >> (i + 1));
        }
        key = key & HASHMASK;

        unsigned short v = e.p->v;
        NodePtr p = Unique[v][key]; // find pointer to appropriate collision chain
        while (p != nullptr)    // search for a match
        {
            if (std::memcmp(e.p->e, p->e, NEDGE * sizeof(Edge)) == 0) {
                // Match found
                e.p->next = nodeAvail;    // put node pointed to by e.p on avail chain
                nodeAvail = e.p;

                // NOTE: reference counting is to be adjusted by function invoking the table lookup
                UTmatch++;        // record hash table match

                e.p = p;// and set it to point to node found (with weight unchanged)
                return e;
            }

            UTcol++;        // record hash collision
            p = p->next;
        }
        e.p->next = Unique[v][key]; // if end of chain is reached, this is a new node
        Unique[v][key] = e.p;       // add it to front of collision chain

        nodecount++;          // count that it exists
        if (nodecount > peaknodecount)
	        peaknodecount = nodecount;

        if (!isTerminal(e))
        	checkSpecialMatrices(e);

        return e;                // and return
    }

    // set compute table to empty and
    // set toffoli gate table to empty and
    // set identity table to empty
    void Package::initComputeTable() {
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
        for (auto & i : IdTable) {
            i.p = nullptr;
        }
    }

	// a simple garbage collector that removes nodes with 0 ref count from the unique
	// tables placing them on the available space chain
	void Package::garbageCollect(bool force)
    {
	    if (!force && nodecount < currentNodeGCLimit && cn.count < currentComplexGCLimit) {
		    return;
	    } // do not collect if below current limits
	    int count = 0;
        int counta = 0;
        for (auto & variable : Unique) {
            for (auto & bucket : variable) {
                NodePtr lastp = nullptr;
                NodePtr p = bucket;
                while (p != nullptr) {
                    if (p->ref == 0) {
                        if (p == terminalNode){
                            std::cerr << "[ERROR] Tried to collect \n";
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
        } else {            // otherwise allocate 2000 new nodes
	        r = new Node[CHUNK_SIZE];
            allocated_node_chunks.push_back(r);
            NodePtr r2 = r + 1;
	        nodeAvail = r2;
	        for (int i = 0; i < CHUNK_SIZE-2; i++, r2++) {
                r2->next = r2+1;
            }
            r2->next = nullptr;
        }
        r->next = nullptr;
        r->ref = 0;            // set reference count to 0
	    r->ident = r->symm = false; // mark as not identity or symmetric
        return r;
    }

    // increment reference counter for node e points to
    // and recursively increment reference counter for
    // each child if this is the first reference
    //
    // a ref count saturates and remains unchanged if it has reached
    // MAXREFCNT
    void Package::incRef(Edge& e) {
	    dd::ComplexNumbers::incRef(e.w);
        if (isTerminal(e))
            return;

        if (e.p->ref == MAXREFCNT) {
            std::cerr << "[WARN] MAXREFCNT reached for e.w=" << e.w << ". Weight will never be collected.\n";
	        debugnode(e.p);
            return;
        }
        e.p->ref++;

        if (e.p->ref == 1) {
            if (!isTerminal(e)) {
	            for (auto& edge : e.p->e)
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
    void Package::decRef(Edge& e) {
	    dd::ComplexNumbers::decRef(e.w);

        if (isTerminal(e)) return;
        if (e.p->ref == MAXREFCNT) return;

        if (e.p->ref == 0) // ERROR CHECK
        {
            std::cerr <<"[ERROR] In decref: " << e.p->ref << " before decref\n";
	        debugnode(e.p);
            std::exit(1);
        }
        e.p->ref--;

        if (e.p->ref == 0) {
            if (!isTerminal(e)) {
	            for (auto& edge : e.p->e) {
		            if (edge.p != nullptr) {
			            decRef(edge);
                    }
                }
            }
            active[e.p->v]--;
            if (active[e.p->v] < 0) {
                std::cerr << "ERROR in decref\n";
                std::exit(1);
            }
            activeNodeCount--;
        }
    }

    // counting number of unique nodes in a DD
    unsigned int Package::nodeCount(const Edge e, std::unordered_set<NodePtr>& visited) const {
        visited.insert(e.p);

        unsigned int sum = 1;
        if (!isTerminal(e)) {
            for (const auto & edge : e.p->e) {
                if (edge.p != nullptr && !visited.count(edge.p)) {
                    sum += nodeCount(edge, visited);
                }
            }
        }
        return sum;
    }

    // counts number of unique nodes in a DD
    unsigned int Package::size(const Edge e) const {
        std::unordered_set<NodePtr> visited(NODECOUNT_BUCKETS); // 2e6
        visited.max_load_factor(10);
        visited.clear();
        return nodeCount(e, visited);
    }

	Edge Package::CTlookup(const Edge& a, const Edge& b, const CTkind which) {
    // Lookup a computation in the compute table
    // return NULL if not a match else returns result of prior computation
        Edge r{nullptr, {nullptr, nullptr}};
        CTlook[which]++;

        if (which == mult || which == fid || which == kron) {
            const unsigned long i = CThash(a, b, which);

            if (CTable2[i].which != which) return r;
            if (!equals(CTable2[i].a, a)) return r;
	        if (!equals(CTable2[i].b, b)) return r;

            CThit[which]++;
            r.p = CTable2[i].r;

            if (std::fabs(CTable2[i].rw.r) < CN::TOLERANCE && std::fabs(CTable2[i].rw.i) < CN::TOLERANCE) {
                return DDzero;
            } else {
                r.w = cn.getCachedComplex(CTable2[i].rw.r, CTable2[i].rw.i);
            }

            return r;
        } else if (which == ad) {
            ComplexValue aw{ a.w.r->val, a.w.i->val};
            ComplexValue bw{ b.w.r->val, b.w.i->val };
            const unsigned long i = CThash2(a.p, aw, b.p, bw, which);

            if (CTable3[i].which != which) return r;
            if (CTable3[i].a != a.p || !CN::equals(CTable3[i].aw, aw)) return r;
            if (CTable3[i].b != b.p || !CN::equals(CTable3[i].bw, bw)) return r;

            CThit[which]++;
            r.p = CTable3[i].r;

            if (std::fabs(CTable3[i].rw.r) < CN::TOLERANCE && std::fabs(CTable3[i].rw.i) < CN::TOLERANCE) {
                return DDzero;
            } else {
	            r.w = cn.getCachedComplex(CTable3[i].rw.r, CTable3[i].rw.i);
            }

            return r;
        } else if (which == conjTransp || which == transp) {
            const unsigned long i = CThash(a, b, which);

            if (CTable1[i].which != which) return r;
	        if (!equals(CTable1[i].a, a)) return r;
	        if (!equals(CTable1[i].b, b)) return r;

            CThit[which]++;
            return CTable1[i].r;

        } else {
            std::cerr << "Undefined kind in CTlookup: " << which << "\n";
            std::exit(1);
        }
    }

    // put an entry into the compute table
    void Package::CTinsert(const Edge& a, const Edge& b, const Edge& r, const CTkind which) {
        if (which == mult || which == fid || which == kron) {
            if (CN::equalsZero(a.w) || CN::equalsZero(b.w)) {
                std::cerr << "[WARN] CTinsert: Edge with near zero weight a.w=" << a.w << "  b.w=" << b.w << "\n";
            }
            assert(((std::uintptr_t)r.w.r & 1u) == 0 && ((std::uintptr_t)r.w.i & 1u) == 0);
            const unsigned long i = CThash(a, b, which);

            CTable2[i].a = a;
            CTable2[i].b = b;
            CTable2[i].which = which;
            CTable2[i].r = r.p;
            CTable2[i].rw.r = r.w.r->val;
            CTable2[i].rw.i = r.w.i->val;
        } else if (which == ad) {
	        ComplexValue aw{ a.w.r->val, a.w.i->val };
	        ComplexValue bw{ b.w.r->val, b.w.i->val };
            const unsigned long i = CThash2(a.p, aw, b.p, bw, which);

            assert(((std::uintptr_t)r.w.r & 1u) == 0 && ((std::uintptr_t)r.w.i & 1u) == 0);

            CTable3[i].a = a.p;
            CTable3[i].aw = aw;
            CTable3[i].b = b.p;
            CTable3[i].bw = bw;
            CTable3[i].r = r.p;
            CTable3[i].rw.r = r.w.r->val;
            CTable3[i].rw.i = r.w.i->val;
            CTable3[i].which = which;

        } else if (which == conjTransp || which == transp) {
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

    unsigned short Package::TThash(const unsigned short n, const unsigned short t, const short line[]) {
        unsigned long i = t;
        for (unsigned short j = 0; j < n; j++){
            if (line[j] == 1){
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

    void Package::TTinsert(unsigned short n, unsigned short m, unsigned short t, const short *line, const Edge& e) {
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
    	Edge e{ getNode(), CN::ONE};
        e.p->v = v;

        std::memcpy(e.p->e, edge, NEDGE * sizeof(Edge));
        e = normalize(e, cached); // normalize it
        e = UTlookup(e);  // look it up in the unique tables
        return e;          // return result
    }

    void Package::printInformation() {
        std::cout << DDversion
                  << "\n  compiled: " << __DATE__ << " " << __TIME__
                  << "\n  Complex size: " << sizeof(Complex) << " bytes (aligned " << alignof(Complex) << " bytes)"
                  << "\n  ComplexValue size: " << sizeof(ComplexValue) << " bytes (aligned " << alignof(ComplexValue) << " bytes)"
                  << "\n  ComplexNumbers size: " << sizeof(ComplexNumbers) << " bytes (aligned " << alignof(ComplexNumbers) << " bytes)"
                  << "\n  NodePtr size: " << sizeof(NodePtr) << " bytes (aligned " << alignof(NodePtr) << " bytes)"
                  << "\n  Edge size: " << sizeof(Edge) << " bytes (aligned " << alignof(Edge) << " bytes)"
                  << "\n  Node size: " << sizeof(Node) << " bytes (aligned " << alignof(Node) << " bytes)"
                  << "\n  CTentry1 size: " << sizeof(CTentry1) << " bytes (aligned " << alignof(CTentry1) << " bytes)"
                  << "\n  CTentry2 size: " << sizeof(CTentry2) << " bytes (aligned " << alignof(CTentry2) << " bytes)"
                  << "\n  CTentry3 size: " << sizeof(CTentry3) << " bytes (aligned " << alignof(CTentry3) << " bytes)"
                  << "\n  TTentry size: " << sizeof(TTentry) << " bytes (aligned " << alignof(TTentry) << " bytes)"
                  << "\n  Package size: " << sizeof(Package) << " bytes (aligned " << alignof(Package) << " bytes)"
                  << "\n  max variables: " << MAXN
                  << "\n  UniqueTable buckets: " << NBUCKET
                  << "\n  ComputeTable slots: " << CTSLOTS
                  << "\n  ToffoliTable slots: " << TTSLOTS
                  << "\n  garbage collection limit: " << GCLIMIT1
                  << "\n  garbage collection increment: " << GCLIMIT_INC
                  << "\n" << std::flush;
    }

    Package::Package() : cn(ComplexNumbers()) {
	    initComputeTable();  // init computed table to empty
        currentNodeGCLimit = GCLIMIT1; // set initial garbage collection limit
	    currentComplexGCLimit = ComplexNumbers::GCLIMIT1;

        for (unsigned short i = 0; i < MAXN; i++) //  set initial variable order to 0,1,2... from bottom up
        {
	        varOrder[i] = invVarOrder[i] = i;
        }
    }

    Package::~Package() {
        for(auto chunk : allocated_list_chunks) {
            delete[] chunk;
        }
        for(auto chunk : allocated_node_chunks) {
            delete[] chunk;
        }
    }

    Edge Package::partialTrace(const Edge a, const std::bitset<MAXN>& eliminate) {
        auto before = cn.cacheCount;
        const auto result = trace(a, a.p->v, eliminate);
        auto after = cn.cacheCount;
        assert(before == after);
        return result;
    }

    ComplexValue Package::trace(const Edge a) {
        auto eliminate = std::bitset<MAXN>{}.set();
        auto before = cn.cacheCount;
        Edge res = partialTrace(a, eliminate);
        auto after = cn.cacheCount;
        assert(before == after);
        return { ComplexNumbers::val(res.w.r), ComplexNumbers::val(res.w.i)};
    }

    // adds two matrices represented by DD
    // the two DD should have the same variable set and ordering
    Edge Package::add2(Edge x, Edge y) {
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
            y.w = cn.getCachedComplex(y.w.r->val, y.w.i->val);
            return y;
        }
        if (y.w == CN::ZERO) {
        	x.w = cn.getCachedComplex(x.w.r->val, x.w.i->val);
            return x;
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
            if (!isTerminal(y) && invVarOrder[y.p->v] > invVarOrder[w]) {
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
                    e1 = { nullptr, CN::ZERO};
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
                    e2 = { nullptr, CN::ZERO };
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

    Edge Package::add(Edge x, Edge y) {
        const auto before = cn.cacheCount;
        Edge result = add2(x, y);

        if (result.w != CN::ZERO) {
	        cn.releaseCached(result.w);
	        result.w = cn.lookup(result.w);
        }
        const auto after = cn.cacheCount;
        assert(after == before);
        return result;
    }

    // new multiply routine designed to handle missing variables properly
    // var is number of variables
    Edge Package::multiply2(Edge& x, Edge& y, unsigned short var) {
        if (x.p == nullptr)
            return x;
        if (y.p == nullptr)
            return y;

        nOps[mult]++;

        if (x.w == CN::ZERO || y.w == CN::ZERO)  {
            return DDzero;
        }

        if (var == 0) {
	        return makeTerminal(cn.mulCached(x.w, y.w));
        }

        const Complex xweight = x.w;
        const Complex yweight = y.w;
        x.w = CN::ONE;
        y.w = CN::ONE;

        Edge r = CTlookup(x, y, mult);
        if (r.p != nullptr) {
            if (r.w != CN::ZERO) {
                CN::mul(r.w, r.w, xweight);
                CN::mul(r.w, r.w, yweight);
                if (CN::equalsZero(r.w)) {
                    cn.releaseCached(r.w);
                    return DDzero;
                }
            }
            return r;
        }

        const short w = varOrder[var - 1];

        if (x.p->v == w && x.p->v == y.p->v) {
            if (x.p->ident) {
            	if (y.p->ident) {
            		r = makeIdent(0, w);
            	} else {
                    r = y;
	            }
                CTinsert(x, y, r, mult);
	            r.w = cn.mulCached(xweight, yweight);
                if (CN::equalsZero(r.w)) {
                    cn.releaseCached(r.w);
                    return DDzero;
                }
                return r;
            }
            if (y.p->ident) {
                r = x;
                CTinsert(x, y, r, mult);
	            r.w = cn.mulCached(xweight, yweight);

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
                        e1 = x;
                    }
                    if (!isTerminal(y) && y.p->v == w) {
                        e2 = y.p->e[j + RADIX * k];
                    } else {
                        e2 = y;
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

        CTinsert(x, y, r, mult);

        if (r.w != CN::ZERO && (xweight != CN::ONE || yweight != CN::ONE)) {
        	if (r.w == CN::ONE) {
        		r.w = cn.mulCached(xweight, yweight);
        	} else {
                CN::mul(r.w, r.w, xweight);
                CN::mul(r.w, r.w, yweight);
            }
            if (CN::equalsZero(r.w)) {
                cn.releaseCached(r.w);
                return DDzero;
            }

        }
        return r;
    }

    Edge Package::multiply(Edge x, Edge y) {
        const auto before = cn.cacheCount;
        unsigned short var = 0;
        if (!isTerminal(x) && (invVarOrder[x.p->v] + 1) > var) {
            var = invVarOrder[x.p->v] + 1;
        }
        if (!isTerminal(y) && (invVarOrder[y.p->v] + 1) > var) {
            var = invVarOrder[y.p->v] + 1;
        }

        Edge e = multiply2(x, y, var);

        if (e.w != ComplexNumbers::ZERO && e.w != ComplexNumbers::ONE) {
	        cn.releaseCached(e.w);
	        e.w = cn.lookup(e.w);
        }
        const auto after = cn.cacheCount;
        assert(before == after);

        return e;
    }

	// returns a pointer to the transpose of the matrix a points to
	Edge Package::transpose(const Edge& a)
    {
        if (a.p == nullptr || isTerminal(a) || a.p->symm) {
            return a;
        }

        Edge r = CTlookup(a, a, transp);     // check in compute table
        if (r.p != nullptr) {
            return r;
        }

	    r = makeNonterminal(a.p->v, { transpose(a.p->e[0]), transpose(a.p->e[2]), transpose(a.p->e[1]), transpose(a.p->e[3])});           // create new top vertex
	    // adjust top weight
	    Complex c = cn.getTempCachedComplex();
        CN::mul(c, r.w, a.w);
	    r.w = cn.lookup(c);

        CTinsert(a, a, r, transp);      // put in compute table
	    return r;
    }

	// returns a pointer to the conjugate transpose of the matrix pointed to by a
	Edge Package::conjugateTranspose(Edge a)
    {
        if (a.p == nullptr)
            return a;          // NULL pointer
        if (isTerminal(a)) {              // terminal case
            a.w = dd::ComplexNumbers::conj(a.w);
            return a;
        }

        Edge r = CTlookup(a, a, conjTransp);  // check if in compute table
        if (r.p != nullptr) {
            return r;
        }

        Edge e[NEDGE];
	    // conjugate transpose submatrices and rearrange as required
        for (int i = 0; i < RADIX; i++) {
            for (int j = i; j < RADIX; j++) {
                e[i * RADIX + j] = conjugateTranspose(a.p->e[j * RADIX + i]);
                if (i != j)
                    e[j * RADIX + i] = conjugateTranspose(
		                    a.p->e[i * RADIX + j]);
            }
        }
	    r = makeNonterminal(a.p->v, e);    // create new top node

	    Complex c = cn.getTempCachedComplex();
        CN::mul(c, r.w, dd::ComplexNumbers::conj(a.w));  // adjust top weight including conjugate
        r.w = cn.lookup(c);

        CTinsert(a, a, r, conjTransp); // put it in the compute table
        return r;
    }

	// build a DD for the identity matrix for variables x to y (x<y)
	Edge Package::makeIdent(short x, short y)
    {
        if (y < 0)
            return DDone;

        if (x == 0 && IdTable[y].p != nullptr) {
            return IdTable[y];
        }
	    if (y >= 1 && (IdTable[y - 1]).p != nullptr) {
	        IdTable[y] = makeNonterminal(varOrder[y], { IdTable[y - 1], DDzero, DDzero, IdTable[y - 1] });
            return IdTable[y];
        }

	    Edge e = makeNonterminal(varOrder[x], { DDone, DDzero, DDzero, DDone });
        for (int k = x + 1; k <= y; k++) {
	        e = makeNonterminal(varOrder[k], { e, DDzero, DDzero, e });
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
    Edge Package::makeGateDD(const Matrix2x2& mat, unsigned short n, const short *line) {
        Edge em[NEDGE], fm[NEDGE];
        short w = 0, z = 0;

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
        for (z = 0; line[w = varOrder[z]] < RADIX; z++) { //process lines below target
            if (line[w] >= 0) { //  control line below target in DD
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
	                    em[i] = makeNonterminal(w, fm);
                    }
                }
            } else { // not connected
	            for (auto& edge : em) {
		            for (int i1 = 0; i1 < RADIX; ++i1) {
			            for (int i2 = 0; i2 < RADIX; ++i2) {
                            if (i1 == i2) {
	                            fm[i1 + i2 * RADIX] = edge;
                            } else {
                                fm[i1 + i2 * RADIX] = DDzero;
                            }
                        }
                    }
		            edge = makeNonterminal(w, fm);
                }
            }
            e = makeIdent(0, z);
        }
	    e = makeNonterminal(varOrder[z], em);  // target line
        for (z++; z < n; z++) { // go through lines above target
            if (line[w = varOrder[z]] >= 0) { //  control line above target in DD
                Edge temp = makeIdent(0, z - 1);
                for (int i = 0; i < RADIX; i++) {
                    for (int j = 0; j < RADIX; j++) {
                        if (i == j) {
                            if (i == line[w]) {
                                em[i * RADIX + j] = e;
                            } else {
                                em[i * RADIX + j] = temp;
                            }
                        } else {
                            em[i * RADIX + j] = DDzero;
                        }
                    }
                }
	            e = makeNonterminal(w, em);
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
	            e = makeNonterminal(w, fm);
            }
        }
        return e;
    }

	Edge Package::makeGateDD(const std::array<ComplexValue, NEDGE>& mat, unsigned short n, const std::array<short, MAXN>& line) {
		std::array<Edge, NEDGE> em{ };
		short w = 0, z = 0;

		for (int i = 0; i < NEDGE; ++i) {
			if (mat[i].r == 0 && mat[i].i == 0) {
				em[i] = DDzero;
			} else {
				em[i] = makeTerminal(cn.lookup(mat[i]));
			}
		}

		//process lines below target
		for (z = 0; line[w = varOrder.at(z)] < RADIX; z++) {
			for (int i1 = 0; i1 < RADIX; i1++) {
				for (int i2 = 0; i2 < RADIX; i2++) {
					int i = i1 * RADIX + i2;
					if (line[w] == 0) { // neg. control
						em[i] = makeNonterminal(w, { em[i], DDzero, DDzero, (i1 == i2) ? makeIdent(0, z - 1) : DDzero });
					} else if (line[w] == 1) { // pos. control
						em[i] = makeNonterminal(w, { (i1 == i2) ? makeIdent(0, z - 1) : DDzero, DDzero, DDzero, em[i] });
					} else { // not connected
						em[i] = makeNonterminal(w, { em[i], DDzero, DDzero, em[i] });
					}
				}
			}
		}

		// target line
		Edge e = makeNonterminal(varOrder[z], em);

		//process lines above target
		for (z++; z < n; z++) {
			w = varOrder[z];
			if (line[w] == 0) { //  neg. control
				e = makeNonterminal(w, { e, DDzero, DDzero, makeIdent(0, z - 1) });
			} else if (line[w] == 1) { // pos. control
				e = makeNonterminal(w, { makeIdent(0, z - 1), DDzero, DDzero, e });
			} else { // not connected
				e = makeNonterminal(w, { e, DDzero, DDzero, e });
			}
		}
		return e;
	}

	// displays DD package statistics
    void Package::statistics() {
    	auto hitRatioAdd = CTlook[ad] == 0? 0 :  (double) CThit[ad] / (double)CTlook[ad];
		auto hitRatioMul = CTlook[mult] == 0? 0 :  (double) CThit[mult] / (double)CTlook[mult];
		auto hitRatioKron = ((CTlook[kron] == 0) ? 0 : (double) CThit[kron] / (double)CTlook[kron]);


        std::cout << "\nDD statistics:"
                  << "\n  Current # nodes in UniqueTable: " << nodecount
                  << "\n  Total compute table lookups: " << CTlook[0] + CTlook[1] + CTlook[2]
                  << "\n  Number of operations:"
                  << "\n    add:  " << nOps[ad]
                  << "\n    mult: " << nOps[mult]
                  << "\n    kron: " << nOps[kron]
                  << "\n  Compute table hit ratios (hits/looks/ratio):"
                  << "\n    adds: " << CThit[ad] << " / " << CTlook[ad] << " / " << hitRatioAdd
                  << "\n    mult: " << CThit[mult] << " / " << CTlook[mult] << " / " << hitRatioMul
		          << "\n    kron: " << CThit[kron] << " / " << CTlook[kron] << " / " << hitRatioKron
                  << "\n  UniqueTable:"
                  << "\n    Collisions: " << UTcol
                  << "\n    Matches:    " << UTmatch
                  << "\n" << std::flush;
    }

    // print number of active nodes for variables 0 to n-1
    void Package::printActive(const int n) {
        std::cout << "#printActive: " << activeNodeCount << ". ";
        for (int i = 0; i < n; i++) {
            std::cout << " " << active[i] << " ";
        }
        std::cout << "\n";
    }

    ComplexValue Package::fidelity(Edge x, Edge y, int var) {
        if (x.p == nullptr || y.p == nullptr || CN::equalsZero(x.w) || CN::equalsZero(y.w))  // the 0 case
        {
            return {0.0,0.0};
        }

        if (var == 0) {
	        Complex c = cn.getTempCachedComplex();
            CN::mul(c, x.w, y.w);
            return {c.r->val, c.i->val};
        }

        Complex xweight = x.w;
        Complex yweight = y.w;
        x.w = CN::ONE;
        y.w = CN::ONE;

        Edge r = CTlookup(x, y, fid);
        if (r.p != nullptr) {
        	if (r.w != CN::ZERO && r.w != CN::ONE) {
		        cn.releaseCached(r.w);
	        }

            CN::mul(r.w, r.w, xweight);
            CN::mul(r.w, r.w, yweight);
            return {r.w.r->val, r.w.i->val};
        }

        short w = varOrder[var - 1];
        ComplexValue sum{ 0.0, 0.0};

        Edge e1{}, e2{};
        for (int i = 0; i < NEDGE; i++) {
            if (!isTerminal(x) && x.p->v == w) {
                e1 = x.p->e[i];
            } else {
                e1 = x;
            }
            if (!isTerminal(y) && y.p->v == w) {
                e2 = y.p->e[i];
                e2.w = CN::conj(e2.w);
            } else {
                e2 = y;
            }
            ComplexValue cv = fidelity(e1, e2, var - 1);

            sum.r += cv.r;
            sum.i += cv.i;
        }

        r = DDzero;
        r.w = cn.getTempCachedComplex(sum.r, sum.i);

        CTinsert(x, y, r, fid);
        CN::mul(r.w, r.w, xweight);
        CN::mul(r.w, r.w, yweight);

        return {r.w.r->val, r.w.i->val};
    }

    fp Package::fidelity(Edge x, Edge y) {
	    if (x.p == nullptr || y.p == nullptr || CN::equalsZero(x.w) || CN::equalsZero(y.w)) { // the 0 case
	        return 0;
	    }

        const auto before = cn.cacheCount;
        short w = invVarOrder[x.p->v];
        if(invVarOrder.at(y.p->v) > w) {
            w = invVarOrder[y.p->v];
        }
        const ComplexValue fid = fidelity(x, y, w + 1);

        const auto after = cn.cacheCount;
        assert(after == before);
        return fid.r*fid.r + fid.i*fid.i;
    }

    Edge Package::kronecker(Edge x, Edge y) {
	    Edge e = kronecker2(x, y);

	    if (e.w != CN::ZERO && e.w != CN::ONE) {
		    cn.releaseCached(e.w);
		    e.w = cn.lookup(e.w);
	    }

	    return e;
    }

	Edge Package::kronecker2(Edge x, Edge y) {

		if (CN::equalsZero(x.w))
			return DDzero;

    	nOps[kron]++;

		if (isTerminal(x)) {
			Edge r = y;
			r.w = cn.mulCached(x.w, y.w);
			return r;
		}

    	Edge r = CTlookup(x,y,kron);
    	if (r.p != nullptr)
		    return r;

    	if (x.p->ident) {
			r = makeNonterminal(y.p->v+1, {y, DDzero, DDzero, y});
			for (int i = 0; i < x.p->v; ++i) {
				r = makeNonterminal(r.p->v+1, {r, DDzero, DDzero, r});
			}

			r.w = cn.getCachedComplex(CN::val(y.w.r),CN::val(y.w.i));
		    CTinsert(x,y,r, kron);
		    return r;
    	}

		Edge e0 = kronecker2(x.p->e[0],y);
		Edge e1 = kronecker2(x.p->e[1],y);
		Edge e2 = kronecker2(x.p->e[2],y);
		Edge e3 = kronecker2(x.p->e[3],y);

		r = makeNonterminal(y.p->v+x.p->v+1, {e0, e1, e2, e3}, true);
	    CN::mul(r.w, r.w, x.w);
		CTinsert(x,y,r,kron);
		return r;
    }

	Edge Package::extend(Edge e, unsigned short h, unsigned short l) {
  	    Edge f = (l>0)? kronecker(e, makeIdent(0,l-1)) : e;
  	    Edge g = (h>0)? kronecker(makeIdent(0, h-1), f): f;
  	    return g;
    }


	Edge Package::trace(Edge a, short v, const std::bitset<MAXN>& eliminate) {
    	short w = invVarOrder[a.p->v];

		if (CN::equalsZero(a.w)) return DDzero;

    	// Base case
    	if (v == -1) {
    		if (isTerminal(a)) return a;
    		std::cerr << "Expected terminal node in trace." << std::endl;
		    exit(1);
    	}

    	if (eliminate[v]) {
    		if (v == w) {
			    Edge r = DDzero;
			    //std::cout << cn.cacheCount << " ";
			    auto t0 = trace(a.p->e[0], v - 1, eliminate);
			    r = add2(r, t0);
			    auto r1 = r;
			    //std::cout << "-> " << cn.cacheCount << " ";
			    auto t1 = trace(a.p->e[3], v - 1, eliminate);
			    r = add2(r, t1);
			    auto r2 = r;
			    //std::cout << "-> " << cn.cacheCount << std::endl;
                CN::mul(r.w, r.w, a.w);
				if (r1.w != CN::ZERO)
					cn.releaseCached(r1.w);

			    if (r2.w != CN::ZERO) {
					cn.releaseCached(r2.w);
				}
			    //cn.lookup(r.w);

			    return r;
    		} else {
			    Edge r = trace(a, v - 1, eliminate);
                CN::mul(r.w, r.w, cn.getTempCachedComplex(RADIX, 0));
			    return r;
    		}
    	} else {
    		if (v == w) {
			    Edge r = makeNonterminal(a.p->v, { trace(a.p->e[0], v - 1, eliminate),
			                                       trace(a.p->e[1], v - 1, eliminate),
			                                       trace(a.p->e[2], v - 1, eliminate),
			                                       trace(a.p->e[3], v - 1, eliminate) }, false);
                CN::mul(r.w, r.w, a.w);
			    return r;
		    } else {
			    return trace(a,v-1,eliminate);
    		}
    	}
    }

    ComplexValue Package::getValueByPath(dd::Edge e, std::string elements) {
        if(dd::Package::isTerminal(e)) {
            return {dd::ComplexNumbers::val(e.w.r), dd::ComplexNumbers::val(e.w.i)};
        }

        dd::Complex c = cn.getTempCachedComplex(1, 0);
        do {
            dd::ComplexNumbers::mul(c, c, e.w);
            int tmp = elements.at(invVarOrder.at(e.p->v))-'0';
            assert(tmp >= 0 && tmp <= dd::NEDGE);
            e = e.p->e[tmp];
        } while(!dd::Package::isTerminal(e));
        dd::ComplexNumbers::mul(c, c, e.w);

        return {dd::ComplexNumbers::val(c.r), dd::ComplexNumbers::val(c.i)};
    }

	void Package::checkSpecialMatrices(Edge &e) {
		e.p->ident = false;       // assume not identity
		e.p->symm = false;           // assume symmetric

		/****************** CHECK IF Symmetric MATRIX *****************/
		if (!e.p->e[0].p->symm || !e.p->e[3].p->symm) return;
		if (!equals(transpose(e.p->e[1]), e.p->e[2])) return;
		e.p->symm = true;

		/****************** CHECK IF Identity MATRIX ***********************/
		if(!(e.p->e[0].p->ident) || (e.p->e[1].w) != CN::ZERO || (e.p->e[2].w) != CN::ZERO || (e.p->e[0].w) != CN::ONE || (e.p->e[3].w) != CN::ONE || !(e.p->e[3].p->ident)) return;
		e.p->ident = true;
	}
}
