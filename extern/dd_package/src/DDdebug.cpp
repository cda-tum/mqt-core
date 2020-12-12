/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDpackage.h"
#include <iomanip>


namespace dd {
    /**
     * Print node information in a single line
     * @param p pointer to node
     * @return string with information
     */
    std::string Package::debugnode_line(NodePtr p) const {
        if (p == DDzero.p) {
            return "terminal";
        }
        std::stringstream sst;
        sst << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(p) << std::dec
            << "[v=" << p->v << "(" << varOrder[p->v]
            << ") nf=" << p->normalizationFactor
            << " uc=" << p->reuseCount
            << " ref=" << p->ref
            << " hash=" << UThash(p)
            << " UT=" << UTcheck({p, CN::ZERO})
            << "]";
        return sst.str();
    }

    void Package::debugnode(NodePtr p) const {
        if (p == DDzero.p) {
            std::clog << "terminal\n";
            return;
        }
        std::clog << "Debug node: " << debugnode_line(p) << "\n";
        for (auto const &edge : p->e) {
            std::clog << "  " << std::hexfloat << std::setw(22) << CN::val(edge.w.r) << " " << std::setw(22) << CN::val(edge.w.i) << std::defaultfloat
                      << "i --> " << debugnode_line(edge.p) << "\n";
        }
        std::clog << std::flush;
    }

    void Package::check_node_is_really_gone(NodePtr pNode, Edge in) {
        if(isTerminal(in)) {
            return;
        }
        if(in.p == pNode) {
            throw std::runtime_error("Found forbidden node");
        }

        for(const auto& child: in.p->e) {
            check_node_is_really_gone(pNode, child);
        }
    }

    bool Package::is_locally_consistent_dd(Edge e) {
        assert(CN::ONE.r->val == 1 && CN::ONE.i->val == 0);
        assert(CN::ZERO.r->val == 0 && CN::ZERO.i->val == 0);

        const bool result = is_locally_consistent_dd2(e);

        if (!result) {
            export2Dot(e, "locally_inconsistent.dot", true, false);
        }

        return result;
    }

    bool Package::is_locally_consistent_dd2(Edge e) {
        auto *ptr_r = CN::get_sane_pointer(e.w.r);
        auto *ptr_i = CN::get_sane_pointer(e.w.i);

        if ((ptr_r->ref == 0 || ptr_i->ref == 0) && e.w != CN::ONE && e.w != CN::ZERO) {
            std::clog << "\nLOCAL INCONSISTENCY FOUND\nOffending Number: " << e.w << " (" << ptr_r->ref << ", " << ptr_i->ref << ")\n\n";
            debugnode(e.p);
            return false;
        }

        if (isTerminal(e)) {
            return true;
        }

        if (!isTerminal(e) && e.p->ref == 0) {
            std::clog << "\nLOCAL INCONSISTENCY FOUND: RC==0\n";
            debugnode(e.p);
            return false;
        }

        for (Edge &child : e.p->e) {
            if(child.p->v + 1 != e.p->v && !isTerminal(child)) {
                std::clog << "\nLOCAL INCONSISTENCY FOUND: Wrong V\n";
                debugnode(e.p);
                return false;
            }
            if (!isTerminal(child) && child.p->ref == 0) {
                std::clog << "\nLOCAL INCONSISTENCY FOUND: RC==0\n";
                debugnode(e.p);
                return false;
            }
            if (!is_locally_consistent_dd2(child)) {
                return false;
            }
        }
        return true;
    }

    bool Package::is_globally_consistent_dd(Edge e) {
        std::map<ComplexTableEntry *, long> weight_counter;
        std::map<NodePtr, unsigned long> node_counter;
        //std::clog << "CHECKING GLOBAL CONSISTENCY\n";
        //export2Dot(e, "globalc.dot", true, false);
        fill_consistency_counter(e, weight_counter, node_counter);

        //std::clog << "Weight Counter " << weight_counter.size() << "\n";
        //for (auto entry: weight_counter) {
        //    std::clog << "  " << entry.first << "  (val=" << CN::val(entry.first) << ") : " << entry.second << "\n";
        //}

        //std::clog << "Node Counter " << node_counter.size() << "\n";
        //for (auto entry: node_counter) {
        //    std::clog << "  " << entry.first << " (v=" << entry.first->v << ", nf=" << entry.first->normalizationFactor << ") : " << entry.second << "\n";
        //}

        check_consistency_counter(e, weight_counter, node_counter);

        return true;
    }

    void Package::fill_consistency_counter(Edge edge, std::map<ComplexTableEntry *, long> &weight_map, std::map<NodePtr, unsigned long> &node_map) {
        weight_map[CN::get_sane_pointer(edge.w.r)]++;
        weight_map[CN::get_sane_pointer(edge.w.i)]++;

        if (isTerminal(edge)) {
            return;
        }
        node_map[edge.p]++;
        for (Edge child : edge.p->e) {
            if (node_map[child.p] == 0) {
                fill_consistency_counter(child, weight_map, node_map);
            } else {
                node_map[child.p]++;
                weight_map[CN::get_sane_pointer(child.w.r)]++;
                weight_map[CN::get_sane_pointer(child.w.i)]++;
            }
        }
    }


    void Package::check_consistency_counter(Edge edge, const std::map<ComplexTableEntry *, long> &weight_map, const std::map<NodePtr, unsigned long> &node_map) {
        auto* r_ptr = CN::get_sane_pointer(edge.w.r);
        auto* i_ptr = CN::get_sane_pointer(edge.w.i);

        assert(edge.p->normalizationFactor == CN::ONE);

        if(weight_map.at(r_ptr) > r_ptr->ref && r_ptr != CN::ONE.r && r_ptr != CN::ZERO.i) {
            std::clog << "\nOffending weight: " <<  edge.w << "\n";
            std::clog << "Bits: " << std::hexfloat << CN::val(edge.w.r) << " " << CN::val(edge.w.i) << std::defaultfloat << "\n";
            debugnode(edge.p);
            throw std::runtime_error("Ref-Count mismatch for " + std::to_string(r_ptr->val) + "(r): " + std::to_string(weight_map.at(r_ptr)) + " occurences in DD but Ref-Count is only " + std::to_string(r_ptr->ref));
        }

        if(weight_map.at(i_ptr) > i_ptr->ref && i_ptr != CN::ZERO.i && i_ptr != CN::ONE.r) {
            std::clog << edge.w << "\n";
            debugnode(edge.p);
            throw std::runtime_error("Ref-Count mismatch for " + std::to_string(i_ptr->val) + "(i): " + std::to_string(weight_map.at(i_ptr)) + " occurences in DD but Ref-Count is only " + std::to_string(i_ptr->ref));
        }

        if (isTerminal(edge)) {
            return;
        }

        if (node_map.at(edge.p) != edge.p->ref) {
            debugnode(edge.p);
            throw std::runtime_error("Ref-Count mismatch for node: " + std::to_string(node_map.at(edge.p)) + " occurences in DD but Ref-Count is " + std::to_string(edge.p->ref));
        }
        for (Edge child : edge.p->e) {
            if(!isTerminal(child) && child.p->v != edge.p->v - 1) {
                std::clog << "child.p->v == " << child.p->v << "\n";
                std::clog << " edge.p->v == " << edge.p->v << "\n";
                debugnode(child.p);
                debugnode(edge.p);
                throw std::runtime_error("Variable level ordering seems wrong");
            }
            check_consistency_counter(child, weight_map, node_map);
        }
    }

    void Package::printUniqueTable(unsigned short n) {
        std::cout << "Unique Table: " << std::endl;
        for (int i = n - 1; i >= 0; --i) {
            auto &unique = Unique[i];
            std::cout << "\t" << i << ":" << std::endl;
            for (size_t key = 0; key < unique.size(); ++key) {
                auto p = unique[key];
                if (unique[key] != nullptr)
                    std::cout << key << ": ";
                while (p != nullptr) {
                    std::cout << "\t\t" << (uintptr_t) p << " " << p->ref << "\t";
                    p = p->next;
                }
                if (unique[key] != nullptr)
                    std::cout << std::endl;
            }
        }
    }

    // print number of active nodes for variables 0 to n-1
    void Package::printActive(const int n) {
        std::cout << "#printActive: " << activeNodeCount << ". ";
        for (int i = 0; i < n; i++) {
            std::cout << " " << active[i] << " ";
        }
        std::cout << "\n";
    }

    // displays DD package statistics
    void Package::statistics() {
        auto hitRatioAdd = CTlook[ad] == 0 ? 0 : (double) CThit[ad] / (double) CTlook[ad];
        auto hitRatioMul = CTlook[mult] == 0 ? 0 : (double) CThit[mult] / (double) CTlook[mult];
        auto hitRatioKron = ((CTlook[kron] == 0) ? 0 : (double) CThit[kron] / (double) CTlook[kron]);
        auto hitRatioRenormalize = ((CTlook[CTkind::renormalize] == 0) ? 0 : (double) CThit[CTkind::renormalize] / (double) CTlook[CTkind::renormalize]);


        std::cout << "\nDD statistics:"
                  << "\n  Current # nodes in UniqueTable: " << nodecount
                  << "\n  Total compute table lookups: " << CTlook[0] + CTlook[1] + CTlook[2]
                  << "\n  Number of operations:"
                  << "\n    add:  " << nOps[ad]
                  << "\n    mult: " << nOps[mult]
                  << "\n    kron: " << nOps[kron]
                  << "\n    renormalize: " << nOps[CTkind::renormalize]
                  << "\n  Compute table hit ratios (hits/looks/ratio):"
                  << "\n    adds: " << CThit[ad] << " / " << CTlook[ad] << " / " << hitRatioAdd
                  << "\n    mult: " << CThit[mult] << " / " << CTlook[mult] << " / " << hitRatioMul
                  << "\n    kron: " << CThit[kron] << " / " << CTlook[kron] << " / " << hitRatioKron
                  << "\n    renormalize: " << CThit[CTkind::renormalize] << " / " << CTlook[CTkind::renormalize] << " / " << hitRatioRenormalize
                  << "\n  UniqueTable:"
                  << "\n    Collisions: " << UTcol
                  << "\n    Matches:    " << UTmatch
                  << "\n" << std::flush;
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

    void Package::printVector(const Edge &e) {
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


    // a slightly better DD print utility
    void Package::printDD(const Edge &e, unsigned int limit) {
        unsigned short n = 0, i = 0;

        ListElementPtr first = newListElement();
        first->p = e.p;
        first->next = nullptr;
        first->w = 0;

        std::cout << "top edge weight " << e.w << "\n";
        ListElementPtr pnext = first;

        while (pnext != nullptr) {
            std::cout << pnext->p->ref << " ";

            std::cout << i << " \t|\t(" << pnext->p->v << ") \t";

            std::cout << "[";
            if (pnext->p != DDzero.p) {
                for (auto const&edge : pnext->p->e) {
                    if (edge.p == nullptr) {
                        std::cout << "NULL ";
                    } else {
                        if (!isTerminal(edge)) {
                            ListElementPtr q = first->next;
                            ListElementPtr lastq = first;
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

            std::cout << "] " << reinterpret_cast<std::uintptr_t>(pnext->p) << "\n";
            i++;
            if (i == limit) {
                std::cout << "Printing terminated after " << limit << " vertices\n";
                return;
            }
            pnext = pnext->next;
        }
    }
}