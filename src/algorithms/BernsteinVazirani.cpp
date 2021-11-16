/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/BernsteinVazirani.hpp"

namespace qc {
    BernsteinVazirani::BernsteinVazirani(const BitString& s, bool dynamic):
        s(s), dynamic(dynamic) {
        dd::QubitCount msb = 0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s.test(i))
                msb = i;
        }
        bitwidth = msb + 1;
        createCircuit();
    }

    BernsteinVazirani::BernsteinVazirani(dd::QubitCount nq, bool dynamic):
        bitwidth(nq), dynamic(dynamic) {
        auto distribution = std::bernoulli_distribution();
        for (dd::QubitCount i = 0; i < nq; ++i) {
            if (distribution(mt))
                s.set(i);
        }
        createCircuit();
    }

    BernsteinVazirani::BernsteinVazirani(const BitString& s, dd::QubitCount nq, bool dynamic):
        s(s), bitwidth(nq), dynamic(dynamic) {
        createCircuit();
    }

    std::ostream& BernsteinVazirani::printStatistics(std::ostream& os) const {
        os << "BernsteinVazirani (" << static_cast<std::size_t>(bitwidth) << ") Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(bitwidth + 1) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\ts: " << expected << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    void BernsteinVazirani::createCircuit() {
        name = "bv_" + s.to_string();

        expected = s.to_string();
        std::reverse(expected.begin(), expected.end());
        while (expected.length() > bitwidth)
            expected.pop_back();
        std::reverse(expected.begin(), expected.end());

        addQubitRegister(1, "flag");

        if (dynamic) {
            addQubitRegister(1, "q");
        } else {
            addQubitRegister(bitwidth, "q");
        }

        addClassicalRegister(bitwidth, "c");

        // prepare flag qubit
        x(0);

        if (dynamic) {
            for (dd::QubitCount i = 0; i < bitwidth; ++i) {
                // initial Hadamard
                h(1);

                // apply controlled-Z gate according to secret bitstring
                if (s.test(i))
                    z(0, 1_pc);

                // final Hadamard
                h(1);

                // measure result
                measure(1, i);

                // reset qubit if not finished
                if (i < bitwidth - 1)
                    reset(1);
            }
        } else {
            // initial Hadamard transformation
            for (dd::QubitCount i = 1; i <= bitwidth; ++i) {
                h(static_cast<dd::Qubit>(i));
            }

            // apply controlled-Z gates according to secret bitstring
            for (dd::QubitCount i = 1; i <= bitwidth; ++i) {
                if (s.test(i - 1))
                    z(0, dd::Control{static_cast<dd::Qubit>(i)});
            }

            // final Hadamard transformation
            for (dd::QubitCount i = 1; i <= bitwidth; ++i) {
                h(static_cast<dd::Qubit>(i));
            }

            // measure results
            for (dd::QubitCount i = 1; i <= bitwidth; i++) {
                measure(static_cast<dd::Qubit>(i), i - 1);
            }
        }
    }
} // namespace qc
