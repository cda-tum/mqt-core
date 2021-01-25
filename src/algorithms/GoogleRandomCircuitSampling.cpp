/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/GoogleRandomCircuitSampling.hpp"

namespace qc {
	GoogleRandomCircuitSampling::GoogleRandomCircuitSampling(const std::string& filename) {
        importGRCS(filename);

        for (unsigned short i = 0; i < nqubits; ++i) {
            initialLayout.insert({ i, i});
            outputPermutation.insert({ i, i});
        }
    }

    GoogleRandomCircuitSampling::GoogleRandomCircuitSampling(const std::string& pathPrefix, unsigned short device, unsigned short depth, unsigned short instance): pathPrefix(pathPrefix) {
        std::stringstream ss;
        ss << pathPrefix;
        ss << "bristlecone/cz_v2/bris_";
        ss << device;
        ss << "/bris_";
        ss << device;
        ss << "_";
        ss << depth;
        ss << "_";
        ss << instance;
        ss << ".txt";
        layout = Bristlecone;
        importGRCS(ss.str());

        for (unsigned short i = 0; i < nqubits; ++i) {
            initialLayout.insert({ i, i});
            outputPermutation.insert({ i, i});
        }
        qregs.insert({"q", std::pair<unsigned short, unsigned short>{0, nqubits}});
	    cregs.insert({"c", std::pair<unsigned short, unsigned short>{0, nqubits}});
    }

    GoogleRandomCircuitSampling::GoogleRandomCircuitSampling(const std::string& pathPrefix, unsigned short x, unsigned short y, unsigned short depth, unsigned short instance): pathPrefix(pathPrefix) {
        std::stringstream ss;
        ss << pathPrefix;
        ss << "rectangular/cz_v2/";
        ss << x;
        ss << "x";
        ss << y;
        ss << "/inst_";
        ss << x;
        ss << "x";
        ss << y;
        ss << "_";
        ss << depth;
        ss << "_";
        ss << instance;
        ss << ".txt";
        layout = Rectangular;
        importGRCS(ss.str());

        for (unsigned short i = 0; i < nqubits; ++i) {
            initialLayout.insert({ i, i});
            outputPermutation.insert({ i, i});
        }
	    qregs.insert({"q", {0, nqubits}});
	    cregs.insert({"c", {0, nqubits}});
    }

    void GoogleRandomCircuitSampling::importGRCS(const std::string& filename) {
        auto ifs = std::ifstream(filename);
        if (!ifs.good()) {
            std::cerr << "Error opening/reading from file: " << filename << std::endl;
            exit(3);
        }
        size_t slash = filename.find_last_of('/');
        size_t dot = filename.find_last_of('.');
        std::string benchmark = filename.substr(slash+1, dot-slash-1);
        name = benchmark;
        layout = (benchmark[0] == 'b')? Bristlecone: Rectangular;
        ifs >> nqubits;
        std::string line;
        std::string identifier;
        unsigned short control = 0;
        unsigned short target = 0;
        unsigned int cycle = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            ss >> cycle;
            if (cycles.size() <= cycle) {
                cycles.emplace_back(std::vector<std::unique_ptr<Operation>>{});
            }

            ss >> identifier;
            if (identifier == "cz") {
                ss >> control;
                ss >> target;
                cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, Control(control), target, Z));
            } else {
                ss >> target;
                if (identifier == "h")
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, H));
                else if (identifier == "t")
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, T));
                else if (identifier == "x_1_2")
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, RX, PI_2));
                else if (identifier == "y_1_2")
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, RY, PI_2));
                else {
                    std::cerr << "Unknown gate '" << identifier << "'\n";
                    exit(1);
                }
            }
        }
    }

    size_t GoogleRandomCircuitSampling::getNops() const {
        size_t nops = 0;
        for (const auto& cycle: cycles)
            nops += cycle.size();
        return nops;
    }

    std::ostream& GoogleRandomCircuitSampling::print(std::ostream& os) const {
        size_t i = 0;
        size_t j = 0;
        for (const auto& cycle:cycles) {
            os << "Cycle " << i++ << ":\n";
            for (const auto& op: cycle) {
                os << std::setw((int)std::log10(getNops())+1) << ++j << ": ";
                op->print(os, initialLayout);
                os << std::endl;
            }
        }
        return os;
    }

    std::ostream& GoogleRandomCircuitSampling::printStatistics(std::ostream& os) {
        os << "GoogleRandomCircuitSampling Statistics:\n";
        os << "\tLayout: " << ((layout == Rectangular)? "Rectangular": "Bristlecone") << std::endl;
        os << "\tn: " << nqubits << std::endl;
        os << "\tm: " << getNops() << std::endl;
        os << "\tc: 1 + " << cycles.size()-2 << " + 1" << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    dd::Edge GoogleRandomCircuitSampling::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
		std::array<short, MAX_QUBITS> line{};
        line.fill(LINE_DEFAULT);
        permutationMap map = initialLayout;
        dd->setMode(dd::Matrix);

        dd::Edge e = dd->makeIdent(0, short(nqubits-1));
        dd->incRef(e);
        for(const auto& cycle:cycles) {
            dd::Edge f = dd->makeIdent(0, short(nqubits-1));
            for(const auto& op: cycle)
                f = dd->multiply(op->getDD(dd, line, map), f);
            dd::Edge g = dd->multiply(f, e);
            dd->decRef(e);
            dd->incRef(g);
            e = g;
            dd->garbageCollect();
        }
        return e;
    }

    dd::Edge GoogleRandomCircuitSampling::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		std::array<short, MAX_QUBITS> line{};
        line.fill(LINE_DEFAULT);
	    permutationMap map = initialLayout;
	    dd->setMode(dd::Vector);

	    dd::Edge e = in;
        dd->incRef(e);

        for (const auto& cycle: cycles) {
            for (const auto& op: cycle) {
                auto tmp = dd->multiply(op->getDD(dd, line, map), e);
                dd->incRef(tmp);
                dd->decRef(e);
                e = tmp;

                dd->garbageCollect();
            }
        }
        return e;
    }
}
