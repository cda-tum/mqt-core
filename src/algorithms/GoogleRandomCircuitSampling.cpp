/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/GoogleRandomCircuitSampling.hpp"

#include <utility>

namespace qc {
    GoogleRandomCircuitSampling::GoogleRandomCircuitSampling(const std::string& filename) {
        importGRCS(filename);
    }

    GoogleRandomCircuitSampling::GoogleRandomCircuitSampling(std::string prefix, const std::uint16_t device, const std::uint16_t depth, const std::uint16_t instance):
        layout(Bristlecone), pathPrefix(std::move(prefix)) {
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

        importGRCS(ss.str());
    }

    GoogleRandomCircuitSampling::GoogleRandomCircuitSampling(std::string prefix, const std::uint16_t x, const std::uint16_t y, const std::uint16_t depth, const std::uint16_t instance):
        pathPrefix(std::move(prefix)) {
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
        importGRCS(ss.str());
    }

    void GoogleRandomCircuitSampling::importGRCS(const std::string& filename) {
        auto ifs = std::ifstream(filename);
        if (!ifs.good()) {
            std::cerr << "Error opening/reading from file: " << filename << std::endl;
            exit(3);
        }
        const std::size_t slash     = filename.find_last_of('/');
        const std::size_t dot       = filename.find_last_of('.');
        std::string       benchmark = filename.substr(slash + 1, dot - slash - 1);
        name                        = benchmark;
        layout                      = (benchmark[0] == 'b') ? Bristlecone : Rectangular;
        std::size_t nq{};
        ifs >> nq;

        addQubitRegister(nq);
        addClassicalRegister(nq);

        std::string line;
        std::string identifier;
        std::size_t control = 0;
        std::size_t target  = 0;
        std::size_t cycle   = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) {
                continue;
            }
            std::stringstream ss(line);
            ss >> cycle;
            if (cycles.size() <= cycle) {
                cycles.emplace_back();
            }

            ss >> identifier;
            if (identifier == "cz") {
                ss >> control;
                ss >> target;
                cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, Control{static_cast<Qubit>(control)}, target, Z));
            } else if (identifier == "is") {
                ss >> control;
                ss >> target;
                cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, qc::Controls{}, control, target, iSWAP));
            } else {
                ss >> target;
                if (identifier == "h") {
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, H));
                } else if (identifier == "t") {
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, T));
                } else if (identifier == "x_1_2") {
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, RX, PI_2));
                } else if (identifier == "y_1_2") {
                    cycles[cycle].emplace_back(std::make_unique<StandardOperation>(nqubits, target, RY, PI_2));
                } else {
                    throw QFRException("Unknown gate '" + identifier);
                }
            }
        }
    }

    std::size_t GoogleRandomCircuitSampling::getNops() const {
        std::size_t nops = 0;
        for (const auto& cycle: cycles) {
            nops += cycle.size();
        }
        return nops;
    }

    std::ostream& GoogleRandomCircuitSampling::print(std::ostream& os) const {
        std::size_t i = 0;
        std::size_t j = 0;
        for (const auto& cycle: cycles) {
            os << "Cycle " << i++ << ":\n";
            for (const auto& op: cycle) {
                os << std::setw(static_cast<int>(std::log10(getNops()) + 1.)) << ++j << ": ";
                op->print(os, initialLayout);
                os << std::endl;
            }
        }
        return os;
    }

    std::ostream& GoogleRandomCircuitSampling::printStatistics(std::ostream& os) const {
        os << "GoogleRandomCircuitSampling Statistics:\n";
        os << "\tLayout: " << ((layout == Rectangular) ? "Rectangular" : "Bristlecone") << std::endl;
        os << "\tn: " << static_cast<std::size_t>(nqubits) << std::endl;
        os << "\tm: " << getNops() << std::endl;
        os << "\tc: 1 + " << cycles.size() - 2 << " + 1" << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
} // namespace qc
