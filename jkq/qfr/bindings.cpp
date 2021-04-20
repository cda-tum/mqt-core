/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QiskitImport.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "dd/Export.hpp"
#include "nlohmann/json.hpp"
#include "pybind11/pybind11.h"
#include "pybind11_json/pybind11_json.hpp"

#include <chrono>

namespace py = pybind11;
namespace nl = nlohmann;
using namespace pybind11::literals;

enum class ConstructionMethod {
    Sequential,
    Recursive
};

NLOHMANN_JSON_SERIALIZE_ENUM(ConstructionMethod, {{ConstructionMethod::Sequential, "sequential"},
                                                  {ConstructionMethod::Recursive, "recursive"}})

void to_json(nlohmann::json& j, dd::CMat& mat) {
    j = nlohmann::json::array();
    for (const auto& row: mat) {
        j.emplace_back(nl::json::array());
        for (const auto& elem: row)
            j.back().emplace_back(std::array<dd::fp, 2>{elem.first, elem.second});
    }
}
void from_json(const nlohmann::json& j, dd::CMat& mat) {
    for (std::size_t i = 0; i < j.size(); ++i) {
        auto& row = j.at(i);
        for (std::size_t y = 0; y < row.size(); ++y) {
            auto amplitude  = row.at(y).get<std::pair<dd::fp, dd::fp>>();
            mat.at(i).at(y) = amplitude;
        }
    }
}

nl::json construct(const std::unique_ptr<qc::QuantumComputation>& qc, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    // carry out actual computation
    auto         dd                 = std::make_unique<dd::Package>(qc->getNqubits());
    auto         start_construction = std::chrono::high_resolution_clock::now();
    qc::MatrixDD e{};
    if (method == ConstructionMethod::Sequential) {
        e = qc->buildFunctionality(dd);
    } else if (method == ConstructionMethod::Recursive) {
        e = qc->buildFunctionalityRecursive(dd);
    }
    auto end_construction      = std::chrono::high_resolution_clock::now();
    auto construction_duration = std::chrono::duration<float>(end_construction - start_construction);

    // populate results
    nl::json results{};

    results["circuit"]  = {};
    auto& circuit       = results["circuit"];
    circuit["name"]     = qc->getName();
    circuit["n_qubits"] = qc->getNqubits();
    circuit["n_gates"]  = qc->getNops();

    results["statistics"]           = {};
    auto& statistics                = results["statistics"];
    statistics["construction_time"] = construction_duration.count();
    statistics["final_nodecount"]   = dd->size(e);
    statistics["max_nodecount"]     = dd->mUniqueTable.getPeakNodeCount();
    statistics["method"]            = method;

    if (storeDD) {
        auto               start_dd_dump = std::chrono::high_resolution_clock::now();
        std::ostringstream oss{};
        dd::serialize(e, oss);
        results["functionality"]["dd"]        = oss.str();
        auto end_dd_dump                      = std::chrono::high_resolution_clock::now();
        auto dd_dump_duration                 = std::chrono::duration<float>(end_dd_dump - start_dd_dump);
        results["statistics"]["dd_dump_time"] = dd_dump_duration.count();
    }

    if (storeMatrix) {
        auto start_matrix_dump                    = std::chrono::high_resolution_clock::now();
        results["functionality"]["matrix"]        = dd->getMatrix(e);
        auto end_matrix_dump                      = std::chrono::high_resolution_clock::now();
        auto matrix_dump_duration                 = std::chrono::duration<float>(end_matrix_dump - start_matrix_dump);
        results["statistics"]["matrix_dump_time"] = matrix_dump_duration.count();
    }

    return results;
}

nl::json construct_circuit(const py::object& circ, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    auto qc = std::make_unique<qc::QuantumComputation>();
    try {
        if (py::isinstance<py::str>(circ)) {
            auto&& file = circ.cast<std::string>();
            qc->import(file);
        } else {
            import(*qc, circ);
        }
    } catch (std::exception const& e) {
        std::stringstream ss{};
        ss << "Could not import circuit: " << e.what();
        return {{"error", ss.str()}};
    }
    return construct(qc, method, storeDD, storeMatrix);
}

nl::json construct_grover(dd::QubitCount nqubits, unsigned int seed = 0, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    std::unique_ptr<qc::QuantumComputation> qc     = std::make_unique<qc::Grover>(nqubits, seed);
    auto                                    grover = dynamic_cast<qc::Grover*>(qc.get());

    auto  results           = construct(qc, method, storeDD, storeMatrix);
    auto& circuit           = results["circuit"];
    circuit["name"]         = "Grover's algorithm";
    circuit["seed"]         = seed;
    circuit["n_iterations"] = grover->iterations;
    circuit["target_state"] = grover->x;
    return results;
}

nl::json construct_qft(dd::QubitCount nqubits, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    std::unique_ptr<qc::QuantumComputation> qc      = std::make_unique<qc::QFT>(nqubits);
    auto                                    results = construct(qc, method, storeDD, storeMatrix);
    auto&                                   circuit = results["circuit"];
    circuit["name"]                                 = "Quantum Fourier Transform";
    return results;
}

nl::json matrix_from_dd(const std::string& serializedDD) {
    nl::json results{};

    auto               dd                    = std::make_unique<dd::Package>();
    auto               start_deserialization = std::chrono::high_resolution_clock::now();
    std::istringstream iss{serializedDD};
    auto               e                        = dd->deserialize<dd::Package::mNode>(iss);
    auto               end_deserialization      = std::chrono::high_resolution_clock::now();
    auto               deserialization_duration = std::chrono::duration<float>(end_deserialization - start_deserialization);
    results["deserialization_time"]             = deserialization_duration.count();

    auto start_matrix_dump      = std::chrono::high_resolution_clock::now();
    results["matrix"]           = dd->getMatrix(e);
    auto end_matrix_dump        = std::chrono::high_resolution_clock::now();
    auto matrix_dump_duration   = std::chrono::duration<float>(end_matrix_dump - start_matrix_dump);
    results["matrix_dump_time"] = matrix_dump_duration.count();

    return results;
}

PYBIND11_MODULE(pyqfr, m) {
    m.doc() = "Python interface for the JKQ QFR quantum functionality representation";

    py::enum_<ConstructionMethod>(m, "ConstructionMethod")
            .value("sequential", ConstructionMethod::Sequential)
            .value("recursive", ConstructionMethod::Recursive)
            .export_values();

    m.def("construct", &construct_circuit, "construct a functional representation of a quantum circuit",
          "circ"_a,
          "method"_a       = ConstructionMethod::Recursive,
          "store_dd"_a     = false,
          "store_matrix"_a = false);
    m.def("construct_grover", &construct_grover, "construct a functional representation for Grover's algorithm",
          "nqubits"_a      = 2,
          "seed"_a         = 0,
          "method"_a       = ConstructionMethod::Recursive,
          "store_dd"_a     = false,
          "store_matrix"_a = false);
    m.def("construct_qft", &construct_qft, "construct a functional representation for the QFT",
          "nqubits"_a      = 2,
          "method"_a       = ConstructionMethod::Recursive,
          "store_dd"_a     = false,
          "store_matrix"_a = false);
    m.def("matrix_from_dd", &matrix_from_dd, "construct matrix from serialized decision diagram",
          "serialized_dd"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
