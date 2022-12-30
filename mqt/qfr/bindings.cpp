/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "dd/Export.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "qiskit/QasmQobjExperiment.hpp"
#include "qiskit/QuantumCircuit.hpp"

#include <chrono>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

enum class ConstructionMethod {
    Sequential,
    Recursive
};

inline std::string toString(ConstructionMethod method) {
    switch (method) {
        case ConstructionMethod::Sequential:
            return "sequential";
        case ConstructionMethod::Recursive:
            return "recursive";
        default:
            return "unknown";
    }
}

py::dict construct(const std::unique_ptr<qc::QuantumComputation>& qc, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    // carry out actual computation
    auto         dd                = std::make_unique<dd::Package<>>(qc->getNqubits());
    auto         startConstruction = std::chrono::high_resolution_clock::now();
    qc::MatrixDD e{};
    if (method == ConstructionMethod::Sequential) {
        e = buildFunctionality(qc.get(), dd);
    } else if (method == ConstructionMethod::Recursive) {
        e = buildFunctionalityRecursive(qc.get(), dd);
    }
    auto endConstruction      = std::chrono::high_resolution_clock::now();
    auto constructionDuration = std::chrono::duration<float>(endConstruction - startConstruction);

    // populate results
    py::dict results{};
    auto     circuit    = py::dict{};
    circuit["name"]     = qc->getName();
    circuit["n_qubits"] = qc->getNqubits();
    circuit["n_gates"]  = qc->getNops();
    results["circuit"]  = circuit;

    auto statistics                 = py::dict{};
    statistics["construction_time"] = constructionDuration.count();
    statistics["final_nodecount"]   = dd->size(e);
    statistics["max_nodecount"]     = dd->mUniqueTable.getPeakNodeCount();
    statistics["method"]            = toString(method);
    results["statistics"]           = statistics;

    if (storeDD || storeMatrix) {
        results["functionality"] = py::dict{};
    }

    if (storeDD) {
        auto               startDdDump = std::chrono::high_resolution_clock::now();
        std::ostringstream oss{};
        dd::serialize(e, oss);
        results["functionality"]["dd"]        = oss.str();
        auto endDdDump                        = std::chrono::high_resolution_clock::now();
        auto ddDumpDuration                   = std::chrono::duration<float>(endDdDump - startDdDump);
        results["statistics"]["dd_dump_time"] = ddDumpDuration.count();
    }

    if (storeMatrix) {
        auto startMatrixDump                      = std::chrono::high_resolution_clock::now();
        results["functionality"]["matrix"]        = dd->getMatrix(e);
        auto endMatrixDump                        = std::chrono::high_resolution_clock::now();
        auto matrixDumpDuration                   = std::chrono::duration<float>(endMatrixDump - startMatrixDump);
        results["statistics"]["matrix_dump_time"] = matrixDumpDuration.count();
    }

    return results;
}

py::dict constructCircuit(const py::object& circ, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    auto qc = std::make_unique<qc::QuantumComputation>();
    try {
        if (py::isinstance<py::str>(circ)) {
            auto&& file = circ.cast<std::string>();
            qc->import(file);
        } else {
            const py::object quantumCircuit       = py::module::import("qiskit").attr("QuantumCircuit");
            const py::object pyQasmQobjExperiment = py::module::import("qiskit.qobj").attr("QasmQobjExperiment");
            if (py::isinstance(circ, quantumCircuit)) {
                qc::qiskit::QuantumCircuit::import(*qc, circ);
            } else if (py::isinstance(circ, pyQasmQobjExperiment)) {
                qc::qiskit::QasmQobjExperiment::import(*qc, circ);
            }
        }
    } catch (std::exception const& e) {
        std::stringstream ss{};
        ss << "Could not import circuit: " << e.what();
        return py::dict("error"_a = ss.str());
    }
    return construct(qc, method, storeDD, storeMatrix);
}

py::dict constructGrover(dd::QubitCount nqubits, unsigned int seed = 0, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    const std::unique_ptr<qc::QuantumComputation> qc     = std::make_unique<qc::Grover>(nqubits, seed);
    auto*                                         grover = dynamic_cast<qc::Grover*>(qc.get());

    auto results                       = construct(qc, method, storeDD, storeMatrix);
    results["circuit"]["name"]         = "Grover's algorithm";
    results["circuit"]["seed"]         = seed;
    results["circuit"]["n_iterations"] = grover->iterations;
    results["circuit"]["target_state"] = grover->targetValue;
    return results;
}

py::dict constructQFT(dd::QubitCount nqubits, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    const std::unique_ptr<qc::QuantumComputation> qc      = std::make_unique<qc::QFT>(nqubits);
    auto                                          results = construct(qc, method, storeDD, storeMatrix);
    results["circuit"]["name"]                            = "Quantum Fourier Transform";
    return results;
}

py::dict matrixFromDD(const std::string& serializedDD) {
    py::dict results{};

    auto               dd                   = std::make_unique<dd::Package<>>();
    auto               startDeserialization = std::chrono::high_resolution_clock::now();
    std::istringstream iss{serializedDD};
    auto               e                       = dd->deserialize<dd::mNode>(iss);
    auto               endDeserialization      = std::chrono::high_resolution_clock::now();
    auto               deserializationDuration = std::chrono::duration<float>(endDeserialization - startDeserialization);
    results["deserialization_time"]            = deserializationDuration.count();

    auto startMatrixDump        = std::chrono::high_resolution_clock::now();
    results["matrix"]           = dd->getMatrix(e);
    auto endMatrixDump          = std::chrono::high_resolution_clock::now();
    auto matrixDumpDuration     = std::chrono::duration<float>(endMatrixDump - startMatrixDump);
    results["matrix_dump_time"] = matrixDumpDuration.count();

    return results;
}

PYBIND11_MODULE(pyqfr, m) {
    m.doc() = "Python interface for the MQT QFR quantum functionality representation";

    py::enum_<ConstructionMethod>(m, "ConstructionMethod")
            .value("sequential", ConstructionMethod::Sequential)
            .value("recursive", ConstructionMethod::Recursive)
            .export_values();

    m.def("construct", &constructCircuit, "construct a functional representation of a quantum circuit",
          "circ"_a,
          "method"_a       = ConstructionMethod::Recursive,
          "store_dd"_a     = false,
          "store_matrix"_a = false);
    m.def("construct_grover", &constructGrover, "construct a functional representation for Grover's algorithm",
          "nqubits"_a      = 2,
          "seed"_a         = 0,
          "method"_a       = ConstructionMethod::Recursive,
          "store_dd"_a     = false,
          "store_matrix"_a = false);
    m.def("construct_qft", &constructQFT, "construct a functional representation for the QFT",
          "nqubits"_a      = 2,
          "method"_a       = ConstructionMethod::Recursive,
          "store_dd"_a     = false,
          "store_matrix"_a = false);
    m.def("matrix_from_dd", &matrixFromDD, "construct matrix from serialized decision diagram",
          "serialized_dd"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
