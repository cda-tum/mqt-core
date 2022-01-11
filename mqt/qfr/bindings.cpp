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

#include <eccs/Ecc.hpp>
#include <eccs/IdEcc.hpp>
#include <eccs/Q3ShorEcc.hpp>
#include <eccs/Q5LaflammeEcc.hpp>
#include <eccs/Q7SteaneEcc.hpp>
#include <eccs/Q9SurfaceEcc.hpp>
#include <eccs/Q9ShorEcc.hpp>
#include <eccs/Q18SurfaceEcc.hpp>

#include <chrono>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    auto         dd                 = std::make_unique<dd::Package<>>(qc->getNqubits());
    auto         start_construction = std::chrono::high_resolution_clock::now();
    qc::MatrixDD e{};
    if (method == ConstructionMethod::Sequential) {
        e = buildFunctionality(qc.get(), dd);
    } else if (method == ConstructionMethod::Recursive) {
        e = buildFunctionalityRecursive(qc.get(), dd);
    }
    auto end_construction      = std::chrono::high_resolution_clock::now();
    auto construction_duration = std::chrono::duration<float>(end_construction - start_construction);

    // populate results
    py::dict results{};
    auto     circuit    = py::dict{};
    circuit["name"]     = qc->getName();
    circuit["n_qubits"] = static_cast<std::size_t>(qc->getNqubits());
    circuit["n_gates"]  = qc->getNops();
    results["circuit"]  = circuit;

    auto statistics                 = py::dict{};
    statistics["construction_time"] = construction_duration.count();
    statistics["final_nodecount"]   = dd->size(e);
    statistics["max_nodecount"]     = dd->mUniqueTable.getPeakNodeCount();
    statistics["method"]            = toString(method);
    results["statistics"]           = statistics;

    if (storeDD || storeMatrix) {
        results["functionality"] = py::dict{};
    }

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

py::dict construct_circuit(const py::object& circ, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    auto qc = std::make_unique<qc::QuantumComputation>();
    try {
        if (py::isinstance<py::str>(circ)) {
            auto&& file = circ.cast<std::string>();
            qc->import(file);
        } else {
            py::object QuantumCircuit       = py::module::import("qiskit").attr("QuantumCircuit");
            py::object pyQasmQobjExperiment = py::module::import("qiskit.qobj").attr("QasmQobjExperiment");
            if (py::isinstance(circ, QuantumCircuit)) {
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

py::dict construct_grover(dd::QubitCount nqubits, unsigned int seed = 0, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    std::unique_ptr<qc::QuantumComputation> qc     = std::make_unique<qc::Grover>(nqubits, seed);
    auto                                    grover = dynamic_cast<qc::Grover*>(qc.get());

    auto results                       = construct(qc, method, storeDD, storeMatrix);
    results["circuit"]["name"]         = "Grover's algorithm";
    results["circuit"]["seed"]         = seed;
    results["circuit"]["n_iterations"] = grover->iterations;
    results["circuit"]["target_state"] = grover->targetValue;
    return results;
}

py::dict construct_qft(dd::QubitCount nqubits, const ConstructionMethod& method = ConstructionMethod::Recursive, bool storeDD = false, bool storeMatrix = false) {
    std::unique_ptr<qc::QuantumComputation> qc      = std::make_unique<qc::QFT>(nqubits);
    auto                                    results = construct(qc, method, storeDD, storeMatrix);
    results["circuit"]["name"]                      = "Quantum Fourier Transform";
    return results;
}

py::dict matrix_from_dd(const std::string& serializedDD) {
    py::dict results{};

    auto               dd                    = std::make_unique<dd::Package<>>();
    auto               start_deserialization = std::chrono::high_resolution_clock::now();
    std::istringstream iss{serializedDD};
    auto               e                        = dd->deserialize<dd::mNode>(iss);
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

py::dict apply_ecc(const py::object &circ, const std::string &eccString, const int ecc_frequency, const bool ecc_no_mc, const bool ecc_clifford_only) {

    qc::QuantumComputation qc{};
    std::string eccName{eccString};

    try {
        if (py::isinstance<py::str>(circ)) {
            auto&& file = circ.cast<std::string>();
            qc.import(file);
        } else {
            py::object QuantumCircuit       = py::module::import("qiskit").attr("QuantumCircuit");
            py::object pyQasmQobjExperiment = py::module::import("qiskit.qobj").attr("QasmQobjExperiment");
            if (py::isinstance(circ, QuantumCircuit)) {
                qc::qiskit::QuantumCircuit::import(qc, circ);
            } else if (py::isinstance(circ, pyQasmQobjExperiment)) {
                qc::qiskit::QasmQobjExperiment::import(qc, circ);
            }
        }
    } catch (std::exception const& e) {
        std::stringstream ss{};
        ss << "Could not import circuit: " << e.what();
        return py::dict("error"_a = ss.str());
    }


    Ecc *mapper = nullptr;
    bool decomposeMC = ecc_no_mc;
    bool cliffOnly = ecc_clifford_only;
    int measureFrequency = ecc_frequency;

    if (eccName.compare(IdEcc::getName()) == 0) {
        mapper = new IdEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    } else if (eccName.compare(Q3ShorEcc::getName()) == 0) {
        mapper = new Q3ShorEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    } else if (eccName.compare(Q5LaflammeEcc::getName()) == 0) {
        mapper = new Q5LaflammeEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    } else if (eccName.compare(Q7SteaneEcc::getName()) == 0) {
        mapper = new Q7SteaneEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    } else if (eccName.compare(Q9ShorEcc::getName()) == 0) {
        mapper = new Q9ShorEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    } else if (eccName.compare(Q9SurfaceEcc::getName()) == 0) {
        mapper = new Q9SurfaceEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    } else if (eccName.compare(Q18SurfaceEcc::getName()) == 0) {
        mapper = new Q18SurfaceEcc(qc, measureFrequency, decomposeMC, cliffOnly);
    }else {
        std::stringstream ss{};
        ss << "No ECC found for " << eccName << " ";
        ss << "Available ECCs: ";
        ss << IdEcc::getName() << ", ";
        ss << Q3ShorEcc::getName() << ", ";
        ss << Q7SteaneEcc::getName() << ", ";
        ss << Q9ShorEcc::getName() << ", ";
        ss << Q9SurfaceEcc::getName() << ", ";
        ss << Q18SurfaceEcc::getName();
        return py::dict("error"_a = ss.str());
    }
    mapper->apply();

    std::ostringstream oss{};
    mapper->dumpResult(oss, qc::OpenQASM);

    return py::dict("circ"_a = oss.str());
}

PYBIND11_MODULE(pyqfr, m) {
    m.doc() = "Python interface for the MQT QFR quantum functionality representation";

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
    m.def("apply_ecc", &apply_ecc, "applying an ecc to a circuit an returning a openQasm dump",
          "circ"_a,
          "eccString"_a,
          "ecc_frequency"_a = 100,
          "ecc_no_mc"_a = false,
          "ecc_clifford_only"_a = false);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
