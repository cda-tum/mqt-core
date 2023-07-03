#include "operations/OpType.hpp"
#include <iostream>
#include <ostream>
#include <python/nanobind.hpp>
#include <nanobind/stl/string.h>
#include <string>
#include "QuantumComputation.hpp"
#include "python/qiskit/nanobind/QuantumCircuit.hpp"


namespace mqt {
  // void register_qiskit(nb::module_);
 
void loadQC(qc::QuantumComputation& qc, const nb::object& circ) {
  try {
    if (nb::isinstance<nb::str>(circ)) {
      auto&& file = nb::cast<std::string>(circ);
      qc.import(file);
    } else {
      qc::qiskit::QuantumCircuit::import(qc, circ);
    }
  } catch (std::exception const& e) {
    std::stringstream ss{};
    ss << "Could not import circuit: " << e.what();
    throw std::invalid_argument(ss.str());
  }
}
  
  enum class foo {
    a
  };

  foo fooFromStr(const std::string& s) {
    return foo::a;
  }
NB_MODULE(_core, m) {
  m.def("load_qc", &loadQC);
  
  nb::enum_<foo>(m, "Foo")
      .value("a", foo::a)
      .export_values()
    .def("__init__", [](foo* t, const std::string& s) { new (t) foo(fooFromStr(s)); });

  nb::enum_<qc::OpType>(m, "OpType")
      .value("none", qc::OpType::None)
      .value("gphase", qc::OpType::GPhase)
      .value("i", qc::OpType::I)
      .value("h", qc::OpType::H)
      .value("x", qc::OpType::X)
      .value("y", qc::OpType::Y)
      .value("z", qc::OpType::Z)
      .value("s", qc::OpType::S)
      .value("sdag", qc::OpType::Sdag)
      .value("t", qc::OpType::T)
      .value("tdag", qc::OpType::Tdag)
      .value("v", qc::OpType::V)
      .value("vdag", qc::OpType::Vdag)
      .value("u3", qc::OpType::U3)
      .value("u2", qc::OpType::U2)
      .value("phase", qc::OpType::Phase)
      .value("sx", qc::OpType::SX)
      .value("sxdag", qc::OpType::SXdag)
      .value("rx", qc::OpType::RX)
      .value("ry", qc::OpType::RY)
      .value("rz", qc::OpType::RZ)
      .value("swap", qc::OpType::SWAP)
      .value("iswap", qc::OpType::iSWAP)
      .value("peres", qc::OpType::Peres)
      .value("peresdag", qc::OpType::Peresdag)
      .value("dcx", qc::OpType::DCX)
      .value("ecr", qc::OpType::ECR)
      .value("rxx", qc::OpType::RXX)
      .value("ryy", qc::OpType::RYY)
      .value("rzz", qc::OpType::RZZ)
      .value("rzx", qc::OpType::RZX)
      .value("xx_minus_yy", qc::OpType::XXminusYY)
      .value("xx_plus_yy", qc::OpType::XXplusYY)
      .value("compound", qc::OpType::Compound)
      .value("measure", qc::OpType::Measure)
      .value("reset", qc::OpType::Reset)
      .value("snapshot", qc::OpType::Snapshot)
      .value("showprobabilities", qc::OpType::ShowProbabilities)
      .value("barrier", qc::OpType::Barrier)
      .value("teleportation", qc::OpType::Teleportation)
      .value("classiccontrolled", qc::OpType::ClassicControlled)
      .export_values()
    .def_static("from_string",
                [](const std::string& s) { return qc::opTypeFromString(s); });

}

} // namespace mqt
