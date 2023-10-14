#include "QuantumComputation.hpp"

void qc::QuantumComputation::importGRCS(std::istream& is) {
  std::size_t nq{};
  is >> nq;
  addQubitRegister(nq);
  addClassicalRegister(nq);

  std::string line;
  std::string identifier;
  Qubit control = 0;
  Qubit target = 0;
  std::size_t cycle = 0;
  while (std::getline(is, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    ss >> cycle;
    ss >> identifier;
    if (identifier == "cz") {
      ss >> control;
      ss >> target;
      cz(control, target);
    } else if (identifier == "is") {
      ss >> control;
      ss >> target;
      iswap(control, target);
    } else {
      ss >> target;
      if (identifier == "h") {
        h(target);
      } else if (identifier == "t") {
        t(target);
      } else if (identifier == "x_1_2") {
        rx(qc::PI_2, target);
      } else if (identifier == "y_1_2") {
        ry(qc::PI_2, target);
      } else {
        throw QFRException("[grcs parser] unknown gate '" + identifier + "'");
      }
    }
  }
}
