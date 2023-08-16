/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#include "operations/NonUnitaryOperation.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace qc {
// Measurement constructor
NonUnitaryOperation::NonUnitaryOperation(const std::size_t nq,
                                         std::vector<Qubit> qubitRegister,
                                         std::vector<Bit> classicalRegister)
    : classics(std::move(classicalRegister)) {
  type = Measure;
  nqubits = nq;
  targets = std::move(qubitRegister);
  Operation::setName();
  if (targets.size() != classics.size()) {
    throw std::invalid_argument(
        "Sizes of qubit register and classical register do not match.");
  }
}
NonUnitaryOperation::NonUnitaryOperation(const std::size_t nq,
                                         const Qubit qubit, const Bit cbit)
    : classics({cbit}) {
  type = Measure;
  nqubits = nq;
  targets = {qubit};
  Operation::setName();
}

// General constructor
NonUnitaryOperation::NonUnitaryOperation(const std::size_t nq, Targets qubits,
                                         OpType op) {
  type = op;
  nqubits = nq;
  targets = std::move(qubits);
  std::sort(targets.begin(), targets.end());
  Operation::setName();
}

std::ostream& NonUnitaryOperation::print(std::ostream& os,
                                         const Permutation& permutation) const {
  switch (type) {
  case Measure:
    printMeasurement(os, targets, classics, permutation);
    break;
  case Reset:
    printReset(os, targets, permutation);
    break;
  default:
    break;
  }
  return os;
}

void NonUnitaryOperation::dumpOpenQASM(std::ostream& of,
                                       const RegisterNames& qreg,
                                       const RegisterNames& creg) const {
  if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
    of << toString(type) << " " << qreg[targets.front()].first;
    if (type == Measure) {
      of << " -> ";
      assert(isWholeQubitRegister(creg, classics.front(), classics.back()));
      of << creg[classics.front()].first;
    }
    of << ";\n";
    return;
  }
  auto classicsIt = classics.cbegin();
  for (const auto& q : targets) {
    of << toString(type) << " " << qreg[q].second;
    if (type == Measure) {
      of << " -> " << creg[*classicsIt].second;
      ++classicsIt;
    }
    of << ";\n";
  }
}

bool NonUnitaryOperation::equals(const Operation& op, const Permutation& perm1,
                                 const Permutation& perm2) const {
  if (const auto* nonunitary = dynamic_cast<const NonUnitaryOperation*>(&op)) {
    if (getType() != nonunitary->getType()) {
      return false;
    }

    if (getType() == Measure) {
      // check number of qubits to be measured
      const auto nq1 = targets.size();
      const auto nq2 = nonunitary->targets.size();
      if (nq1 != nq2) {
        return false;
      }

      // these are just sanity checks and should always be fulfilled
      assert(targets.size() == classics.size());
      assert(nonunitary->targets.size() == nonunitary->classics.size());

      std::set<std::pair<Qubit, Bit>> measurements1{};
      auto qubitIt1 = targets.cbegin();
      auto classicIt1 = classics.cbegin();
      while (qubitIt1 != targets.cend()) {
        if (perm1.empty()) {
          measurements1.emplace(*qubitIt1, *classicIt1);
        } else {
          measurements1.emplace(perm1.at(*qubitIt1), *classicIt1);
        }
        ++qubitIt1;
        ++classicIt1;
      }

      std::set<std::pair<Qubit, Bit>> measurements2{};
      auto qubitIt2 = nonunitary->targets.cbegin();
      auto classicIt2 = nonunitary->classics.cbegin();
      while (qubitIt2 != nonunitary->targets.cend()) {
        if (perm2.empty()) {
          measurements2.emplace(*qubitIt2, *classicIt2);
        } else {
          measurements2.emplace(perm2.at(*qubitIt2), *classicIt2);
        }
        ++qubitIt2;
        ++classicIt2;
      }

      return measurements1 == measurements2;
    }
    return Operation::equals(op, perm1, perm2);
  }
  return false;
}

void NonUnitaryOperation::printMeasurement(
    std::ostream& os, const std::vector<Qubit>& q, const std::vector<Bit>& c,
    const Permutation& permutation) const {
  auto qubitIt = q.cbegin();
  auto classicIt = c.cbegin();
  os << name << "\t";
  if (permutation.empty()) {
    for (std::size_t i = 0; i < nqubits; ++i) {
      if (qubitIt != q.cend() && *qubitIt == i) {
        os << "\033[34m" << *classicIt << "\t"
           << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << "|\t";
      }
    }
  } else {
    for (const auto& [physical, logical] : permutation) {
      if (qubitIt != q.cend() && *qubitIt == physical) {
        os << "\033[34m" << *classicIt << "\t"
           << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << "|\t";
      }
    }
  }
}

void NonUnitaryOperation::printReset(std::ostream& os,
                                     const std::vector<Qubit>& q,
                                     const Permutation& permutation) const {
  auto qubitIt = q.cbegin();
  os << name << "\t";
  if (permutation.empty()) {
    for (std::size_t i = 0; i < nqubits; ++i) {
      if (qubitIt != q.cend() && *qubitIt == i) {
        os << "\033[31mr\t\033[0m";
        ++qubitIt;
      } else {
        os << "|\t";
      }
    }
  } else {
    for (const auto& [physical, logical] : permutation) {
      if (qubitIt != q.cend() && *qubitIt == physical) {
        os << "\033[31mr\t\033[0m";
        ++qubitIt;
      } else {
        os << "|\t";
      }
    }
  }
}
} // namespace qc
