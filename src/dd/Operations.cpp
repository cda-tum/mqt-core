#include "dd/Operations.hpp"

namespace dd {
template <class Config>
void dumpTensor(qc::Operation* op, std::ostream& of,
                std::vector<std::size_t>& inds, size_t& gateIdx,
                std::unique_ptr<dd::Package<Config>>& dd) {
  const auto type = op->getType();
  if (op->isStandardOperation()) {
    auto nqubits = op->getNqubits();
    const auto& controls = op->getControls();
    const auto& targets = op->getTargets();

    // start of tensor
    of << "[";

    // save tags including operation type, involved qubits, and gate index
    of << "[\"" << op->getName() << "\", ";

    // obtain an ordered map of involved qubits and add corresponding tags
    std::map<qc::Qubit, std::variant<qc::Qubit, qc::Control>> orderedQubits{};
    for (const auto& control : controls) {
      orderedQubits.emplace(control.qubit, control);
      of << "\"Q" << control.qubit << "\", ";
    }
    for (const auto& target : targets) {
      orderedQubits.emplace(target, target);
      of << "\"Q" << target << "\", ";
    }
    of << "\"GATE" << gateIdx << "\"], ";
    ++gateIdx;

    // generate indices
    // in order to conform to the DD variable ordering that later provides the
    // tensor data the ordered map has to be traversed in reverse order in order
    // to correctly determine the indices
    std::stringstream ssIn{};
    std::stringstream ssOut{};
    auto iter = orderedQubits.rbegin();
    auto qubit = iter->first;
    auto& idx = inds[qubit];
    ssIn << "\"q" << qubit << "_" << idx << "\"";
    ++idx;
    ssOut << "\"q" << qubit << "_" << idx << "\"";
    ++iter;
    while (iter != orderedQubits.rend()) {
      qubit = iter->first;
      auto& ind = inds[qubit];
      ssIn << ", \"q" << qubit << "_" << ind << "\"";
      ++ind;
      ssOut << ", \"q" << qubit << "_" << ind << "\"";
      ++iter;
    }
    of << "[" << ssIn.str() << ", " << ssOut.str() << "], ";

    // write tensor dimensions
    const std::size_t localQubits = targets.size() + controls.size();
    const std::size_t globalQubits = nqubits;
    of << "[";
    for (std::size_t q = 0U; q < localQubits; ++q) {
      if (q != 0U) {
        of << ", ";
      }
      of << 2 << ", " << 2;
    }
    of << "], ";

    // obtain a local representation of the underlying operation
    qc::Qubit localIdx = 0;
    qc::Controls localControls{};
    qc::Targets localTargets{};
    for (const auto& [q, var] : orderedQubits) {
      if (std::holds_alternative<qc::Qubit>(var)) {
        localTargets.emplace_back(localIdx);
      } else {
        const auto* control = std::get_if<qc::Control>(&var);
        localControls.emplace(qc::Control{localIdx, control->type});
      }
      ++localIdx;
    }
    // temporarily change nqubits
    op->setNqubits(localQubits);

    // get DD for local operation
    auto localOp = op->clone();
    localOp->setControls(localControls);
    localOp->setTargets(localTargets);
    const auto localDD = getDD(localOp.get(), dd);

    // translate local DD to matrix
    const auto localMatrix = localDD.getMatrix();

    // restore nqubits
    op->setNqubits(globalQubits);

    // set appropriate precision for dumping numbers
    const auto precision = of.precision();
    of.precision(std::numeric_limits<dd::fp>::max_digits10);

    // write tensor data
    of << "[";
    for (std::size_t row = 0U; row < localMatrix.size(); ++row) {
      const auto& r = localMatrix[row];
      for (std::size_t col = 0U; col < r.size(); ++col) {
        if (row != 0U || col != 0U) {
          of << ", ";
        }

        const auto& elem = r[col];
        of << "[" << elem.real() << ", " << elem.imag() << "]";
      }
    }
    of << "]";

    // restore old precision
    of.precision(precision);

    // end of tensor
    of << "]";
  } else if (auto* compoundOp = dynamic_cast<qc::CompoundOperation*>(op)) {
    for (const auto& operation : *compoundOp) {
      if (operation != (*compoundOp->begin())) {
        of << ",\n";
      }
      dumpTensor(operation.get(), of, inds, gateIdx, dd);
    }
  } else if (type == qc::Barrier) {
    return;
  } else if (type == qc::Measure) {
    std::clog << "Skipping measurement in tensor dump.\n";
  } else {
    throw qc::QFRException("Dumping of tensors is currently only supported for "
                           "StandardOperations.");
  }
}

template void
dumpTensor<DDPackageConfig>(qc::Operation* op, std::ostream& of,
                            std::vector<std::size_t>& inds, size_t& gateIdx,
                            std::unique_ptr<dd::Package<DDPackageConfig>>& dd);
} // namespace dd
