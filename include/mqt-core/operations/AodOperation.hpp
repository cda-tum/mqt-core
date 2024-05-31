#pragma once

#include "Control.hpp"
#include "Definitions.hpp"
#include "OpType.hpp"
#include "operations/Operation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace na {

enum class Dimension : std::uint8_t { X = 0, Y = 1 };
struct SingleOperation {
  Dimension dir;
  qc::fp start;
  qc::fp end;

  SingleOperation(const Dimension d, const qc::fp s, const qc::fp e)
      : dir(d), start(s), end(e) {}

  [[nodiscard]] std::string toQASMString() const {
    std::stringstream ss;
    ss << static_cast<char32_t>(dir) << ", " << start << ", " << end << "; ";
    return ss.str();
  }
};
class AodOperation : public qc::Operation {
  std::vector<SingleOperation> operations;

  static std::vector<Dimension>
  convertToDimension(const std::vector<uint32_t>& dirs);

public:
  AodOperation() = default;
  AodOperation(qc::OpType s, std::vector<qc::Qubit> qubits,
               const std::vector<Dimension>& dirs,
               const std::vector<qc::fp>& starts,
               const std::vector<qc::fp>& ends);
  AodOperation(qc::OpType s, std::vector<qc::Qubit> qubits,
               const std::vector<uint32_t>& dirs,
               const std::vector<qc::fp>& starts,
               const std::vector<qc::fp>& ends);
  AodOperation(const std::string& typeName, std::vector<qc::Qubit> qubits,
               const std::vector<uint32_t>& dirs,
               const std::vector<qc::fp>& starts,
               const std::vector<qc::fp>& ends);
  AodOperation(qc::OpType s, std::vector<qc::Qubit> qubits,
               const std::vector<std::tuple<Dimension, qc::fp, qc::fp>>& ops);
  AodOperation(qc::OpType type, std::vector<qc::Qubit> targets,
               std::vector<SingleOperation> operations);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<AodOperation>(*this);
  }

  void addControl([[maybe_unused]] qc::Control c) override {}
  void clearControls() override {}
  void removeControl([[maybe_unused]] qc::Control c) override {}
  qc::Controls::iterator removeControl(qc::Controls::iterator it) override {
    return it;
  }

  [[nodiscard]] std::vector<qc::fp> getEnds(Dimension dir) const {
    std::vector<qc::fp> ends;
    for (const auto& op : operations) {
      if (op.dir == dir) {
        ends.push_back(op.end);
      }
    }
    return ends;
  }

  [[nodiscard]] std::vector<qc::fp> getStarts(Dimension dir) const {
    std::vector<qc::fp> starts;
    for (const auto& op : operations) {
      if (op.dir == dir) {
        starts.push_back(op.start);
      }
    }
    return starts;
  }

  [[nodiscard]] qc::fp getMaxDistance(Dimension dir) const {
    const auto distances = getDistances(dir);
    if (distances.empty()) {
      return 0;
    }
    return *std::max_element(distances.begin(), distances.end());
  }

  [[nodiscard]] std::vector<qc::fp> getDistances(Dimension dir) const {
    std::vector<qc::fp> params;
    for (const auto& op : operations) {
      if (op.dir == dir) {
        params.push_back(std::abs(op.end - op.start));
      }
    }
    return params;
  }

  void dumpOpenQASM(std::ostream& of, const qc::RegisterNames& qreg,
                    const qc::RegisterNames& creg, size_t indent,
                    bool openQASM3) const override;

  void invert() override;
};
} // namespace na
