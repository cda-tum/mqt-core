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

namespace qc {

enum class Dimension : std::uint8_t { X = 0, Y = 1 };
struct SingleOperation {
  Dimension dir;
  fp start;
  fp end;

  SingleOperation(Dimension dir, fp start, fp end)
      : dir(dir), start(start), end(end) {}

  [[nodiscard]] std::string toQASMString() const {
    std::stringstream ss;
    ss << static_cast<char32_t>(dir) << ", " << start << ", " << end << "; ";
    return ss.str();
  }
};
class AodOperation : public Operation {
  std::vector<SingleOperation> operations;

  static std::vector<Dimension>
  convertToDimension(const std::vector<uint32_t>& dirs);

public:
  AodOperation() = default;
  AodOperation(OpType type, std::vector<Qubit> targets,
               const std::vector<Dimension>& dirs,
               const std::vector<fp>& starts, const std::vector<fp>& ends);
  AodOperation(OpType type, std::vector<Qubit> targets,
               const std::vector<uint32_t>& dirs, const std::vector<fp>& starts,
               const std::vector<fp>& ends);
  AodOperation(const std::string& type, std::vector<Qubit> targets,
               const std::vector<uint32_t>& dirs, const std::vector<fp>& starts,
               const std::vector<fp>& ends);
  AodOperation(OpType type, std::vector<Qubit> targets,
               const std::vector<std::tuple<Dimension, fp, fp>>& operations);
  AodOperation(OpType type, std::vector<Qubit> targets,
               std::vector<SingleOperation> operations);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<AodOperation>(*this);
  }

  void addControl([[maybe_unused]] Control c) override {}
  void clearControls() override {}
  void removeControl([[maybe_unused]] Control c) override {}
  Controls::iterator removeControl(Controls::iterator it) override {
    return it;
  }

  [[nodiscard]] std::vector<fp> getEnds(Dimension dir) const {
    std::vector<fp> ends;
    for (const auto& op : operations) {
      if (op.dir == dir) {
        ends.push_back(op.end);
      }
    }
    return ends;
  }

  [[nodiscard]] std::vector<fp> getStarts(Dimension dir) const {
    std::vector<fp> starts;
    for (const auto& op : operations) {
      if (op.dir == dir) {
        starts.push_back(op.start);
      }
    }
    return starts;
  }

  [[nodiscard]] fp getMaxDistance(Dimension dir) const {
    const auto distances = getDistances(dir);
    if (distances.empty()) {
      return 0;
    }
    return *std::max_element(distances.begin(), distances.end());
  }

  [[nodiscard]] std::vector<fp> getDistances(Dimension dir) const {
    std::vector<fp> params;
    for (const auto& op : operations) {
      if (op.dir == dir) {
        params.push_back(std::abs(op.end - op.start));
      }
    }
    return params;
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, size_t indent,
                    bool openQASM3) const override;

  void invert() override;
};
} // namespace qc
