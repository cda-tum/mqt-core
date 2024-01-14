#pragma once

#include "Permutation.hpp"
#include "QuantumComputation.hpp"
#include "mqt_core_export.h"
#include "operations/Expression.hpp"
#include "operations/Operation.hpp"
#include "zx/ZXDefinitions.hpp"
#include "zx/ZXDiagram.hpp"

#include <cstddef>
#include <optional>

namespace zx {
class MQT_CORE_EXPORT FunctionalityConstruction {
  using op_it = qc::QuantumComputation::const_iterator;

public:
  static ZXDiagram buildFunctionality(const qc::QuantumComputation* qc);

  static bool transformableToZX(const qc::QuantumComputation* qc);

  static bool transformableToZX(const qc::Operation* op);

protected:
  static bool checkSwap(const op_it& it, const op_it& end, Qubit ctrl,
                        Qubit target, const qc::Permutation& p);
  static void addZSpider(ZXDiagram& diag, zx::Qubit qubit,
                         std::vector<Vertex>& qubits,
                         const PiExpression& phase = PiExpression(),
                         EdgeType type = EdgeType::Simple);
  static void addXSpider(ZXDiagram& diag, Qubit qubit,
                         std::vector<Vertex>& qubits,
                         const PiExpression& phase = PiExpression(),
                         EdgeType type = EdgeType::Simple);
  static void
  addRz(ZXDiagram& diag, const PiExpression& phase, Qubit target,
        std::vector<Vertex>& qubits,
        const std::optional<double>& unconvertedPhase = std::nullopt);
  static void addRx(ZXDiagram& diag, const PiExpression& phase, Qubit target,
                    std::vector<Vertex>& qubits);
  static void
  addRy(ZXDiagram& diag, const PiExpression& phase, Qubit target,
        std::vector<Vertex>& qubits,
        const std::optional<double>& unconvertedPhase = std::nullopt);
  static void addCnot(ZXDiagram& diag, Qubit ctrl, Qubit target,
                      std::vector<Vertex>& qubits,
                      EdgeType type = EdgeType::Simple);
  static void addCphase(ZXDiagram& diag, const PiExpression& phase, Qubit ctrl,
                        Qubit target, std::vector<Vertex>& qubits);
  static void addSwap(ZXDiagram& diag, Qubit target, Qubit target2,
                      std::vector<Vertex>& qubits);
  static void
  addRzz(ZXDiagram& diag, const PiExpression& phase, Qubit target,
         Qubit target2, std::vector<Vertex>& qubits,
         const std::optional<double>& unconvertedPhase = std::nullopt);
  static void
  addRxx(ZXDiagram& diag, const PiExpression& phase, Qubit target,
         Qubit target2, std::vector<Vertex>& qubits,
         const std::optional<double>& unconvertedPhase = std::nullopt);
  static void
  addRzx(ZXDiagram& diag, const PiExpression& phase, Qubit target,
         Qubit target2, std::vector<Vertex>& qubits,
         const std::optional<double>& unconvertedPhase = std::nullopt);
  static void addDcx(ZXDiagram& diag, Qubit qubit1, Qubit qubit2,
                     std::vector<Vertex>& qubits);
  static void
  addXXplusYY(ZXDiagram& diag, const PiExpression& theta,
              const PiExpression& beta, Qubit qubit0, Qubit qubit1,
              std::vector<Vertex>& qubits,
              const std::optional<double>& unconvertedBeta = std::nullopt);
  static void
  addXXminusYY(ZXDiagram& diag, const PiExpression& theta,
               const PiExpression& beta, Qubit qubit0, Qubit qubit1,
               std::vector<Vertex>& qubits,
               const std::optional<double>& unconvertedBeta = std::nullopt);
  static void addCcx(ZXDiagram& diag, Qubit ctrl0, Qubit ctrl1, Qubit target,
                     std::vector<Vertex>& qubits);
  static op_it parseOp(ZXDiagram& diag, op_it it, op_it end,
                       std::vector<Vertex>& qubits, const qc::Permutation& p);
  static op_it parseCompoundOp(ZXDiagram& diag, op_it it, op_it end,
                               std::vector<Vertex>& qubits,
                               const qc::Permutation& initialLayout);

  static PiExpression toPiExpr(const qc::SymbolOrNumber& param);
  static PiExpression parseParam(const qc::Operation* op, std::size_t i);
};
} // namespace zx
