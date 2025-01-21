#include "ir/QuantumComputation.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <set>

namespace mqt::ir::opt {
/// Analysis pattern that creates MLIR instructions from a given
/// qc::QuantumComputation. These instructions replace an existing `AllocOp`
/// that has the `to_replace` attribute set.
struct FromQuantumComputationPattern final : mlir::OpRewritePattern<AllocOp> {
  qc::QuantumComputation& circuit;

  explicit FromQuantumComputationPattern(mlir::MLIRContext* context,
                                         qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

  mlir::LogicalResult match(AllocOp op) const override {
    return op->hasAttr("to_replace") ? mlir::success() : mlir::failure();
  }

  /**
   * @brief Creates an ExtractOp that extracts the qubit at the given index from
   * the given register.
   *
   * @param reg The register to extract the qubit from.
   * @param index The index of the qubit to extract.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created ExtractOp.
   */
  ExtractOp createRegisterAccess(AllocOp& reg, size_t index,
                                 mlir::PatternRewriter& rewriter) const {
    return rewriter.create<ExtractOp>(
        reg.getLoc(),
        mlir::TypeRange{QubitRegisterType::get(rewriter.getContext()),
                        QubitType::get(rewriter.getContext())},
        reg.getResult(), nullptr, rewriter.getI64IntegerAttr(index));
  }

  /**
   * @brief Creates a unitary operation on a given qubit with any number of
   * positive or negative controls.
   *
   * @param loc The location of the operation.
   * @param inQubit The qubit to apply the unitary operation to.
   * @param controlQubitsPositive The positive control qubits.
   * @param controlQubitsNegative The negative control qubits.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created UnitaryOp.
   */
  UnitaryInterface createUnitaryOp(mlir::Location loc, qc::OpType type,
                                   mlir::Value inQubit,
                                   mlir::ValueRange controlQubitsPositive,
                                   mlir::ValueRange controlQubitsNegative,
                                   mlir::PatternRewriter& rewriter) const {
    std::vector<mlir::Type> resultTypes(1 + controlQubitsPositive.size() +
                                            controlQubitsNegative.size(),
                                        inQubit.getType());
    mlir::TypeRange outTypes(resultTypes);

    switch (type) {
    case qc::OpType::X:
      return rewriter.create<XOp>(loc, outTypes, mlir::ValueRange{},
                                  mlir::ValueRange{inQubit},
                                  controlQubitsPositive, controlQubitsNegative);
    default:
      throw std::runtime_error("Unsupported operation type");
    }
  }

  /**
   * @brief Creates a MeasureOp on a given qubit.
   *
   * @param loc The location of the operation.
   * @param targetQubit The qubit to measure.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created MeasureOp.
   */
  MeasureOp createMeasureOp(mlir::Location loc, mlir::Value targetQubit,
                            mlir::PatternRewriter& rewriter) const {
    return rewriter.create<MeasureOp>(
        loc,
        mlir::TypeRange{QubitType::get(rewriter.getContext()),
                        rewriter.getI1Type()},
        targetQubit);
  }

  /**
   * @brief Updates the inputs of the function's `return` operation.
   *
   * The previous operation used constant `false` values for the returns of type
   * `i1`. After this update, the measurement results are used instead.
   *
   * @param returnOperation The `return` operation to update.
   * @param measurementValues The values to use as the new return values.
   * @param rewriter The pattern rewriter to use.
   */
  void updateReturnOperation(mlir::Operation* returnOperation,
                             std::vector<mlir::Value>& measurementValues,
                             mlir::PatternRewriter& rewriter) const {
    const auto cloned = rewriter.clone(*returnOperation);
    for (size_t i = 0; i < measurementValues.size(); i++) {
      cloned->setOperand(i + 1, measurementValues[i]);
    }
    rewriter.replaceOp(returnOperation, cloned);
  }

  void rewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    std::size_t numQubits = circuit.getNqubits();

    // Prepare list of measurement results for later use.
    std::vector<mlir::Value> measurementValues(numQubits);

    // Create a new qubit register with the correct number of qubits.
    auto newAlloc = rewriter.create<AllocOp>(
        op.getLoc(), QubitRegisterType::get(rewriter.getContext()), nullptr,
        rewriter.getIntegerAttr(rewriter.getI64Type(), numQubits));
    newAlloc->setAttr("mqt_core", rewriter.getUnitAttr());

    // We start by first extracting each qubit from the register. The current
    // `Value` representations of each qubit are stored in the
    // `currentQubitVariables` vector.
    std::vector<mlir::Value> currentQubitVariables(numQubits);
    for (size_t i = 0; i < numQubits; i++) {
      currentQubitVariables[i] =
          createRegisterAccess(newAlloc, i, rewriter).getOutQubit();
    }

    // Iterate over each operation in the circuit and create the corresponding
    // MLIR operations.
    for (const auto& o : circuit) {
      // Collect the positive and negative control qubits for the operation in
      // separate vectors.
      std::vector<int> controlQubitIndicesPositive;
      std::vector<int> controlQubitIndicesNegative;
      std::vector<mlir::Value> controlQubitsPositive;
      std::vector<mlir::Value> controlQubitsNegative;
      for (const auto& control : o->getControls()) {
        if (control.type == qc::Control::Type::Pos) {
          controlQubitIndicesPositive.push_back(control.qubit);
          controlQubitsPositive.push_back(currentQubitVariables[control.qubit]);
        } else {
          controlQubitIndicesNegative.push_back(control.qubit);
          controlQubitsNegative.push_back(currentQubitVariables[control.qubit]);
        }
      }

      if (o->getType() == qc::OpType::X) {
        // For unitary operations, we call the `createUnitaryOp` function. We
        // then have to update the `currentQubitVariables` vector with the new
        // qubit values.
        UnitaryInterface newUnitaryOp = createUnitaryOp(
            op->getLoc(), o->getType(),
            currentQubitVariables[o->getTargets()[0]], controlQubitsPositive,
            controlQubitsNegative, rewriter);
        currentQubitVariables[o->getTargets()[0]] =
            newUnitaryOp.getOutQubits()[0];
        for (size_t i = 0; i < controlQubitsPositive.size(); i++) {
          currentQubitVariables[controlQubitIndicesPositive[i]] =
              newUnitaryOp.getOutQubits()[i + 1];
        }
        for (size_t i = 0; i < controlQubitsNegative.size(); i++) {
          currentQubitVariables[controlQubitIndicesNegative[i]] =
              newUnitaryOp.getOutQubits()[i + 1 + controlQubitsPositive.size()];
        }
      } else if (o->getType() == qc::OpType::Measure) {
        // For measurement operations, we call the `createMeasureOp` function.
        // We then update the `currentQubitVariables` and `measurementValues`
        // vectors.
        MeasureOp newMeasureOp = createMeasureOp(
            op->getLoc(), currentQubitVariables[o->getTargets()[0]], rewriter);
        currentQubitVariables[o->getTargets()[0]] =
            newMeasureOp.getOutQubits()[0];
        measurementValues[o->getTargets()[0]] = newMeasureOp.getOutBits()[0];
      } else {
        llvm::outs() << "ERROR: Unsupported operation type " << o->getType()
                     << "\n";
      }
    }

    // Finally, the return operation needs to be updated with the measurement
    // results and then replace the original `alloc` operation with the updated
    // one.
    auto returnOperation = *op->getUsers().begin();
    updateReturnOperation(returnOperation, measurementValues, rewriter);
    rewriter.replaceOp(op, newAlloc);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `FromQuantumComputationPattern`.
 *
 * @param patterns The pattern set to populate.
 * @param circuit The quantum computation to create MLIR instructions from.
 */
void populateFromQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                            qc::QuantumComputation& circuit) {
  patterns.add<FromQuantumComputationPattern>(patterns.getContext(), circuit);
}

} // namespace mqt::ir::opt
