/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "Permutation.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/Control.hpp"
#include "operations/Expression.hpp"
#include "operations/OpType.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qc {
class QuantumComputation {
public:
  using iterator = std::vector<std::unique_ptr<Operation>>::iterator;
  using const_iterator =
      std::vector<std::unique_ptr<Operation>>::const_iterator;
  using reverse_iterator =
      std::vector<std::unique_ptr<Operation>>::reverse_iterator;
  using const_reverse_iterator =
      std::vector<std::unique_ptr<Operation>>::const_reverse_iterator;

protected:
  std::vector<std::unique_ptr<Operation>> ops;
  std::size_t nqubits = 0;
  std::size_t nclassics = 0;
  std::size_t nancillae = 0;
  std::string name;

  // register names are used as keys, while the values are `{startIndex,
  // length}` pairs
  QuantumRegisterMap qregs;
  ClassicalRegisterMap cregs;
  QuantumRegisterMap ancregs;

  std::vector<bool> ancillary;
  std::vector<bool> garbage;

  std::mt19937_64 mt;
  std::size_t seed = 0;

  fp globalPhase = 0.;

  std::unordered_set<sym::Variable> occurringVariables;

public:
  QuantumComputation() = default;
  explicit QuantumComputation(std::size_t nq, std::size_t nc = 0U,
                              std::size_t s = 0);
  explicit QuantumComputation(const std::string& filename, std::size_t s = 0U);
  QuantumComputation(QuantumComputation&& qc) noexcept = default;
  QuantumComputation& operator=(QuantumComputation&& qc) noexcept = default;
  QuantumComputation(const QuantumComputation& qc);
  QuantumComputation& operator=(const QuantumComputation& qc);
  ~QuantumComputation() = default;

  // physical qubits are used as keys, logical qubits as values
  Permutation initialLayout{};
  Permutation outputPermutation{};

  /**
   * @brief Construct a QuantumComputation from an OpenQASM string
   * @param qasm The OpenQASM 2.0 or 3.0 string
   * @return The constructed QuantumComputation
   */
  [[nodiscard]] static QuantumComputation fromQASM(const std::string& qasm);

  /**
   * @brief Construct a QuantumComputation from CompoundOperation object
   * @details The function creates a copy of each operation in the compound
   * operation. It uses the largest qubit index in the CompoundOperation for
   * determining the number of qubits. It adds a single quantum register with
   * all qubits from 0 to the largest qubit index and a corresponding classical
   * register with the same size. The initial layout as well as the output
   * permutation are set to the identity permutation.
   * @param op The CompoundOperation to convert to a quantum circuit
   * @return The constructed QuantumComputation
   */
  [[nodiscard]] static QuantumComputation
  fromCompoundOperation(const CompoundOperation& op);

  [[nodiscard]] std::size_t getNops() const noexcept { return ops.size(); }
  [[nodiscard]] std::size_t getNqubits() const noexcept {
    return nqubits + nancillae;
  }
  [[nodiscard]] std::size_t getNancillae() const noexcept { return nancillae; }
  [[nodiscard]] std::size_t getNqubitsWithoutAncillae() const noexcept {
    return nqubits;
  }
  [[nodiscard]] const std::vector<bool>& getAncillary() const noexcept {
    return ancillary;
  }
  [[nodiscard]] const std::vector<bool>& getGarbage() const noexcept {
    return garbage;
  }
  [[nodiscard]] std::size_t getNcbits() const noexcept { return nclassics; }
  [[nodiscard]] std::string getName() const noexcept { return name; }
  [[nodiscard]] const QuantumRegisterMap& getQregs() const noexcept {
    return qregs;
  }
  [[nodiscard]] const ClassicalRegisterMap& getCregs() const noexcept {
    return cregs;
  }
  [[nodiscard]] const QuantumRegisterMap& getANCregs() const noexcept {
    return ancregs;
  }
  [[nodiscard]] decltype(mt)& getGenerator() noexcept { return mt; }

  [[nodiscard]] fp getGlobalPhase() const noexcept { return globalPhase; }

  [[nodiscard]] const std::unordered_set<sym::Variable>&
  getVariables() const noexcept {
    return occurringVariables;
  }

  [[nodiscard]] std::size_t getNmeasuredQubits() const noexcept;
  [[nodiscard]] std::size_t getNgarbageQubits() const;

  void setName(const std::string& n) noexcept { name = n; }

  [[nodiscard]] std::size_t getNindividualOps() const;
  [[nodiscard]] std::size_t getNsingleQubitOps() const;
  [[nodiscard]] std::size_t getDepth() const;

  [[nodiscard]] std::string getQubitRegister(Qubit physicalQubitIndex) const;
  [[nodiscard]] std::string getClassicalRegister(Bit classicalIndex) const;
  /// Returns the highest qubit index used as a value in the initial layout
  [[nodiscard]] Qubit getHighestLogicalQubitIndex() const;
  /// Returns the highest qubit index used as a key in the initial layout
  [[nodiscard]] Qubit getHighestPhysicalQubitIndex() const;
  [[nodiscard]] std::pair<std::string, Qubit>
  getQubitRegisterAndIndex(Qubit physicalQubitIndex) const;
  [[nodiscard]] std::pair<std::string, Bit>
  getClassicalRegisterAndIndex(Bit classicalIndex) const;
  /**
   * @brief Returns the physical qubit index of the given logical qubit index
   * @details Iterates over the initial layout dictionary and returns the key
   * corresponding to the given value.
   * @param logicalQubitIndex The logical qubit index to look for
   * @return The physical qubit index of the given logical qubit index
   */
  [[nodiscard]] Qubit getPhysicalQubitIndex(Qubit logicalQubitIndex);

  [[nodiscard]] Qubit
  getIndexFromQubitRegister(const std::pair<std::string, Qubit>& qubit) const;
  [[nodiscard]] Bit getIndexFromClassicalRegister(
      const std::pair<std::string, std::size_t>& clbit) const;
  [[nodiscard]] bool isIdleQubit(Qubit physicalQubit) const;
  [[nodiscard]] bool isLastOperationOnQubit(const const_iterator& opIt,
                                            const const_iterator& end) const;
  [[nodiscard]] bool physicalQubitIsAncillary(Qubit physicalQubitIndex) const;
  [[nodiscard]] bool
  logicalQubitIsAncillary(const Qubit logicalQubitIndex) const {
    return ancillary[logicalQubitIndex];
  }
  /**
   * @brief Sets the given logical qubit to be ancillary
   * @details Removes the qubit from the qubit register and adds it to the
   * ancillary register, if such a register exists. Otherwise a new ancillary
   * register is created.
   * @param logicalQubitIndex
   */
  void setLogicalQubitAncillary(Qubit logicalQubitIndex);
  /**
   * @brief Sets all logical qubits in the range [minLogicalQubitIndex,
   * maxLogicalQubitIndex] to be ancillary
   * @details Removes the qubits from the qubit register and adds it to the
   * ancillary register, if such a register exists. Otherwise a new ancillary
   * register is created.
   * @param minLogicalQubitIndex first qubit that is set to be ancillary
   * @param maxLogicalQubitIndex last qubit that is set to be ancillary
   */
  void setLogicalQubitsAncillary(Qubit minLogicalQubitIndex,
                                 Qubit maxLogicalQubitIndex);
  [[nodiscard]] bool
  logicalQubitIsGarbage(const Qubit logicalQubitIndex) const {
    return garbage[logicalQubitIndex];
  }
  void setLogicalQubitGarbage(Qubit logicalQubitIndex);
  /**
   * @brief Sets all logical qubits in the range [minLogicalQubitIndex,
   * maxLogicalQubitIndex] to be garbage
   * @param minLogicalQubitIndex first qubit that is set to be garbage
   * @param maxLogicalQubitIndex last qubit that is set to be garbage
   */
  void setLogicalQubitsGarbage(Qubit minLogicalQubitIndex,
                               Qubit maxLogicalQubitIndex);

  /// checks whether the given logical qubit exists in the initial layout.
  /// \param logicalQubitIndex the logical qubit index to check
  /// \return whether the given logical qubit exists in the initial layout and
  /// to which physical qubit it is mapped
  [[nodiscard]] std::pair<bool, std::optional<Qubit>>
  containsLogicalQubit(Qubit logicalQubitIndex) const;

  ///---------------------------------------------------------------------------
  ///                            \n Operations \n
  ///---------------------------------------------------------------------------

  /// Adds a global phase to the quantum circuit.
  /// \param angle the angle to add
  void gphase(fp angle);

#define DECLARE_SINGLE_TARGET_OPERATION(op)                                    \
  void op(Qubit target);                                                       \
  void c##op(const Control& control, Qubit target);                            \
  void mc##op(const Controls& controls, const Qubit target);

  DECLARE_SINGLE_TARGET_OPERATION(i)
  DECLARE_SINGLE_TARGET_OPERATION(x)
  DECLARE_SINGLE_TARGET_OPERATION(y)
  DECLARE_SINGLE_TARGET_OPERATION(z)
  DECLARE_SINGLE_TARGET_OPERATION(h)
  DECLARE_SINGLE_TARGET_OPERATION(s)
  DECLARE_SINGLE_TARGET_OPERATION(sdg)
  DECLARE_SINGLE_TARGET_OPERATION(t)
  DECLARE_SINGLE_TARGET_OPERATION(tdg)
  DECLARE_SINGLE_TARGET_OPERATION(v)
  DECLARE_SINGLE_TARGET_OPERATION(vdg)
  DECLARE_SINGLE_TARGET_OPERATION(sx)
  DECLARE_SINGLE_TARGET_OPERATION(sxdg)

#undef DECLARE_SINGLE_TARGET_OPERATION

#define DECLARE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(op, param)            \
  void op(const SymbolOrNumber&(param), Qubit target);                         \
  void c##op(const SymbolOrNumber&(param), const Control& control,             \
             Qubit target);                                                    \
  void mc##op(const SymbolOrNumber&(param), const Controls& controls,          \
              Qubit target);

  DECLARE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rx, theta)
  DECLARE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(ry, theta)
  DECLARE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rz, theta)
  DECLARE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(p, theta)

#undef DECLARE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION

#define DECLARE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)      \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          Qubit target);                                                       \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const Control& control, const Qubit target);                      \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const Controls& controls, const Qubit target);

  DECLARE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(u2, phi, lambda)

#undef DECLARE_SINGLE_TARGET_TWO_PARAMETER_OPERATION

#define DECLARE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(op, param0, param1,    \
                                                        param2)                \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const SymbolOrNumber&(param2), Qubit target);                        \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const SymbolOrNumber&(param2), const Control& control,            \
             Qubit target);                                                    \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const SymbolOrNumber&(param2), const Controls& controls,         \
              Qubit target);

  DECLARE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(u, theta, phi, lambda)

#undef DECLARE_SINGLE_TARGET_THREE_PARAMETER_OPERATION

#define DECLARE_TWO_TARGET_OPERATION(op)                                       \
  void op(const Qubit target0, const Qubit target1);                           \
  void c##op(const Control& control, Qubit target0, Qubit target1);            \
  void mc##op(const Controls& controls, Qubit target0, Qubit target1);

  DECLARE_TWO_TARGET_OPERATION(swap) // NOLINT: bugprone-exception-escape
  DECLARE_TWO_TARGET_OPERATION(dcx)
  DECLARE_TWO_TARGET_OPERATION(ecr)
  DECLARE_TWO_TARGET_OPERATION(iswap)
  DECLARE_TWO_TARGET_OPERATION(iswapdg)
  DECLARE_TWO_TARGET_OPERATION(peres)
  DECLARE_TWO_TARGET_OPERATION(peresdg)
  DECLARE_TWO_TARGET_OPERATION(move)

#undef DECLARE_TWO_TARGET_OPERATION

#define DECLARE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(op, param)               \
  void op(const SymbolOrNumber&(param), Qubit target0, Qubit target1);         \
  void c##op(const SymbolOrNumber&(param), const Control& control,             \
             Qubit target0, Qubit target1);                                    \
  void mc##op(const SymbolOrNumber&(param), const Controls& controls,          \
              Qubit target0, Qubit target1);

  DECLARE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rxx, theta)
  DECLARE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(ryy, theta)
  DECLARE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzz, theta)
  DECLARE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzx, theta)

#undef DECLARE_TWO_TARGET_SINGLE_PARAMETER_OPERATION

#define DECLARE_TWO_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)         \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          Qubit target0, Qubit target1);                                       \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const Control& control, Qubit target0, Qubit target1);            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const Controls& controls, Qubit target0, Qubit target1);

  DECLARE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_minus_yy, theta, beta)
  DECLARE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_plus_yy, theta, beta)

#undef DECLARE_TWO_TARGET_TWO_PARAMETER_OPERATION

  void measure(Qubit qubit, std::size_t bit);
  void measure(Qubit qubit, const std::pair<std::string, Bit>& registerBit);
  void measure(const Targets& qubits, const std::vector<Bit>& bits);

  /**
   * @brief Add measurements to all qubits
   * @param addBits Whether to add new classical bits to the circuit
   * @details This function adds measurements to all qubits in the circuit and
   * appends a new classical register (named "meas") to the circuit if addBits
   * is true. Otherwise, qubit q is measured into classical bit q.
   */
  void measureAll(bool addBits = true);

  void reset(Qubit target);
  void reset(const Targets& targets);

  void barrier();
  void barrier(Qubit target);
  void barrier(const Targets& targets);

  void classicControlled(OpType op, Qubit target,
                         const ClassicalRegister& controlRegister,
                         std::uint64_t expectedValue = 1U,
                         ComparisonKind cmp = Eq,
                         const std::vector<fp>& params = {});
  void classicControlled(OpType op, Qubit target, Control control,
                         const ClassicalRegister& controlRegister,
                         std::uint64_t expectedValue = 1U,
                         ComparisonKind cmp = Eq,
                         const std::vector<fp>& params = {});
  void classicControlled(OpType op, Qubit target, const Controls& controls,
                         const ClassicalRegister& controlRegister,
                         std::uint64_t expectedValue = 1U,
                         ComparisonKind cmp = Eq,
                         const std::vector<fp>& params = {});

  /// strip away qubits with no operations applied to them and which do not pop
  /// up in the output permutation \param force if true, also strip away idle
  /// qubits occurring in the output permutation
  void stripIdleQubits(bool force = false, bool reduceIOpermutations = true);

  void import(const std::string& filename);
  void import(const std::string& filename, Format format);
  void import(std::istream& is, Format format);
  void initializeIOMapping();
  // append measurements to the end of the circuit according to the tracked
  // output permutation
  void appendMeasurementsAccordingToOutputPermutation(
      const std::string& registerName = "c");

  // this function augments a given circuit by additional registers
  void addQubitRegister(std::size_t nq, const std::string& regName = "q");
  void addClassicalRegister(std::size_t nc, const std::string& regName = "c");
  void addAncillaryRegister(std::size_t nq, const std::string& regName = "anc");
  // a function to combine all quantum registers (qregs and ancregs) into a
  // single register (useful for circuits mapped to a device)
  void unifyQuantumRegisters(const std::string& regName = "q");

  // removes a specific logical qubit and returns the index of the physical
  // qubit in the initial layout as well as the index of the removed physical
  // qubit's output permutation i.e., initialLayout[physical_qubit] =
  // logical_qubit and outputPermutation[physicalQubit] = output_qubit
  std::pair<Qubit, std::optional<Qubit>> removeQubit(Qubit logicalQubitIndex);

  // adds physical qubit as ancillary qubit and gives it the appropriate output
  // mapping
  void addAncillaryQubit(Qubit physicalQubitIndex,
                         std::optional<Qubit> outputQubitIndex);
  // try to add logical qubit to circuit and assign it to physical qubit with
  // certain output permutation value
  void addQubit(Qubit logicalQubitIndex, Qubit physicalQubitIndex,
                std::optional<Qubit> outputQubitIndex);

  QuantumComputation instantiate(const VariableAssignment& assignment) const;
  void instantiateInplace(const VariableAssignment& assignment);

  void addVariable(const SymbolOrNumber& expr);

  template <typename... Vars> void addVariables(const Vars&... vars) {
    (addVariable(vars), ...);
  }

  [[nodiscard]] bool isVariableFree() const;

  /**
   * @brief Invert the circuit
   * @details Inverts the circuit by inverting all operations and reversing the
   * order of the operations. Additionally, the initial layout and output
   * permutation are swapped. If the circuit has different initial
   * layout and output permutation sizes, the initial layout and output
   * permutation will not be swapped.
   */
  void invert();

  [[nodiscard]] bool operator==(const QuantumComputation& rhs) const;
  [[nodiscard]] bool operator!=(const QuantumComputation& rhs) const {
    return !(*this == rhs);
  }

  /**
   * printing
   */
  std::ostream& print(std::ostream& os) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QuantumComputation& qc) {
    return qc.print(os);
  }

  std::ostream& printStatistics(std::ostream& os) const;

  std::ostream& printRegisters(std::ostream& os = std::cout) const;

  static std::ostream& printPermutation(const Permutation& permutation,
                                        std::ostream& os = std::cout);

  void dump(const std::string& filename, Format format) const;
  void dump(const std::string& filename) const;
  void dump(std::ostream& of, Format format) const;
  void dumpOpenQASM2(std::ostream& of) const { dumpOpenQASM(of, false); }
  void dumpOpenQASM3(std::ostream& of) const { dumpOpenQASM(of, true); }

  /**
   * @brief Dumps the circuit in OpenQASM format to the given output stream
   * @details One might want to call `ensureContiguousInitialLayout` before
   * calling this function to ensure full layout information is available.
   * @param of The output stream to write the OpenQASM representation to
   * @param openQasm3 Whether to use OpenQASM 3.0 or 2.0
   */
  void dumpOpenQASM(std::ostream& of, bool openQasm3) const;

  /**
   * @brief Returns the OpenQASM representation of the circuit
   * @param qasm3 Whether to use OpenQASM 3.0 or 2.0
   * @return The OpenQASM representation of the circuit
   */
  [[nodiscard]] std::string toQASM(bool qasm3 = true) const;

  // this convenience method allows to turn a circuit into a compound operation.
  std::unique_ptr<CompoundOperation> asCompoundOperation() {
    return std::make_unique<CompoundOperation>(std::move(ops));
  }

  // this convenience method allows to turn a circuit into an operation.
  std::unique_ptr<Operation> asOperation();

  void reset();

  /**
   * @brief Reorders the operations in the quantum computation to establish a
   * canonical order
   * @details Uses iterative breadth-first search starting from the topmost
   * qubit.
   */
  void reorderOperations();

  /**
   * @brief Check whether the quantum computation contains dynamic circuit
   * primitives
   * @details Dynamic circuit primitives are mid-circuit measurements, resets,
   * or classical control flow operations. This method traverses the whole
   * circuit once until it finds a dynamic operation.
   * @return Whether the quantum computation contains dynamic circuit primitives
   */
  [[nodiscard]] bool isDynamic() const;

protected:
  void importOpenQASM3(std::istream& is);
  void importReal(std::istream& is);
  int readRealHeader(std::istream& is);
  void readRealGateDescriptions(std::istream& is, int line);
  void importTFC(std::istream& is);
  int readTFCHeader(std::istream& is, std::map<std::string, Qubit>& varMap);
  void readTFCGateDescriptions(std::istream& is, int line,
                               std::map<std::string, Qubit>& varMap);
  void importQC(std::istream& is);
  int readQCHeader(std::istream& is, std::map<std::string, Qubit>& varMap);
  void readQCGateDescriptions(std::istream& is, int line,
                              std::map<std::string, Qubit>& varMap);

  [[nodiscard]] std::size_t getSmallestAncillary() const {
    for (std::size_t i = 0; i < ancillary.size(); ++i) {
      if (ancillary[i]) {
        return i;
      }
    }
    return ancillary.size();
  }

  [[nodiscard]] std::size_t getSmallestGarbage() const {
    for (std::size_t i = 0; i < garbage.size(); ++i) {
      if (garbage[i]) {
        return i;
      }
    }
    return garbage.size();
  }
  [[nodiscard]] bool isLastOperationOnQubit(const const_iterator& opIt) const {
    const auto end = ops.cend();
    return isLastOperationOnQubit(opIt, end);
  }
  void checkQubitRange(Qubit qubit) const;
  void checkQubitRange(Qubit qubit, const Controls& controls) const;
  void checkQubitRange(Qubit qubit0, Qubit qubit1,
                       const Controls& controls) const;
  void checkQubitRange(const std::vector<Qubit>& qubits) const;
  void checkBitRange(Bit bit) const;
  void checkBitRange(const std::vector<Bit>& bits) const;
  void checkClassicalRegister(const ClassicalRegister& creg) const;

  /**
   * Pass-Through
   */
public:
  // Iterators (pass-through)
  auto begin() noexcept { return ops.begin(); }
  [[nodiscard]] auto begin() const noexcept { return ops.begin(); }
  [[nodiscard]] auto cbegin() const noexcept { return ops.cbegin(); }
  auto end() noexcept { return ops.end(); }
  [[nodiscard]] auto end() const noexcept { return ops.end(); }
  [[nodiscard]] auto cend() const noexcept { return ops.cend(); }
  auto rbegin() noexcept { return ops.rbegin(); }
  [[nodiscard]] auto rbegin() const noexcept { return ops.rbegin(); }
  [[nodiscard]] auto crbegin() const noexcept { return ops.crbegin(); }
  auto rend() noexcept { return ops.rend(); }
  [[nodiscard]] auto rend() const noexcept { return ops.rend(); }
  [[nodiscard]] auto crend() const noexcept { return ops.crend(); }

  // Capacity (pass-through)
  [[nodiscard]] bool empty() const noexcept { return ops.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] std::size_t max_size() const noexcept { return ops.max_size(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return ops.capacity(); }

  void reserve(const std::size_t newCap) { ops.reserve(newCap); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  void shrink_to_fit() { ops.shrink_to_fit(); }

  // Modifiers (pass-through)
  void clear() noexcept { ops.clear(); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  void pop_back() { ops.pop_back(); }
  void resize(const std::size_t count) { ops.resize(count); }
  iterator erase(const const_iterator pos) { return ops.erase(pos); }
  iterator erase(const const_iterator first, const const_iterator last) {
    return ops.erase(first, last);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void push_back(const T& op) {
    ops.push_back(std::make_unique<T>(op));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T, class... Args> void emplace_back(Args&&... args) {
    ops.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>& op) {
    ops.emplace_back(std::move(op));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>&& op) {
    ops.emplace_back(std::move(op));
  }

  template <class T> iterator insert(const_iterator pos, T&& op) {
    return ops.insert(pos, std::forward<T>(op));
  }

  [[nodiscard]] const auto& at(const std::size_t i) const { return ops.at(i); }
  [[nodiscard]] auto& at(const std::size_t i) { return ops.at(i); }
  [[nodiscard]] const auto& front() const { return ops.front(); }
  [[nodiscard]] const auto& back() const { return ops.back(); }

  // reverse
  void reverse();
};
} // namespace qc
