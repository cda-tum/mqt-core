#pragma once

#include "Definitions.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/StandardOperation.hpp"
#include "operations/SymbolicOperation.hpp"
#include "parsers/qasm_parser/Parser.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qc {
class CircuitOptimizer;

class QuantumComputation {
public:
  using iterator = typename std::vector<std::unique_ptr<Operation>>::iterator;
  using const_iterator =
      typename std::vector<std::unique_ptr<Operation>>::const_iterator;

  friend class CircuitOptimizer;

protected:
  std::vector<std::unique_ptr<Operation>> ops{};
  std::size_t nqubits = 0;
  std::size_t nclassics = 0;
  std::size_t nancillae = 0;
  std::size_t maxControls = 0;
  std::string name;

  // register names are used as keys, while the values are `{startIndex,
  // length}` pairs
  QuantumRegisterMap qregs{};
  ClassicalRegisterMap cregs{};
  QuantumRegisterMap ancregs{};

  std::mt19937_64 mt;
  std::size_t seed = 0;

  fp globalPhase = 0.;

  std::unordered_set<sym::Variable> occuringVariables;

  void importOpenQASM(std::istream& is);
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
  void importGRCS(std::istream& is);

  template <class RegisterType>
  static void printSortedRegisters(const RegisterMap<RegisterType>& regmap,
                                   const std::string& identifier,
                                   std::ostream& of) {
    // sort regs by start index
    std::map<decltype(RegisterType::first),
             std::pair<std::string, RegisterType>>
        sortedRegs{};
    for (const auto& reg : regmap) {
      sortedRegs.insert({reg.second.first, reg});
    }

    for (const auto& reg : sortedRegs) {
      of << identifier << " " << reg.second.first << "["
         << reg.second.second.second << "];" << std::endl;
    }
  }
  template <class RegisterType>
  static void consolidateRegister(RegisterMap<RegisterType>& regs) {
    bool finished = false;
    while (!finished) {
      for (const auto& qreg : regs) {
        finished = true;
        auto regname = qreg.first;
        // check if lower part of register
        if (regname.length() > 2 &&
            regname.compare(regname.size() - 2, 2, "_l") == 0) {
          auto lowidx = qreg.second.first;
          auto lownum = qreg.second.second;
          // search for higher part of register
          auto highname = regname.substr(0, regname.size() - 1) + 'h';
          auto it = regs.find(highname);
          if (it != regs.end()) {
            auto highidx = it->second.first;
            auto highnum = it->second.second;
            // fusion of registers possible
            if (lowidx + lownum == highidx) {
              finished = false;
              auto targetname = regname.substr(0, regname.size() - 2);
              auto targetidx = lowidx;
              auto targetnum = lownum + highnum;
              regs.insert({targetname, {targetidx, targetnum}});
              regs.erase(regname);
              regs.erase(highname);
            }
          }
          break;
        }
      }
    }
  }

  template <class RegisterType>
  static void createRegisterArray(const RegisterMap<RegisterType>& regs,
                                  RegisterNames& regnames) {
    regnames.clear();
    std::stringstream ss;
    // sort regs by start index
    std::map<decltype(RegisterType::first),
             std::pair<std::string, RegisterType>>
        sortedRegs{};
    for (const auto& reg : regs) {
      sortedRegs.insert({reg.second.first, reg});
    }

    for (const auto& reg : sortedRegs) {
      for (decltype(RegisterType::second) i = 0; i < reg.second.second.second;
           i++) {
        ss << reg.second.first << "[" << i << "]";
        regnames.push_back(std::make_pair(reg.second.first, ss.str()));
        ss.str(std::string());
      }
    }
  }

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

public:
  QuantumComputation() = default;
  explicit QuantumComputation(const std::size_t nq, const std::size_t nc = 0U,
                              const std::size_t s = 0)
      : seed(s) {
    addQubitRegister(nq);
    if (nc > 0) {
      addClassicalRegister(nc);
    }
    if (seed != 0) {
      mt.seed(seed);
    } else {
      // create and properly seed rng
      std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
          randomData{};
      std::random_device rd;
      std::generate(std::begin(randomData), std::end(randomData),
                    [&rd]() { return rd(); });
      std::seed_seq seeds(std::begin(randomData), std::end(randomData));
      mt.seed(seeds);
    }
  }
  explicit QuantumComputation(const std::string& filename,
                              const std::size_t s = 0U)
      : seed(s) {
    import(filename);
    if (seed != 0U) {
      mt.seed(seed);
    } else {
      // create and properly seed rng
      std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
          randomData{};
      std::random_device rd;
      std::generate(std::begin(randomData), std::end(randomData),
                    [&rd]() { return rd(); });
      std::seed_seq seeds(std::begin(randomData), std::end(randomData));
      mt.seed(seeds);
    }
  }
  QuantumComputation(QuantumComputation&& qc) noexcept = default;
  QuantumComputation& operator=(QuantumComputation&& qc) noexcept = default;
  QuantumComputation(const QuantumComputation& qc)
      : nqubits(qc.nqubits), nclassics(qc.nclassics), nancillae(qc.nancillae),
        maxControls(qc.maxControls), name(qc.name), qregs(qc.qregs),
        cregs(qc.cregs), ancregs(qc.ancregs), mt(qc.mt), seed(qc.seed),
        globalPhase(qc.globalPhase), occuringVariables(qc.occuringVariables),
        initialLayout(qc.initialLayout),
        outputPermutation(qc.outputPermutation), ancillary(qc.ancillary),
        garbage(qc.garbage) {
    ops.reserve(qc.ops.size());
    for (const auto& op : qc.ops) {
      emplace_back(op->clone());
    }
  }
  QuantumComputation& operator=(const QuantumComputation& qc) {
    if (this != &qc) {
      nqubits = qc.nqubits;
      nclassics = qc.nclassics;
      nancillae = qc.nancillae;
      maxControls = qc.maxControls;
      name = qc.name;
      qregs = qc.qregs;
      cregs = qc.cregs;
      ancregs = qc.ancregs;
      mt = qc.mt;
      seed = qc.seed;
      globalPhase = qc.globalPhase;
      occuringVariables = qc.occuringVariables;
      initialLayout = qc.initialLayout;
      outputPermutation = qc.outputPermutation;
      ancillary = qc.ancillary;
      garbage = qc.garbage;

      ops.clear();
      ops.reserve(qc.ops.size());
      for (const auto& op : qc.ops) {
        emplace_back(op->clone());
      }
    }
    return *this;
  }
  virtual ~QuantumComputation() = default;

  [[nodiscard]] virtual std::size_t getNops() const { return ops.size(); }
  [[nodiscard]] std::size_t getNqubits() const { return nqubits + nancillae; }
  [[nodiscard]] std::size_t getNancillae() const { return nancillae; }
  [[nodiscard]] std::size_t getNqubitsWithoutAncillae() const {
    return nqubits;
  }
  [[nodiscard]] std::size_t getNcbits() const { return nclassics; }
  [[nodiscard]] std::string getName() const { return name; }
  [[nodiscard]] const QuantumRegisterMap& getQregs() const { return qregs; }
  [[nodiscard]] const ClassicalRegisterMap& getCregs() const { return cregs; }
  [[nodiscard]] const QuantumRegisterMap& getANCregs() const { return ancregs; }
  [[nodiscard]] decltype(mt)& getGenerator() { return mt; }

  [[nodiscard]] fp getGlobalPhase() const { return globalPhase; }

  void setName(const std::string& n) { name = n; }

  // physical qubits are used as keys, logical qubits as values
  Permutation initialLayout{};
  Permutation outputPermutation{};

  std::vector<bool> ancillary{};
  std::vector<bool> garbage{};

  [[nodiscard]] std::size_t getNindividualOps() const;
  [[nodiscard]] std::size_t getNsingleQubitOps() const;
  [[nodiscard]] std::size_t getDepth() const;

  [[nodiscard]] std::string getQubitRegister(Qubit physicalQubitIndex) const;
  [[nodiscard]] std::string getClassicalRegister(Bit classicalIndex) const;
  static Qubit getHighestLogicalQubitIndex(const Permutation& permutation);
  [[nodiscard]] Qubit getHighestLogicalQubitIndex() const {
    return getHighestLogicalQubitIndex(initialLayout);
  };
  [[nodiscard]] std::pair<std::string, Qubit>
  getQubitRegisterAndIndex(Qubit physicalQubitIndex) const;
  [[nodiscard]] std::pair<std::string, Bit>
  getClassicalRegisterAndIndex(Bit classicalIndex) const;

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
  void setLogicalQubitAncillary(const Qubit logicalQubitIndex) {
    if (logicalQubitIsAncillary(logicalQubitIndex)) {
      return;
    }
    ancillary[logicalQubitIndex] = true;
    nancillae++;
    nqubits--;
  }
  [[nodiscard]] bool
  logicalQubitIsGarbage(const Qubit logicalQubitIndex) const {
    return garbage[logicalQubitIndex];
  }
  void setLogicalQubitGarbage(Qubit logicalQubitIndex);
  [[nodiscard]] const std::vector<bool>& getAncillary() const {
    return ancillary;
  }
  [[nodiscard]] const std::vector<bool>& getGarbage() const { return garbage; }

  /// checks whether the given logical qubit exists in the initial layout.
  /// \param logicalQubitIndex the logical qubit index to check
  /// \return whether the given logical qubit exists in the initial layout and
  /// to which physical qubit it is mapped
  [[nodiscard]] std::pair<bool, std::optional<Qubit>>
  containsLogicalQubit(Qubit logicalQubitIndex) const;

  /// Adds a global phase to the quantum circuit.
  /// \param angle the angle to add
  void gphase(const fp& angle) {
    globalPhase += angle;
    // normalize to [0, 2pi)
    while (globalPhase < 0) {
      globalPhase += 2 * PI;
    }
    while (globalPhase >= 2 * PI) {
      globalPhase -= 2 * PI;
    }
  }

  ///---------------------------------------------------------------------------
  ///                            \n Operations \n
  ///---------------------------------------------------------------------------

#define DEFINE_SINGLE_TARGET_OPERATION(op)                                     \
  void op(const Qubit target) { mc##op(Controls{}, target); }                  \
  void c##op(const Control& control, const Qubit target) {                     \
    mc##op(Controls{control}, target);                                         \
  }                                                                            \
  void mc##op(const Controls& controls, const Qubit target) {                  \
    checkQubitRange(target, controls);                                         \
    emplace_back<StandardOperation>(getNqubits(), controls, target,            \
                                    OP_NAME_TO_TYPE.at(#op));                  \
  }

  DEFINE_SINGLE_TARGET_OPERATION(i)
  DEFINE_SINGLE_TARGET_OPERATION(x)
  DEFINE_SINGLE_TARGET_OPERATION(y)
  DEFINE_SINGLE_TARGET_OPERATION(z)
  DEFINE_SINGLE_TARGET_OPERATION(h)
  DEFINE_SINGLE_TARGET_OPERATION(s)
  DEFINE_SINGLE_TARGET_OPERATION(sdg)
  DEFINE_SINGLE_TARGET_OPERATION(t)
  DEFINE_SINGLE_TARGET_OPERATION(tdg)
  DEFINE_SINGLE_TARGET_OPERATION(v)
  DEFINE_SINGLE_TARGET_OPERATION(vdg)
  DEFINE_SINGLE_TARGET_OPERATION(sx)
  DEFINE_SINGLE_TARGET_OPERATION(sxdg)

#define DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(op, param)             \
  void op(const SymbolOrNumber&(param), const Qubit target) {                  \
    mc##op(param, Controls{}, target);                                         \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param), const Control& control,             \
             const Qubit target) {                                             \
    mc##op(param, Controls{control}, target);                                  \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param), const Controls& controls,          \
              const Qubit target) {                                            \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param)) {                                   \
      emplace_back<StandardOperation>(getNqubits(), controls, target,          \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{std::get<fp>(param)});       \
    } else {                                                                   \
      addVariables(param);                                                     \
      emplace_back<SymbolicOperation>(getNqubits(), controls, target,          \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param});                     \
    }                                                                          \
  }

  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rx, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(ry, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rz, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(p, theta)

#define DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)       \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const Qubit target) {                                                \
    mc##op(param0, param1, Controls{}, target);                                \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const Control& control, const Qubit target) {                     \
    mc##op(param0, param1, Controls{control}, target);                         \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const Controls& controls, const Qubit target) {                  \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1)) {                                  \
      emplace_back<StandardOperation>(                                         \
          getNqubits(), controls, target, OP_NAME_TO_TYPE.at(#op),             \
          std::vector{std::get<fp>(param0), std::get<fp>(param1)});            \
    } else {                                                                   \
      addVariables(param0, param1);                                            \
      emplace_back<SymbolicOperation>(getNqubits(), controls, target,          \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param0, param1});            \
    }                                                                          \
  }

  DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(u2, phi, lambda)

#define DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(op, param0, param1,     \
                                                       param2)                 \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const SymbolOrNumber&(param2), const Qubit target) {                 \
    mc##op(param0, param1, param2, Controls{}, target);                        \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const SymbolOrNumber&(param2), const Control& control,            \
             const Qubit target) {                                             \
    mc##op(param0, param1, param2, Controls{control}, target);                 \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const SymbolOrNumber&(param2), const Controls& controls,         \
              const Qubit target) {                                            \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1) &&                                  \
        std::holds_alternative<fp>(param2)) {                                  \
      emplace_back<StandardOperation>(                                         \
          getNqubits(), controls, target, OP_NAME_TO_TYPE.at(#op),             \
          std::vector{std::get<fp>(param0), std::get<fp>(param1),              \
                      std::get<fp>(param2)});                                  \
    } else {                                                                   \
      addVariables(param0, param1, param2);                                    \
      emplace_back<SymbolicOperation>(getNqubits(), controls, target,          \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param0, param1, param2});    \
    }                                                                          \
  }

  DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(u, theta, phi, lambda)

#define DEFINE_TWO_TARGET_OPERATION(op)                                        \
  void op(const Qubit target0, const Qubit target1) {                          \
    mc##op(Controls{}, target0, target1);                                      \
  }                                                                            \
  void c##op(const Control& control, const Qubit target0,                      \
             const Qubit target1) {                                            \
    mc##op(Controls{control}, target0, target1);                               \
  }                                                                            \
  void mc##op(const Controls& controls, const Qubit target0,                   \
              const Qubit target1) {                                           \
    checkQubitRange(target0, target1, controls);                               \
    emplace_back<StandardOperation>(getNqubits(), controls, target0, target1,  \
                                    OP_NAME_TO_TYPE.at(#op));                  \
  }

  DEFINE_TWO_TARGET_OPERATION(swap)
  DEFINE_TWO_TARGET_OPERATION(dcx)
  DEFINE_TWO_TARGET_OPERATION(ecr)
  DEFINE_TWO_TARGET_OPERATION(iswap)
  DEFINE_TWO_TARGET_OPERATION(peres)
  DEFINE_TWO_TARGET_OPERATION(peresdg)

#define DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(op, param)                \
  void op(const SymbolOrNumber&(param), const Qubit target0,                   \
          const Qubit target1) {                                               \
    mc##op(param, Controls{}, target0, target1);                               \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param), const Control& control,             \
             const Qubit target0, const Qubit target1) {                       \
    mc##op(param, Controls{control}, target0, target1);                        \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param), const Controls& controls,          \
              const Qubit target0, const Qubit target1) {                      \
    checkQubitRange(target0, target1, controls);                               \
    if (std::holds_alternative<fp>(param)) {                                   \
      emplace_back<StandardOperation>(getNqubits(), controls, target0,         \
                                      target1, OP_NAME_TO_TYPE.at(#op),        \
                                      std::vector{std::get<fp>(param)});       \
    } else {                                                                   \
      addVariables(param);                                                     \
      emplace_back<SymbolicOperation>(getNqubits(), controls, target0,         \
                                      target1, OP_NAME_TO_TYPE.at(#op),        \
                                      std::vector{param});                     \
    }                                                                          \
  }

  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rxx, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(ryy, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzz, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzx, theta)

#define DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)          \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const Qubit target0, const Qubit target1) {                          \
    mc##op(param0, param1, Controls{}, target0, target1);                      \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const Control& control, const Qubit target0,                      \
             const Qubit target1) {                                            \
    mc##op(param0, param1, Controls{control}, target0, target1);               \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const Controls& controls, const Qubit target0,                   \
              const Qubit target1) {                                           \
    checkQubitRange(target0, target1, controls);                               \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1)) {                                  \
      emplace_back<StandardOperation>(                                         \
          getNqubits(), controls, target0, target1, OP_NAME_TO_TYPE.at(#op),   \
          std::vector{std::get<fp>(param0), std::get<fp>(param1)});            \
    } else {                                                                   \
      addVariables(param0, param1);                                            \
      emplace_back<SymbolicOperation>(getNqubits(), controls, target0,         \
                                      target1, OP_NAME_TO_TYPE.at(#op),        \
                                      std::vector{param0, param1});            \
    }                                                                          \
  }

  DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_minus_yy, theta, beta)
  DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_plus_yy, theta, beta)

#undef DEFINE_SINGLE_TARGET_OPERATION
#undef DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_OPERATION
#undef DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION

  void measure(const Qubit qubit, const std::size_t bit) {
    checkQubitRange(qubit);
    checkBitRange(bit);
    emplace_back<NonUnitaryOperation>(getNqubits(), qubit, bit);
  }

  void measure(Qubit qubit, const std::pair<std::string, Bit>& registerBit);

  void measure(const Targets& qubits, const std::vector<Bit>& bits) {
    checkQubitRange(qubits);
    checkBitRange(bits);
    emplace_back<NonUnitaryOperation>(getNqubits(), qubits, bits);
  }

  /**
   * @brief Add measurements to all qubits
   * @param addBits Whether to add new classical bits to the circuit
   * @details This function adds measurements to all qubits in the circuit and
   * appends a new classical register (named "meas") to the circuit if addBits
   * is true. Otherwise, qubit q is measured into classical bit q.
   */
  void measureAll(bool addBits = true);

  void reset(const Qubit target) {
    checkQubitRange(target);
    emplace_back<NonUnitaryOperation>(getNqubits(), std::vector<Qubit>{target},
                                      qc::Reset);
  }
  void reset(const Targets& targets) {
    checkQubitRange(targets);
    emplace_back<NonUnitaryOperation>(getNqubits(), targets, qc::Reset);
  }

  void barrier() {
    std::vector<Qubit> targets(getNqubits());
    std::iota(targets.begin(), targets.end(), 0);
    emplace_back<StandardOperation>(getNqubits(), targets, qc::Barrier);
  }
  void barrier(const Qubit target) {
    checkQubitRange(target);
    emplace_back<StandardOperation>(getNqubits(), target, qc::Barrier);
  }
  void barrier(const Targets& targets) {
    checkQubitRange(targets);
    emplace_back<StandardOperation>(getNqubits(), targets, qc::Barrier);
  }

  void classicControlled(const OpType op, const Qubit target,
                         const ClassicalRegister& controlRegister,
                         const std::uint64_t expectedValue = 1U,
                         const std::vector<fp>& params = {}) {
    classicControlled(op, target, Controls{}, controlRegister, expectedValue,
                      params);
  }
  void classicControlled(const OpType op, const Qubit target,
                         const Control control,
                         const ClassicalRegister& controlRegister,
                         const std::uint64_t expectedValue = 1U,
                         const std::vector<fp>& params = {}) {
    classicControlled(op, target, Controls{control}, controlRegister,
                      expectedValue, params);
  }
  void classicControlled(const OpType op, const Qubit target,
                         const Controls& controls,
                         const ClassicalRegister& controlRegister,
                         const std::uint64_t expectedValue = 1U,
                         const std::vector<fp>& params = {}) {
    checkQubitRange(target, controls);
    checkClassicalRegister(controlRegister);
    std::unique_ptr<Operation> gate = std::make_unique<StandardOperation>(
        getNqubits(), controls, target, op, params);
    emplace_back<ClassicControlledOperation>(std::move(gate), controlRegister,
                                             expectedValue);
  }

  /// strip away qubits with no operations applied to them and which do not pop
  /// up in the output permutation \param force if true, also strip away idle
  /// qubits occurring in the output permutation
  void stripIdleQubits(bool force = false, bool reduceIOpermutations = true);

  void import(const std::string& filename);
  void import(const std::string& filename, Format format);
  void import(std::istream& is, Format format) {
    import(std::move(is), format);
  }
  void import(std::istream&& is, Format format);
  void initializeIOMapping();
  // append measurements to the end of the circuit according to the tracked
  // output permutation
  void appendMeasurementsAccordingToOutputPermutation(
      const std::string& registerName = "c");
  // search for current position of target value in map and afterwards exchange
  // it with the value at new position
  static void findAndSWAP(Qubit targetValue, Qubit newPosition,
                          Permutation& map) {
    for (const auto& q : map) {
      if (q.second == targetValue) {
        std::swap(map.at(newPosition), map.at(q.first));
        break;
      }
    }
  }

  // this function augments a given circuit by additional registers
  void addQubitRegister(std::size_t, const std::string& regName = "q");
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

  void updateMaxControls(const std::size_t ncontrols) {
    maxControls = std::max(ncontrols, maxControls);
  }

  void instantiate(const VariableAssignment& assignment);

  void addVariable(const SymbolOrNumber& expr);

  template <typename... Vars> void addVariables(const Vars&... vars) {
    (addVariable(vars), ...);
  }

  [[nodiscard]] bool isVariableFree() const {
    return std::all_of(ops.begin(), ops.end(), [](const auto& op) {
      return !op->isSymbolicOperation();
    });
  }

  [[nodiscard]] const std::unordered_set<sym::Variable>& getVariables() const {
    return occuringVariables;
  }

  /**
   * @brief Invert the circuit
   * @details Inverts the circuit by inverting all operations and reversing the
   * order of the operations. Additionally, the initial layout and output
   * permutation are swapped. If the circuit has different initial
   * layout and output permutation sizes, the initial layout and output
   * permutation will not be swapped.
   */
  void invert() {
    for (auto& op : ops) {
      op->invert();
    }
    std::reverse(ops.begin(), ops.end());

    if (initialLayout.size() == outputPermutation.size()) {
      std::swap(initialLayout, outputPermutation);
    } else {
      std::cerr << "Warning: Inverting a circuit with different initial layout "
                   "and output permutation sizes. This is not supported yet.\n"
                   "The circuit will be inverted, but the initial layout and "
                   "output permutation will not be swapped.\n";
    }
  }

  /**
   * printing
   */
  virtual std::ostream& print(std::ostream& os) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QuantumComputation& qc) {
    return qc.print(os);
  }

  static void printBin(std::size_t n, std::stringstream& ss);

  virtual std::ostream& printStatistics(std::ostream& os) const;

  std::ostream& printRegisters(std::ostream& os = std::cout) const;

  static std::ostream& printPermutation(const Permutation& permutation,
                                        std::ostream& os = std::cout);

  virtual void dump(const std::string& filename, Format format);
  virtual void dump(const std::string& filename);
  virtual void dump(std::ostream& of, Format format) {
    dump(std::move(of), format);
  }
  virtual void dump(std::ostream&& of, Format format);
  virtual void dumpOpenQASM(std::ostream& of);

  // this convenience method allows to turn a circuit into a compound operation.
  std::unique_ptr<CompoundOperation> asCompoundOperation() {
    return std::make_unique<CompoundOperation>(getNqubits(), std::move(ops));
  }

  // this convenience method allows to turn a circuit into an operation.
  std::unique_ptr<Operation> asOperation() {
    if (ops.empty()) {
      return {};
    }
    if (ops.size() == 1) {
      auto op = std::move(ops.front());
      ops.clear();
      return op;
    }
    return asCompoundOperation();
  }

  virtual void reset() {
    ops.clear();
    nqubits = 0;
    nclassics = 0;
    nancillae = 0;
    qregs.clear();
    cregs.clear();
    ancregs.clear();
    initialLayout.clear();
    outputPermutation.clear();
  }

  /**
   * Pass-Through
   */

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
  void pop_back() { return ops.pop_back(); }
  void resize(std::size_t count) { ops.resize(count); }
  iterator erase(const_iterator pos) { return ops.erase(pos); }
  iterator erase(const_iterator first, const_iterator last) {
    return ops.erase(first, last);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void push_back(const T& op) {
    if (!ops.empty() && !op.isControlled() && !ops.back()->isControlled()) {
      std::cerr << op.getName() << std::endl;
    }

    ops.push_back(std::make_unique<T>(op));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T, class... Args> void emplace_back(Args&&... args) {
    ops.emplace_back(std::make_unique<T>(args...));
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
  void reverse() { std::reverse(ops.begin(), ops.end()); }
};
} // namespace qc
