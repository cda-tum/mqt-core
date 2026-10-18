/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QDMIDevice.h"

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/Translation/StandardGate.h"
#include "qdmi/common/Common.hpp"
#include "qdmi/common/DeviceConfiguration.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace mqt::catalyst {
namespace {

struct Registry {
  std::mutex mutex;
  uint64_t next = 1;
  std::map<uint64_t, std::weak_ptr<Context>> contexts;
};

struct Readout {
  std::vector<std::pair<std::string, size_t>> outcomes;
};

using Gate = mlir::qc::StandardGate;
struct NamedGate {
  std::string_view name;
  Gate gate;
  size_t controls = 0;
};

} // namespace

static void require(bool valid, const std::string& message) {
  if (!valid) {
    throw std::runtime_error(message);
  }
}

template <class T> static T unwrap(llvm::Expected<T> value) {
  if (!value) {
    throw std::runtime_error(llvm::toString(value.takeError()));
  }
  return std::move(*value);
}

static void validateShots(size_t shots) {
  require(shots > 0 && shots <= static_cast<size_t>(INT64_MAX),
          "QDMI requires a positive shot count that fits int64_t.");
}

static size_t product(size_t lhs, size_t rhs) {
  require(rhs == 0 || lhs <= SIZE_MAX / rhs,
          "Result dimensions overflow size_t.");
  return lhs * rhs;
}

static size_t dimension(size_t width) {
  require(width < std::numeric_limits<size_t>::digits,
          "Dense result dimension does not fit size_t.");
  return size_t{1} << width;
}

static Registry& registry() {
  // Catalyst may destroy pooled devices after other static objects.
  static auto* instance = new Registry();
  return *instance;
}

using Parameters = std::array<std::optional<qdmi::CustomJobParameter>, 5>;

static Parameters borrowParameters(const JobParameters& parameters) {
  Parameters result;
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i]) {
      result[i] = std::visit(
          [](const auto& value) -> qdmi::CustomJobParameter {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                         std::vector<std::byte>>) {
              return std::span<const std::byte>(value);
            } else {
              return value;
            }
          },
          *parameters[i]);
    }
  }
  return result;
}

static Readout readout(const qdmi::Job& job, size_t width, size_t shots,
                       size_t programIndex) {
  validateShots(shots);
  Readout result;
  std::string shotsError;
  try {
    for (auto& bits : job.getShots(programIndex)) {
      result.outcomes.emplace_back(std::move(bits), 1);
    }
  } catch (const std::exception& error) {
    shotsError = error.what();
  }
  if (result.outcomes.empty()) {
    try {
      for (const auto& [bits, count] : job.getCounts(programIndex)) {
        if (count != 0) {
          result.outcomes.emplace_back(bits, count);
        }
      }
    } catch (const std::exception& error) {
      throw std::runtime_error("Cannot read QDMI samples; shots: " +
                               shotsError + "; counts: " + error.what());
    }
  }
  size_t total = 0;
  for (auto& [bits, count] : result.outcomes) {
    std::erase(bits, ' ');
    require(bits.size() == width &&
                bits.find_first_not_of("01") == std::string::npos,
            "QDMI returned a malformed " + std::to_string(width) +
                "-bit sample.");
    require(count <= shots - total,
            "QDMI returned more samples than requested.");
    total += count;
    std::ranges::reverse(bits);
  }
  require(total == shots, "QDMI returned " + std::to_string(total) +
                              " samples for a " + std::to_string(shots) +
                              "-shot job.");
  return result;
}

static size_t basisIndex(const std::string& bits,
                         const std::vector<size_t>& wires) {
  size_t index = 0;
  for (const auto wire : wires) {
    index = (index << 1U) | static_cast<size_t>(bits[wire] - '0');
  }
  return index;
}

constexpr std::array GATES{
    NamedGate{.name = "Identity", .gate = Gate::Id},
    NamedGate{.name = "PauliX", .gate = Gate::X},
    NamedGate{.name = "PauliY", .gate = Gate::Y},
    NamedGate{.name = "PauliZ", .gate = Gate::Z},
    NamedGate{.name = "Hadamard", .gate = Gate::H},
    NamedGate{.name = "S", .gate = Gate::S},
    NamedGate{.name = "T", .gate = Gate::T},
    NamedGate{.name = "SX", .gate = Gate::SX},
    NamedGate{.name = "RX", .gate = Gate::RX},
    NamedGate{.name = "RY", .gate = Gate::RY},
    NamedGate{.name = "RZ", .gate = Gate::RZ},
    NamedGate{.name = "PhaseShift", .gate = Gate::P},
    NamedGate{.name = "GlobalPhase", .gate = Gate::GPhase},
    NamedGate{.name = "U1", .gate = Gate::P},
    NamedGate{.name = "U2", .gate = Gate::U2},
    NamedGate{.name = "U3", .gate = Gate::U3},
    NamedGate{.name = "SWAP", .gate = Gate::SWAP},
    NamedGate{.name = "ISWAP", .gate = Gate::ISWAP},
    NamedGate{.name = "ECR", .gate = Gate::ECR},
    NamedGate{.name = "IsingXX", .gate = Gate::RXX},
    NamedGate{.name = "IsingYY", .gate = Gate::RYY},
    NamedGate{.name = "IsingZZ", .gate = Gate::RZZ},
    NamedGate{.name = "CNOT", .gate = Gate::X, .controls = 1},
    NamedGate{.name = "CY", .gate = Gate::Y, .controls = 1},
    NamedGate{.name = "CZ", .gate = Gate::Z, .controls = 1},
    NamedGate{.name = "Toffoli", .gate = Gate::X, .controls = 2},
    NamedGate{.name = "CSWAP", .gate = Gate::SWAP, .controls = 1},
    NamedGate{.name = "ControlledPhaseShift", .gate = Gate::P, .controls = 1},
    NamedGate{.name = "CRX", .gate = Gate::RX, .controls = 1},
    NamedGate{.name = "CRY", .gate = Gate::RY, .controls = 1},
    NamedGate{.name = "CRZ", .gate = Gate::RZ, .controls = 1},
};

static uint64_t parseContext(const char* kwargs) {
  require(kwargs != nullptr, "Missing QDMI context arguments.");
  std::string_view text(kwargs);
  const auto whitespace = [&] {
    const auto first = text.find_first_not_of(" \t\r\n");
    text.remove_prefix(first == std::string_view::npos ? text.size() : first);
  };
  const auto consume = [&](char expected) {
    whitespace();
    require(!text.empty() && text.front() == expected,
            "Invalid QDMI context arguments.");
    text.remove_prefix(1);
  };
  consume('{');
  whitespace();
  require(!text.empty() && (text.front() == '\'' || text.front() == '"'),
          "Invalid QDMI context key.");
  const char quote = text.front();
  text.remove_prefix(1);
  require(text.starts_with("context"), "Expected the QDMI context key.");
  text.remove_prefix(7);
  require(!text.empty() && text.front() == quote, "Invalid QDMI context key.");
  text.remove_prefix(1);
  consume(':');
  whitespace();
  uint64_t id = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), id);
  require(parsed.ec == std::errc{} && id != 0, "Invalid QDMI context ID.");
  text.remove_prefix(static_cast<size_t>(parsed.ptr - text.data()));
  consume('}');
  whitespace();
  require(text.empty(), "Unexpected QDMI context arguments.");
  return id;
}

struct Context::Impl {
  qdmi::Device device;
  JobParameters parameters;
  mlir::TargetEnvironment environment;
  size_t qubits;
  uint64_t id = 0;
  mutable std::mutex mutex;

  Impl(qdmi::Device input, JobParameters values)
      : device(std::move(input)), parameters(std::move(values)),
        environment(unwrap(mlir::targetEnvironmentFromDevice(device))),
        qubits(device.getQubitsNum()) {}
};

struct PreparedProgram::Impl {
  std::shared_ptr<Context> context;
  mlir::CompiledProgram program;
  size_t qubits;
};

std::string libraryPath() {
  static const char ANCHOR = 0;
  return qdmi::detail::pathToString(qdmi::detail::moduleDirectory(&ANCHOR) /
                                    MQT_CATALYST_LIBRARY_FILENAME);
}

Context::Context(qdmi::Device device, JobParameters parameters)
    : impl_(std::make_unique<Impl>(std::move(device), std::move(parameters))) {}

std::shared_ptr<Context> Context::create(qdmi::Device device,
                                         JobParameters parameters) {
  auto context = std::shared_ptr<Context>(
      new Context(std::move(device), std::move(parameters)));
  auto& entries = registry();
  const std::scoped_lock lock(entries.mutex);
  require(entries.next != 0, "QDMI context ID space exhausted.");
  context->impl_->id = entries.next++;
  entries.contexts.emplace(context->id(), context);
  return context;
}

Context::~Context() {
  auto& entries = registry();
  const std::scoped_lock lock(entries.mutex);
  entries.contexts.erase(id());
}

uint64_t Context::id() const noexcept { return impl_->id; }
size_t Context::qubits() const noexcept { return impl_->qubits; }

PreparedProgram::PreparedProgram(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedProgram::~PreparedProgram() = default;
size_t PreparedProgram::qubits() const noexcept { return impl_->qubits; }
QDMI_Program_Format PreparedProgram::programFormat() const noexcept {
  return impl_->program.programFormat();
}
const std::string& PreparedProgram::payload() const noexcept {
  return impl_->program.payload();
}

qdmi::Job PreparedProgram::submit(size_t shots) const {
  validateShots(shots);
  auto& context = *impl_->context->impl_;
  const std::scoped_lock lock(context.mutex);
  const auto p = borrowParameters(context.parameters);
  return unwrap(mlir::submitProgram(context.device, impl_->program,
                                    static_cast<int64_t>(shots), p[0], p[1],
                                    p[2], p[3], p[4]));
}

std::optional<qdmi::Job> Context::trySubmitBatch(
    const std::vector<std::shared_ptr<PreparedProgram>>& programs,
    size_t shots) const {
  validateShots(shots);
  require(!programs.empty() && programs.front() != nullptr,
          "Cannot submit an empty program batch.");
  const auto format = programs.front()->programFormat();
  const std::scoped_lock lock(impl_->mutex);
  const auto destination =
      unwrap(mlir::targetEnvironmentFromDevice(impl_->device, format));
  for (const auto& program : programs) {
    require(program && program->impl_->context.get() == this &&
                program->programFormat() == format,
            "A program batch must share its QDMI context and format.");
    if (auto error = mlir::validateTargetCompatibility(
            program->impl_->program.environment(), destination)) {
      throw std::runtime_error(llvm::toString(std::move(error)));
    }
  }
  const auto p = borrowParameters(impl_->parameters);
  if (qdmi::isBinaryProgramFormat(format)) {
    std::vector<std::span<const std::byte>> payloads;
    payloads.reserve(programs.size());
    for (const auto& program : programs) {
      payloads.push_back(std::as_bytes(std::span(program->payload())));
    }
    return impl_->device.trySubmitJob(payloads, format, shots, p[0], p[1], p[2],
                                      p[3], p[4]);
  }
  std::vector<std::string> payloads;
  payloads.reserve(programs.size());
  for (const auto& program : programs) {
    payloads.push_back(program->payload());
  }
  return impl_->device.trySubmitJob(payloads, format, shots, p[0], p[1], p[2],
                                    p[3], p[4]);
}

std::vector<int8_t> PreparedProgram::samples(const qdmi::Job& job, size_t shots,
                                             size_t programIndex) const {
  const std::scoped_lock lock(impl_->context->impl_->mutex);
  const auto result = readout(job, qubits(), shots, programIndex);
  std::vector<int8_t> values;
  values.reserve(product(shots, qubits()));
  for (const auto& [bits, count] : result.outcomes) {
    for (size_t i = 0; i < count; ++i) {
      for (const char bit : bits) {
        values.push_back(static_cast<int8_t>(bit - '0'));
      }
    }
  }
  return values;
}

struct QDMIDevice::Impl {
  using Observable = std::vector<size_t>;
  std::shared_ptr<Context> context;
  std::shared_ptr<mlir::MLIRContext> compiler;
  mlir::OwningOpRef<mlir::ModuleOp> recording;
  std::unique_ptr<mlir::qc::QCProgramBuilder> builder;
  llvm::SmallVector<mlir::Value> qubits;
  std::vector<QubitIdType> ids;
  QubitIdType nextQubit = 0;
  std::vector<Observable> observables;
  ObsIdType firstObservable = 0;
  ObsIdType nextObservable = 0;
  size_t shots = 0;
  std::shared_ptr<PreparedProgram> prepared;
  std::optional<Readout> result;

  explicit Impl(std::shared_ptr<Context> input) : context(std::move(input)) {
    require(context != nullptr, "Missing QDMI context.");
  }

  [[nodiscard]] std::vector<size_t>
  indices(const std::vector<QubitIdType>& wires) const {
    require(!ids.empty(), "No active QDMI qubit register.");
    std::vector<size_t> result;
    result.reserve(wires.size());
    for (const auto wire : wires) {
      require(wire >= ids.front() &&
                  static_cast<uint64_t>(wire - ids.front()) < ids.size(),
              "Invalid or released QDMI qubit ID.");
      const auto index = static_cast<size_t>(wire - ids.front());
      require(std::ranges::find(result, index) == result.end(),
              "Duplicate QDMI qubit operand.");
      result.push_back(index);
    }
    return result;
  }

  [[nodiscard]] const Observable& observable(ObsIdType id) const {
    require(id >= firstObservable &&
                static_cast<uint64_t>(id - firstObservable) <
                    observables.size(),
            "Invalid or released QDMI observable ID.");
    return observables[static_cast<size_t>(id - firstObservable)];
  }

  ObsIdType addObservable(Observable value) {
    require(nextObservable != INTPTR_MAX,
            "QDMI observable ID space exhausted.");
    observables.push_back(std::move(value));
    return nextObservable++;
  }

  const Readout& read(QDMIDevice& owner) {
    validateShots(shots);
    if (!result) {
      auto program = owner.prepare();
      auto job = program->submit(shots);
      const std::scoped_lock lock(context->impl_->mutex);
      require(job.wait() && job.check() == QDMI_JOB_STATUS_DONE,
              "QDMI job did not complete successfully.");
      result = readout(job, ids.size(), shots, 0);
    }
    return *result;
  }
};

QDMIDevice::QDMIDevice(std::shared_ptr<Context> context)
    : impl_(std::make_unique<Impl>(std::move(context))) {}
QDMIDevice::~QDMIDevice() = default;

std::vector<QubitIdType> QDMIDevice::AllocateQubits(size_t count) {
  require(impl_->ids.empty(), "Dynamic QDMI qubit allocation is unsupported.");
  require(count > 0 && count <= impl_->context->qubits() &&
              count <= static_cast<size_t>(INTPTR_MAX) -
                           static_cast<size_t>(impl_->nextQubit),
          "QDMI qubit allocation exceeds the device capacity.");
  impl_->compiler = mlir::createCompilerContext();
  impl_->builder =
      std::make_unique<mlir::qc::QCProgramBuilder>(impl_->compiler.get());
  impl_->builder->initialize();
  impl_->recording = impl_->builder->getInsertionBlock()
                         ->getParentOp()
                         ->getParentOfType<mlir::ModuleOp>();
  auto reg = impl_->builder->allocQubitRegister(static_cast<int64_t>(count));
  impl_->qubits = std::move(reg.qubits);
  impl_->ids.resize(count);
  std::iota(impl_->ids.begin(), impl_->ids.end(), impl_->nextQubit);
  impl_->nextQubit += static_cast<QubitIdType>(count);
  return impl_->ids;
}

void QDMIDevice::ReleaseQubits(const std::vector<QubitIdType>& wires) {
  const auto indices = impl_->indices(wires);
  require(indices.size() == impl_->ids.size(),
          "Partial QDMI qubit release is unsupported.");
  impl_->builder.reset();
  impl_->recording = nullptr;
  impl_->qubits.clear();
  impl_->ids.clear();
  impl_->prepared.reset();
  impl_->result.reset();
  impl_->observables.clear();
  impl_->firstObservable = impl_->nextObservable;
  impl_->compiler.reset();
}

size_t QDMIDevice::GetNumQubits() const { return impl_->ids.size(); }
void QDMIDevice::SetDeviceShots(size_t shots) {
  validateShots(shots);
  if (impl_->shots != shots) {
    impl_->result.reset();
  }
  impl_->shots = shots;
}
size_t QDMIDevice::GetDeviceShots() const { return impl_->shots; }
void QDMIDevice::SetDevicePRNG(std::mt19937* generator) {
  require(generator == nullptr, "Catalyst PRNG seeding is unsupported; use the "
                                "QDMI provider's job parameters.");
}

void QDMIDevice::NamedOperation(
    const std::string& name, const std::vector<double>& parameters,
    const std::vector<QubitIdType>& wires, bool inverse,
    const std::vector<QubitIdType>& controls,
    const std::vector<bool>& controlValues,
    const std::vector<std::string>& optionalParameters) {
  require(impl_->builder != nullptr,
          "QDMI gates require an active register before terminal measurement.");
  require(optionalParameters.empty(),
          "Optional Catalyst gate parameters are unsupported.");
  require(controls.size() == controlValues.size(),
          "Control wires and values have different lengths.");
  const auto* const gate = std::ranges::find(GATES, name, &NamedGate::name);
  require(gate != GATES.end(), "Unsupported QDMI gate: " + name);
  const auto& descriptor = mlir::qc::getStandardGateDescriptor(gate->gate);
  require(parameters.size() == descriptor.parameterCount &&
              (gate->gate == Gate::GPhase || gate->gate == Gate::Id ||
               wires.size() == descriptor.targetCount + gate->controls),
          "Invalid parameter or wire count for " + name + '.');
  require(std::ranges::all_of(
              parameters, [](double value) { return std::isfinite(value); }),
          "Non-finite parameter for " + name + '.');
  auto operands = controls;
  operands.insert(operands.end(), wires.begin(), wires.end());
  const auto indices = impl_->indices(operands);
  if (gate->gate == Gate::Id) {
    return;
  }
  auto& builder = *impl_->builder;
  llvm::SmallVector<mlir::Value> controlQubits;
  llvm::SmallVector<mlir::Value> targets;
  llvm::SmallVector<mlir::Value> angles;
  for (size_t i = 0; i < indices.size(); ++i) {
    (i < controls.size() + gate->controls ? controlQubits : targets)
        .push_back(impl_->qubits[indices[i]]);
  }
  if (gate->gate == Gate::GPhase) {
    targets.clear();
  }
  for (const double angle : parameters) {
    angles.push_back(
        builder.floatConstant(gate->gate == Gate::GPhase ? -angle : angle));
  }
  const auto primitive = [&](mlir::ValueRange values) {
    require(mlir::succeeded(mlir::qc::emitStandardGate(
                builder, builder.getLoc(), gate->gate, angles, values)),
            "Cannot emit QDMI gate " + name + '.');
  };
  const auto emit = [&](mlir::ValueRange values) {
    if (inverse) {
      builder.inv(values, primitive);
    } else {
      primitive(values);
    }
  };
  for (size_t i = 0; i < controls.size(); ++i) {
    if (!controlValues[i]) {
      builder.x(controlQubits[i]);
    }
  }
  if (controlQubits.empty()) {
    emit(targets);
  } else {
    builder.ctrl(controlQubits, targets, emit);
  }
  for (size_t i = 0; i < controls.size(); ++i) {
    if (!controlValues[i]) {
      builder.x(controlQubits[i]);
    }
  }
}

Result QDMIDevice::Measure(QubitIdType, std::optional<int32_t>) {
  throw std::runtime_error(
      "Interactive mid-circuit QDMI measurement is unsupported.");
}

std::shared_ptr<PreparedProgram> QDMIDevice::prepare() {
  if (impl_->prepared) {
    return impl_->prepared;
  }
  require(impl_->builder != nullptr,
          "No QDMI circuit is available for preparation.");
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      impl_->compiler.get(), [&](mlir::Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        stream << '\n';
        return mlir::success();
      });
  auto& builder = *impl_->builder;
  auto bits = builder.allocClassicalBitRegister(
      static_cast<int64_t>(impl_->qubits.size()), "c");
  for (size_t i = 0; i < impl_->qubits.size(); ++i) {
    builder.measure(impl_->qubits[i], bits, static_cast<int64_t>(i));
  }
  builder.retype(bits.getType());
  static_cast<void>(impl_->recording.release());
  auto moduleOp = builder.finalize({bits});
  impl_->builder.reset();
  auto program =
      mlir::QCProgram::fromModule(impl_->compiler, std::move(moduleOp));
  require(mlir::succeeded(program),
          "Invalid recorded QDMI circuit: " + diagnostics);
  auto compiled = mlir::CompiledProgram::compile(
      std::move(*program), impl_->context->impl_->environment);
  if (!compiled) {
    throw std::runtime_error(llvm::toString(compiled.takeError()) + '\n' +
                             diagnostics);
  }
  impl_->prepared = std::shared_ptr<PreparedProgram>(
      new PreparedProgram(std::make_unique<PreparedProgram::Impl>(
          impl_->context, std::move(*compiled), GetNumQubits())));
  return impl_->prepared;
}

ObsIdType
QDMIDevice::Observable(ObsId id,
                       const std::vector<std::complex<double>>& matrix,
                       const std::vector<QubitIdType>& wires) {
  auto indices = impl_->indices(wires);
  require(matrix.empty() && (id == ObsId::Identity || id == ObsId::PauliZ),
          "QDMI supports only Identity and PauliZ native observables.");
  require(id == ObsId::Identity || wires.size() == 1,
          "PauliZ requires exactly one wire.");
  if (id == ObsId::Identity) {
    indices.clear();
  }
  return impl_->addObservable(std::move(indices));
}

ObsIdType
QDMIDevice::TensorObservable(const std::vector<ObsIdType>& observables) {
  require(!observables.empty(), "An observable product cannot be empty.");
  Impl::Observable product;
  for (const auto id : observables) {
    const auto& value = impl_->observable(id);
    product.insert(product.end(), value.begin(), value.end());
  }
  return impl_->addObservable(std::move(product));
}

void QDMIDevice::Sample(DataView<double, 2>& samples) {
  PartialSample(samples, impl_->ids);
}

void QDMIDevice::PartialSample(DataView<double, 2>& samples,
                               const std::vector<QubitIdType>& wires) {
  const auto indices = impl_->indices(wires);
  require(samples.size() == product(impl_->shots, indices.size()),
          "Unexpected Catalyst sample buffer size.");
  const auto& result = impl_->read(*this);
  auto output = samples.begin();
  for (const auto& [bits, count] : result.outcomes) {
    for (size_t shot = 0; shot < count; ++shot) {
      for (const auto wire : indices) {
        *output++ = static_cast<double>(bits[wire] - '0');
      }
    }
  }
}

void QDMIDevice::Counts(DataView<double, 1>& eigenvalues,
                        DataView<int64_t, 1>& counts) {
  PartialCounts(eigenvalues, counts, impl_->ids);
}

void QDMIDevice::PartialCounts(DataView<double, 1>& eigenvalues,
                               DataView<int64_t, 1>& counts,
                               const std::vector<QubitIdType>& wires) {
  const auto indices = impl_->indices(wires);
  require(indices.size() <= std::numeric_limits<double>::digits,
          "Catalyst count keys cannot exactly represent more than 53 qubits; "
          "use samples.");
  const auto size = dimension(indices.size());
  require(eigenvalues.size() == size && counts.size() == size,
          "Unexpected Catalyst counts buffer size.");
  const auto& result = impl_->read(*this);
  for (size_t i = 0; i < size; ++i) {
    eigenvalues(i) = static_cast<double>(i);
    counts(i) = 0;
  }
  for (const auto& [bits, count] : result.outcomes) {
    counts(basisIndex(bits, indices)) += static_cast<int64_t>(count);
  }
}

void QDMIDevice::Probs(DataView<double, 1>& probabilities) {
  PartialProbs(probabilities, impl_->ids);
}

void QDMIDevice::PartialProbs(DataView<double, 1>& probabilities,
                              const std::vector<QubitIdType>& wires) {
  const auto indices = impl_->indices(wires);
  require(probabilities.size() == dimension(indices.size()),
          "Unexpected Catalyst probability buffer size.");
  const auto& result = impl_->read(*this);
  for (auto& probability : probabilities) {
    probability = 0.0;
  }
  for (const auto& [bits, count] : result.outcomes) {
    probabilities(basisIndex(bits, indices)) +=
        static_cast<double>(count) / static_cast<double>(impl_->shots);
  }
}

double QDMIDevice::Expval(ObsIdType observable) {
  const auto& parity = impl_->observable(observable);
  const auto& result = impl_->read(*this);
  size_t negative = 0;
  for (const auto& [bits, count] : result.outcomes) {
    bool odd = false;
    for (const auto wire : parity) {
      odd = odd != (bits[wire] == '1');
    }
    if (odd) {
      negative += count;
    }
  }
  return 1.0 - 2.0 * static_cast<double>(negative) /
                   static_cast<double>(impl_->shots);
}

double QDMIDevice::Var(ObsIdType observable) {
  const auto mean = Expval(observable);
  return std::max(0.0, 1.0 - mean * mean);
}

} // namespace mqt::catalyst

extern "C" Catalyst::Runtime::QuantumDevice*
MQTQDMIFactory(const char* kwargs) {
  const auto id = mqt::catalyst::parseContext(kwargs);
  auto& registry = mqt::catalyst::registry();
  std::shared_ptr<mqt::catalyst::Context> context;
  {
    const std::scoped_lock lock(registry.mutex);
    if (const auto entry = registry.contexts.find(id);
        entry != registry.contexts.end()) {
      context = entry->second.lock();
    }
  }
  mqt::catalyst::require(context != nullptr,
                         "Unknown or expired QDMI context ID.");
  return new mqt::catalyst::QDMIDevice(std::move(context));
}
