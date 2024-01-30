#include "operations/ClassicControlledOperation.hpp"
#include "python/pybind11.hpp"

namespace mqt {

void registerClassicControlledOperation(py::module& m) {
  auto ccop = py::class_<qc::ClassicControlledOperation, qc::Operation>(
      m, "ClassicControlledOperation",
      "Quantum operation controlled by classical results.");

  ccop.def(
      py::init([](qc::Operation* operation, qc::ClassicalRegister controlReg,
                  std::uint64_t expectedVal) {
        auto op = operation->clone();
        return std::make_unique<qc::ClassicControlledOperation>(op, controlReg,
                                                                expectedVal);
      }),
      "operation"_a, "control_register"_a, "expected_value"_a = 1U,
      "Applies operation if the control register has the expected value.");
  ccop.def_property_readonly(
      "operation", &qc::ClassicControlledOperation::getOperation,
      "Returns the operation controlled by the classical register.",
      py::return_value_policy::reference_internal);
  ccop.def_property_readonly(
      "control_register", &qc::ClassicControlledOperation::getControlRegister,
      "Returns the classical register controlling the operation.");
  ccop.def_property_readonly(
      "expected_value", &qc::ClassicControlledOperation::getExpectedValue,
      "Returns the expected value of the classical register.");
  ccop.def("__repr__", [](const qc::ClassicControlledOperation& op) {
    std::stringstream ss;
    const auto& controlReg = op.getControlRegister();
    ss << "ClassicControlledOperation(<...op...>, "
       << "control_register=(" << controlReg.first << ", " << controlReg.second
       << "), "
       << "expected_value=" << op.getExpectedValue() << ")";
    return ss.str();
  });
}

} // namespace mqt
