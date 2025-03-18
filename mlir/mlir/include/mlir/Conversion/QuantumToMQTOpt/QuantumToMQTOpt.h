#ifndef LIB_CONVERSION_QUANTUMTOMQTOPT_QUANTUMTOMQTOPT_H_
#define LIB_CONVERSION_QUANTUMTOMQTOPT_QUANTUMTOMQTOPT_H_

#include "mlir/Pass/Pass.h" // from @llvm-project

namespace mlir::mqt::ir::conversions {

#define GEN_PASS_DECL
#include "mlir/Conversion/QuantumToMQTOpt/QuantumToMQTOpt.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/QuantumToMQTOpt/QuantumToMQTOpt.h.inc"

} // namespace mlir::mqt::ir::conversions

#endif // LIB_CONVERSION_QUANTUMTOMQTOPT_QUANTUMTOMQTOPT_H_
