# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QuantumToMQTOpt.td)
mlir_tablegen(QuantumToMQTOpt.h.inc -gen-pass-decls -name QuantumToMQTOpt)
add_public_tablegen_target(QuantumToMQTOptIncGen)

add_mlir_doc(QuantumToMQTOpt QuantumToMQTOpt ./ -gen-pass-doc)
