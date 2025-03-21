# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect_library(MLIRMQTOpt MQTOptOps.cpp DEPENDS MLIRMQTOptOpsIncGen
                         MLIRMQTOptInterfacesIncGen)
