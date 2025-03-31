// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#pragma once

include "mlir/Pass/PassBase.td"

def MQTOptToCatalystQuantum : Pass<"mqtopt-to-quantum"> {
  let summary = "Lower `mqtopt` to `quantum` dialect.";

  let description = [{
    This pass lowers the `mqtopt` to `quantum` dialect.
  }];
  let dependentDialects = [
    "catalyst::quantum::CatalystQuantumDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}

#endif  // LIB_CONVERSION_MQTOPTTOQUANTUM
