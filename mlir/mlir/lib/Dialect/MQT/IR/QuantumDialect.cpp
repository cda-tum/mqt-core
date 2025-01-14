// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/MQT/QuantumDialect.h"

#include "mlir/Dialect/MQT/QuantumOps.h"
#include "mlir/IR/DialectImplementation.h" // needed for generated type parser

#include "llvm/ADT/TypeSwitch.h" // needed for generated type parser

using namespace mlir;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// Quantum dialect definitions.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQT/QuantumOpsDialect.cpp.inc"

void QuantumDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQT/QuantumOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MQT/QuantumAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQT/QuantumOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Quantum type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQT/QuantumOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MQT/QuantumAttributes.cpp.inc"
