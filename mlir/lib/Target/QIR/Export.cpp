/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Target/QIR/Export.h"

#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <system_error>

namespace mlir::qir {

[[nodiscard]] static std::unique_ptr<llvm::Module>
translateToLLVM(ModuleOp mod, llvm::LLVMContext& context) {
  auto llvmModule = translateModuleToLLVMIR(mod, context);
  if (!llvmModule) {
    mod.emitError("failed to translate QIR MLIR to LLVM IR");
    return nullptr;
  }
  qir::normalizeQIRModuleFlags(*llvmModule);
  if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
    mod.emitError("exported QIR failed LLVM IR verification");
    return nullptr;
  }
  return llvmModule;
}

LogicalResult translateQIRToLLVMIR(ModuleOp moduleOp,
                                   llvm::raw_ostream& output) {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(moduleOp, context);
  if (!llvmModule) {
    return failure();
  }
  llvmModule->print(output, nullptr);
  return success();
}

LogicalResult translateQIRToBitcode(ModuleOp moduleOp,
                                    llvm::raw_ostream& output) {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(moduleOp, context);
  if (!llvmModule) {
    return failure();
  }
  llvm::WriteBitcodeToFile(*llvmModule, output);
  return success();
}

LogicalResult writeQIRBitcode(ModuleOp moduleOp, StringRef filename) {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(moduleOp, context);
  if (!llvmModule) {
    return failure();
  }

  std::error_code error;
  llvm::raw_fd_ostream stream(filename, error, llvm::sys::fs::OF_None);
  if (error) {
    moduleOp.emitError() << "failed to open bitcode output file '" << filename
                         << "': " << error.message();
    return failure();
  }
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  stream.flush();
  if (stream.has_error()) {
    stream.clear_error();
    moduleOp.emitError() << "failed to write bitcode file '" << filename << "'";
    return failure();
  }
  return success();
}

} // namespace mlir::qir
