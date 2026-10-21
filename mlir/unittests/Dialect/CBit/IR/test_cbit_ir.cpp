/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_cbit_ir.cpp
/// Unit tests for the CBit MLIR dialect.

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h" // IWYU pragma: keep
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {
class CBitIRTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect,
                    mqt::MQTDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(const StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }
};

TEST_F(CBitIRTest, ParsesAndPrintsRegisterOperations) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() -> !cbit.reg<2> {
        %c0 = arith.constant 0 : index
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<2>
        cbit.store %false, %reg[%c0] : !cbit.reg<2>
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        %value = cbit.read %reg : !cbit.reg<2> -> i2
        cbit.write %value, %reg : i2, !cbit.reg<2>
        return %reg : !cbit.reg<2>
      }
    }
  )mlir");

  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(verify(*moduleOp).succeeded());

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  moduleOp->print(stream);
  EXPECT_NE(
      printed.find("cbit.alloc(#cbit.init<zero>) {mqt.register_name = \"c\"}"),
      std::string::npos)
      << printed;
  EXPECT_NE(printed.find("!cbit.reg<2>"), std::string::npos);
  EXPECT_NE(printed.find("cbit.store"), std::string::npos);
  EXPECT_NE(printed.find("cbit.load"), std::string::npos);
  EXPECT_NE(printed.find("cbit.read"), std::string::npos);
  EXPECT_NE(printed.find("%b = cbit.load"), std::string::npos) << printed;
  EXPECT_NE(printed.find("%val = cbit.read"), std::string::npos) << printed;
  EXPECT_NE(printed.find("cbit.write"), std::string::npos);
  EXPECT_TRUE(parse(printed));
}

TEST_F(CBitIRTest, PrintsSourceRegisterNames) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %first = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "result"} : !cbit.reg<2>
        %second = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "7 bits"} : !cbit.reg<2>
        %third = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  moduleOp->print(stream);
  EXPECT_NE(printed.find("%result = cbit.alloc"), std::string::npos) << printed;
  EXPECT_NE(printed.find("%_7_bits = cbit.alloc"), std::string::npos)
      << printed;
  EXPECT_NE(printed.find("%c = cbit.alloc"), std::string::npos) << printed;
  EXPECT_TRUE(parse(printed));
}

TEST_F(CBitIRTest, RejectsReadWidthMismatch) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %value = cbit.read %reg : !cbit.reg<2> -> i3
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsUnsignedIntegerType) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %value = "cbit.read"(%reg) : (!cbit.reg<2>) -> ui2
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsWriteWidthMismatch) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %value = arith.constant 0 : i3
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.write %value, %reg : i3, !cbit.reg<2>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsNonPositiveRegisterWidth) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %reg = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<0>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsConstantOutOfBoundsIndex) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c2 = arith.constant 2 : index
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %bit = cbit.load %reg[%c2] : !cbit.reg<2>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsNegativeConstantIndex) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %neg = arith.constant -1 : index
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %bit = cbit.load %reg[%neg] : !cbit.reg<2>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsInvalidOperandTypes) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %bit = "cbit.load"(%reg, %false)
            : (!cbit.reg<1>, i1) -> i1
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, ReportsMemoryEffects) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %c0 = arith.constant 0 : index
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        cbit.store %false, %reg[%c0] : !cbit.reg<1>
        %bit = cbit.load %reg[%c0] : !cbit.reg<1>
        %value = cbit.read %reg : !cbit.reg<1> -> i1
        cbit.write %value, %reg : i1, !cbit.reg<1>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  cbit::AllocOp alloc;
  cbit::LoadOp load;
  cbit::ReadOp read;
  cbit::StoreOp store;
  cbit::WriteOp write;
  moduleOp->walk([&](cbit::AllocOp op) { alloc = op; });
  moduleOp->walk([&](cbit::LoadOp op) { load = op; });
  moduleOp->walk([&](cbit::ReadOp op) { read = op; });
  moduleOp->walk([&](cbit::StoreOp op) { store = op; });
  moduleOp->walk([&](cbit::WriteOp op) { write = op; });

  ASSERT_NE(alloc.getOperation(), nullptr);
  ASSERT_NE(load.getOperation(), nullptr);
  ASSERT_NE(read.getOperation(), nullptr);
  ASSERT_NE(store.getOperation(), nullptr);
  ASSERT_NE(write.getOperation(), nullptr);

  SmallVector<MemoryEffects::EffectInstance> effects;
  alloc.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Allocate>(effects.front().getEffect()));

  effects.clear();
  load.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Read>(effects.front().getEffect()));
  EXPECT_EQ(effects.front().getValue(), load.getReg());

  effects.clear();
  read.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Read>(effects.front().getEffect()));
  EXPECT_EQ(effects.front().getValue(), read.getReg());

  effects.clear();
  store.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Write>(effects.front().getEffect()));
  EXPECT_EQ(effects.front().getValue(), store.getReg());

  effects.clear();
  write.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Write>(effects.front().getEffect()));
  EXPECT_EQ(effects.front().getValue(), write.getReg());
}

TEST_F(CBitIRTest, RemovesWriteOnlyRegistersAndPreservesEffectfulProducers) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() -> !cbit.reg<1> {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %temporary = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        %bit = func.call @produce() : () -> i1
        cbit.store %bit, %temporary[%c0] : !cbit.reg<1>
        cbit.write %bit, %temporary : i1, !cbit.reg<1>
        %result = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        cbit.store %true, %result[%c0] : !cbit.reg<1>
        return %result : !cbit.reg<1>
      }
      func.func private @produce() -> i1
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto funcOp = *moduleOp->getOps<func::FuncOp>().begin();
  EXPECT_TRUE(llvm::hasSingleElement(funcOp.getOps<cbit::AllocOp>()));
  EXPECT_TRUE(llvm::hasSingleElement(funcOp.getOps<cbit::StoreOp>()));
  EXPECT_TRUE(funcOp.getOps<cbit::WriteOp>().empty());
  ASSERT_TRUE(llvm::hasSingleElement(funcOp.getOps<func::CallOp>()));
  auto producer = *funcOp.getOps<func::CallOp>().begin();
  EXPECT_TRUE(producer.getResult(0).use_empty());
}

TEST_F(CBitIRTest, ForwardsStraightLineStoresAndZeroInitialization) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() -> (i1, i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %zero = cbit.load %reg[%c0] : !cbit.reg<2>
        cbit.store %true, %reg[%c1] : !cbit.reg<2>
        %stored = cbit.load %reg[%c1] : !cbit.reg<2>
        return %zero, %stored : i1, i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

  auto funcOp = *moduleOp->getOps<func::FuncOp>().begin();
  EXPECT_TRUE(funcOp.getOps<cbit::AllocOp>().empty());
  EXPECT_TRUE(funcOp.getOps<cbit::StoreOp>().empty());
  auto returnOp = *funcOp.getOps<func::ReturnOp>().begin();
  std::string canonicalized;
  llvm::raw_string_ostream canonicalizedStream(canonicalized);
  moduleOp->print(canonicalizedStream);
  APInt zero;
  APInt stored;
  EXPECT_TRUE(matchPattern(returnOp.getOperand(0), m_ConstantInt(&zero)))
      << canonicalized;
  EXPECT_TRUE(matchPattern(returnOp.getOperand(1), m_ConstantInt(&stored)))
      << canonicalized;
  EXPECT_TRUE(zero.isZero());
  EXPECT_TRUE(stored.isOne());
}

TEST_F(CBitIRTest, DoesNotForwardAcrossAnAmbiguousStore) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main(%dynamic: index) -> (i1, i1) {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.store %true, %reg[%dynamic] : !cbit.reg<2>
        %value = cbit.load %reg[%c0] : !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        %expected = arith.constant 0 : i2
        %matches = arith.cmpi eq, %snapshot, %expected : i2
        return %value, %matches : i1, i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

  auto funcOp = *moduleOp->getOps<func::FuncOp>().begin();
  auto returnOp = *funcOp.getOps<func::ReturnOp>().begin();
  EXPECT_TRUE(returnOp.getOperand(0).getDefiningOp<cbit::LoadOp>());
  EXPECT_TRUE(returnOp.getOperand(1).getDefiningOp<arith::CmpIOp>());
}

TEST_F(CBitIRTest, ForwardsKnownValuesAcrossPureOperations) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @load(%value: i1, %other: i1, %reg: !cbit.reg<2>) -> (i1, i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        cbit.store %value, %reg[%c0] : !cbit.reg<2>
        cbit.store %other, %reg[%c1] : !cbit.reg<2>
        %extra = arith.xori %value, %other : i1
        %loaded = cbit.load %reg[%c0] : !cbit.reg<2>
        return %loaded, %extra : i1, i1
      }
      func.func @read(%value: i65, %other: i65, %reg: !cbit.reg<65>) -> (i65, i65) {
        cbit.write %value, %reg : i65, !cbit.reg<65>
        %extra = arith.xori %value, %other : i65
        %loaded = cbit.read %reg : !cbit.reg<65> -> i65
        return %loaded, %extra : i65, i65
      }
      func.func @single_bit(%value: i1, %other: i1, %reg: !cbit.reg<1>) -> (i1, i1, i1) {
        %c0 = arith.constant 0 : index
        cbit.store %value, %reg[%c0] : !cbit.reg<1>
        %extra = arith.xori %value, %other : i1
        %snapshot = cbit.read %reg : !cbit.reg<1> -> i1
        cbit.write %other, %reg : i1, !cbit.reg<1>
        %loaded = cbit.load %reg[%c0] : !cbit.reg<1>
        return %snapshot, %loaded, %extra : i1, i1, i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  for (auto funcOp : moduleOp->getOps<func::FuncOp>()) {
    SCOPED_TRACE(funcOp.getSymName().str());
    auto returnOp = cast<func::ReturnOp>(funcOp.front().getTerminator());
    EXPECT_EQ(returnOp.getOperand(0), funcOp.getArgument(0));
    EXPECT_TRUE(funcOp.getOps<cbit::LoadOp>().empty());
    EXPECT_TRUE(funcOp.getOps<cbit::ReadOp>().empty());
    if (funcOp.getSymName() == "single_bit") {
      EXPECT_EQ(returnOp.getOperand(1), funcOp.getArgument(1));
    }
  }
}

TEST_F(CBitIRTest, ReusesLoadsAndReadsAcrossPureOperations) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @load(%reg: !cbit.reg<2>, %index: index, %other: i1) -> (i1, i1, i1) {
        %first = cbit.load %reg[%index] : !cbit.reg<2>
        %extra = arith.xori %first, %other : i1
        %second = cbit.load %reg[%index] : !cbit.reg<2>
        return %first, %second, %extra : i1, i1, i1
      }
      func.func @read(%reg: !cbit.reg<65>, %other: i65) -> (i65, i65, i65) {
        %first = cbit.read %reg : !cbit.reg<65> -> i65
        %extra = arith.xori %first, %other : i65
        %second = cbit.read %reg : !cbit.reg<65> -> i65
        return %first, %second, %extra : i65, i65, i65
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  for (auto funcOp : moduleOp->getOps<func::FuncOp>()) {
    auto returnOp = cast<func::ReturnOp>(funcOp.front().getTerminator());
    EXPECT_EQ(returnOp.getOperand(0), returnOp.getOperand(1));
  }
}

TEST_F(CBitIRTest, ForwardsZeroReadsWithoutDefiningUndefinedStorage) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @zero(%a: i65, %b: i65) -> (i65, i65) {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<65>
        %extra = arith.xori %a, %b : i65
        %value = cbit.read %reg : !cbit.reg<65> -> i65
        return %value, %extra : i65, i65
      }
      func.func @undefined() -> (i65, i1) {
        %c0 = arith.constant 0 : index
        %reg = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<65>
        %value = cbit.read %reg : !cbit.reg<65> -> i65
        %bit = cbit.load %reg[%c0] : !cbit.reg<65>
        return %value, %bit : i65, i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto zero = moduleOp->lookupSymbol<func::FuncOp>("zero");
  auto zeroReturn = cast<func::ReturnOp>(zero.front().getTerminator());
  EXPECT_TRUE(matchPattern(zeroReturn.getOperand(0), m_Zero()));
  auto undefined = moduleOp->lookupSymbol<func::FuncOp>("undefined");
  auto undefinedReturn =
      cast<func::ReturnOp>(undefined.front().getTerminator());
  EXPECT_TRUE(undefinedReturn.getOperand(0).getDefiningOp<cbit::ReadOp>());
  EXPECT_TRUE(undefinedReturn.getOperand(1).getDefiningOp<cbit::LoadOp>());
}

TEST_F(CBitIRTest, DoesNotForwardAcrossPotentialWrites) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func private @mutate(!cbit.reg<2>)
      func.func @call(%reg: !cbit.reg<2>, %value: i2) -> (i1, i2) {
        %c0 = arith.constant 0 : index
        %low = arith.trunci %value : i2 to i1
        cbit.store %low, %reg[%c0] : !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        func.call @mutate(%reg) : (!cbit.reg<2>) -> ()
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        %whole = cbit.read %reg : !cbit.reg<2> -> i2
        %difference = arith.xori %whole, %snapshot : i2
        return %bit, %difference : i1, i2
      }
      func.func @region(%reg: !cbit.reg<2>, %condition: i1, %value: i2) -> (i1, i2) {
        %c0 = arith.constant 0 : index
        %low = arith.trunci %value : i2 to i1
        cbit.store %low, %reg[%c0] : !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        scf.if %condition {
          cbit.store %condition, %reg[%c0] : !cbit.reg<2>
        }
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        %whole = cbit.read %reg : !cbit.reg<2> -> i2
        %difference = arith.xori %whole, %snapshot : i2
        return %bit, %difference : i1, i2
      }
      func.func @other_register(%reg: !cbit.reg<2>, %other: !cbit.reg<2>, %value: i1, %changed: i1) -> (i1, i2) {
        %c0 = arith.constant 0 : index
        cbit.store %value, %reg[%c0] : !cbit.reg<2>
        %snapshot = cbit.read %reg : !cbit.reg<2> -> i2
        cbit.store %changed, %other[%c0] : !cbit.reg<2>
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        %whole = cbit.read %reg : !cbit.reg<2> -> i2
        %difference = arith.xori %whole, %snapshot : i2
        return %bit, %difference : i1, i2
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  for (auto funcOp : moduleOp->getOps<func::FuncOp>()) {
    if (funcOp.isDeclaration()) {
      continue;
    }
    auto returnOp = cast<func::ReturnOp>(funcOp.front().getTerminator());
    EXPECT_TRUE(returnOp.getOperand(0).getDefiningOp<cbit::LoadOp>());
    auto difference = returnOp.getOperand(1).getDefiningOp<arith::XOrIOp>();
    ASSERT_TRUE(difference);
    EXPECT_TRUE(difference.getLhs().getDefiningOp<cbit::ReadOp>());
    EXPECT_TRUE(difference.getRhs().getDefiningOp<cbit::ReadOp>());
  }
}
} // namespace
