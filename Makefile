.PHONY: plugin

CATALYST_BASE_DIR := /Users/patrickhopf/Code/catalyst/mlir
CATALYST_BUILD_DIR := $(CATALYST_BASE_DIR)/build
LLVM_BUILD_DIR := $(CATALYST_BASE_DIR)/llvm-project/build
LLVM_EXTERNAL_LIT := $(LLVM_BUILD_DIR)/bin/llvm-lit
CATALYST_TOOLS_DIR := $(LLVM_BUILD_DIR)/bin

plugin:
	cmake -B build -G Ninja \
		-DCatalyst_DIR=$(CATALYST_BUILD_DIR)/lib/cmake/catalyst \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=$(LLVM_EXTERNAL_LIT) \
		-DBUILD_MQT_CORE_MLIR=ON \


	cmake --build build
	cmake --build build --target quantum-opt
