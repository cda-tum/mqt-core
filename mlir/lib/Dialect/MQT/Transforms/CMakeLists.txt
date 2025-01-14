set(LLVM_TARGET_DEFINITIONS ThePass.td)
mlir_tablegen(ThePass.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRMQTThePassIncGen)

add_mlir_library(MLIRMQTThePass ThePass.cpp LINK_LIBS MLIRMQT DEPENDS MLIRMQTThePassIncGen)
target_include_directories(MLIRMQTThePass PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
