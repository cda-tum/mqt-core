set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MQTOpt)
add_public_tablegen_target(MLIRMQTOptTransformsIncGen)

add_mlir_doc(Passes MQTOptPasses ./ -gen-pass-doc)