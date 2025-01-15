set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MQT)
add_public_tablegen_target(MLIRMQTTransformsIncGen)

add_mlir_doc(Passes MQTPasses ./ -gen-pass-doc)
